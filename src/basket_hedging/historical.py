"""One historical ETF basket experiment using the existing engine and policies.

Daily OHLC is observed; execution spread/depth, release clocks and funding are
declared research assumptions. The nontraded basket has model marks, not quotes.
"""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo
import csv
import json

import numpy as np

from .contracts import DataGap, Funding, InputError, Instrument, Opening
from .engine import Engine
from .historical_hedge import PRICING_DESCRIPTION, SurfaceBasketValuator, reprice_basket_scenarios
from .learning_data import conditioning_features as gaussian_features
from .market import Borrow, Event, Fixing, Observation, Quote, Rules
from .optimization import SolverConvergenceError
from .policies import ScheduledDeltaPolicy, UnhedgedPolicy
from .replay import Case, jsonable, write_json
from .risk import UnnettedLiabilityStressMargin
from .scenario_hedge import ScenarioHedge
from .surfaces import IVSurface, PAPER_GRID
from .valuation import StressScenario


TRAIN_END = "2017-12-29"
VALIDATION_START, VALIDATION_END = "2018-01-02", "2019-12-31"
TEST_START = "2020-01-02"
MODEL_AVAILABLE = datetime(2020, 1, 1, tzinfo=timezone.utc)
POLICIES = ("unhedged", "delta", "parametric", "gaussian", "volgan")


@dataclass(frozen=True)
class HistoricalConfig:
    holding_sessions: int = 20
    capital: float = 100000.0
    basket_initial_value: float = 300.0
    multiplier: float = 100.0
    half_spread: float = 0.0005
    quote_size: float = 10000.0
    borrow_rate: float = 0.01
    safety_buffer: float = 1000.0
    alpha: float = 0.1
    scenarios: int = 256
    pricing_samples: int = 512
    pricing_seed: int = 731
    scenario_seed: int = 9173

    def __post_init__(self):
        if (self.holding_sessions < 1 or self.scenarios < 2 or self.pricing_samples < 2
                or self.capital <= 0 or self.basket_initial_value <= 0 or self.multiplier <= 0
                or not 0 <= self.half_spread < 1 or self.quote_size <= 0
                or min(self.borrow_rate, self.safety_buffer, self.alpha) < 0
                or not all(np.isfinite(value) for value in asdict(self).values())):
            raise InputError("invalid historical experiment configuration")


def session_time(date: str, *, opening: bool = False) -> datetime:
    """Assumed EOD release/open clocks, not vendor-certified quote timestamps."""
    day = datetime.fromisoformat(str(date))
    return day.replace(hour=9 if opening else 16, minute=30 if opening else 0,
                       tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)


def validate_corporate_actions(arrays: dict) -> None:
    """Unknown distributions cannot be replaced by zero or adjusted-close returns."""
    for key in ("dividends", "splits"):
        values = arrays[key]
        if not np.isfinite(values).all():
            symbols = list(map(str, arrays["symbols"][~np.isfinite(values).all(axis=0)]))
            raise DataGap(f"unknown {key} for {symbols}; financial benchmark blocked")
    if np.any(arrays["dividends"] < 0):
        raise InputError("negative cash distribution")
    if np.any(arrays["splits"] != 1):
        raise DataGap("stock splits require explicit contract/position adjustments; benchmark blocked")


def training_dynamics(arrays: dict) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Frozen joint calendar-day GBM moments, fitted only through TRAIN_END."""
    indices = np.where(arrays["dates"] <= TRAIN_END)[0]
    returns, durations = [], []
    for index in indices[1:]:
        if arrays["consecutive"][index]:
            returns.append(np.log(arrays["close"][index] / arrays["close"][index - 1]))
            durations.append((session_time(arrays["dates"][index])
                              - session_time(arrays["dates"][index - 1])).total_seconds() / 86400)
    if len(returns) < 30:
        raise InputError("insufficient pre-test returns for frozen joint calibration")
    returns, durations = np.array(returns), np.array(durations)
    mean = returns.sum(axis=0) / durations.sum()
    normalized = (returns - mean * durations[:, None]) / np.sqrt(durations[:, None])
    covariance = np.cov(normalized, rowvar=False, ddof=1)
    sd = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(sd, sd)
    # Construct an exactly symmetric record from the symmetric sample estimator.
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1)
    return mean, covariance, tuple(tuple(map(float, row)) for row in correlation)


def expand_calendar(arrays: dict, manifest: dict) -> dict:
    """Keep observed exchange sessions with missing joint data in the contract clock.

    Expiry and rebalancing dates cannot depend on which future option rows happen
    to be present. Missing sessions become visible data-gap events, never longer
    contracts or compressed one-session GAN transitions.
    """
    missing = {day for gap in manifest.get("common_date_gaps", []) for day in gap["missing_dates"]}
    calendar = np.array(sorted(set(arrays["dates"]) | missing))
    locations = np.searchsorted(calendar, arrays["dates"])
    result = {}
    for key, value in arrays.items():
        if key == "dates":
            result[key] = calendar
        elif key in {"open", "close", "dividends", "splits", "iv", "valid_surface", "consecutive"}:
            fill = False if value.dtype.kind == "b" else np.nan
            result[key] = np.full((len(calendar), *value.shape[1:]), fill, dtype=value.dtype)
            result[key][locations] = value
        else:
            result[key] = value.copy()
    return result


def historical_context(arrays: dict, index: int, horizon: datetime) -> dict:
    """Only the released prefix enters conditioning; neither labels nor future IV."""
    from .volgan import conditioning_features as volgan_features

    if index < 21 or not arrays["consecutive"][index - 20:index + 1].all():
        raise DataGap("missing consecutive return history for conditioning")
    current = arrays["close"][index]
    iv = arrays["iv"][index]
    if not arrays["valid_surface"][index].all():
        raise DataGap("current joint IV surfaces unavailable")
    stocks = tuple(map(str, arrays["symbols"]))
    times = [session_time(date) for date in arrays["dates"][index - 21:index + 1]]
    history = tuple(Observation(f"close/{time.isoformat()}/{stock}", stock, float(price), time, time)
                    for time, row in zip(times, arrays["close"][index - 21:index + 1])
                    for stock, price in zip(stocks, row))
    spots = dict(zip(stocks, map(float, current)))
    features, _ = gaussian_features(stocks, history, spots, times[-1], horizon)
    returns = np.diff(np.log(arrays["close"][index - 21:index + 1]), axis=0)
    return dict(iv=iv.tolist(), gaussian_features=features.tolist(),
                volgan_features=volgan_features(returns, iv).tolist(),
                input_ids=[obs.id for obs in history] + [f"surface/{times[-1].isoformat()}/{stock}" for stock in stocks],
                state_available_at=times[-1].isoformat())


class HistoricalScenarios:
    """Joint scenario adapter; no replay arrays, iterator, future prices or quotes."""

    def __init__(self, kind, valuator, config, *, mean=None, covariance=None, model=None):
        if kind not in {"parametric", "gaussian", "volgan"}:
            raise InputError("unknown historical scenario provider")
        self.kind, self.valuator, self.config, self.model = kind, valuator, config, model
        self.mean, self.covariance = mean, covariance

    def generate(self, view, fixed_positions, hedge_ids):
        state = view.valuation.metadata.get("historical_state")
        if state is None:
            raise DataGap("joint scenario conditioning is unavailable")
        if datetime.fromisoformat(state["state_available_at"]) > view.time:
            raise InputError("unreleased historical conditioning state")
        horizon = view.rules["next_rebalance_time"]
        if horizon is None:
            raise DataGap("next trading-session close is missing")
        days = (horizon - view.time).total_seconds() / 86400
        stocks = self.valuator.parameters.stock_ids
        spots = np.array([view.valuation.marks[key] for key in stocks])
        iv = np.asarray(state["iv"])
        seed = int.from_bytes(sha256(f"historical/{self.config.scenario_seed}/{view.time.isoformat()}".
                                    encode()).digest()[:16], "big")
        random = np.random.default_rng(seed)
        if self.kind == "volgan":
            if self.model.available_at > view.time or tuple(self.model.asset_ids) != stocks:
                raise InputError("VolGAN checkpoint is unavailable or has different assets")
            future, surfaces = self.model.sample(np.asarray(state["volgan_features"]), spots,
                random.normal(size=(self.config.scenarios, self.model.config.noise_dim)))
        else:
            normals = random.normal(size=(self.config.scenarios, len(stocks)))
            if self.kind == "gaussian":
                if self.model.available_at > view.time or self.model.stock_ids != stocks:
                    raise InputError("Gaussian checkpoint is unavailable or has different assets")
                log_returns = self.model.sample_log_returns(np.asarray(state["gaussian_features"]), days, normals)
            else:
                log_returns = days * self.mean + np.sqrt(days) * normals @ np.linalg.cholesky(self.covariance).T
            future = spots * np.exp(log_returns)
            surfaces = np.broadcast_to(iv, (self.config.scenarios, *iv.shape))
        return reprice_basket_scenarios(view, fixed_positions, hedge_ids,
            future_spots=future, future_surfaces=surfaces, grid=PAPER_GRID, horizon=horizon,
            valuator=self.valuator, seed=seed, input_ids=tuple(state["input_ids"]))


def episode(arrays: dict, start: int, config: HistoricalConfig, valuator: SurfaceBasketValuator):
    """Precompute shared released packets; the engine still consumes them incrementally."""
    stocks = tuple(map(str, arrays["symbols"]))
    expiry_index = start + config.holding_sessions
    exit_index = expiry_index + 1
    if exit_index >= len(arrays["dates"]):
        raise InputError("episode needs expiry plus the subsequent opening observation")
    expiry = session_time(arrays["dates"][expiry_index])
    end = session_time(arrays["dates"][exit_index], opening=True)
    if not np.isfinite(arrays["close"][start]).all():
        raise DataGap("basket inception prices unavailable; liability not originated")
    coefficients = tuple((stock, config.basket_initial_value / len(stocks) / float(price))
                         for stock, price in zip(stocks, arrays["close"][start]))
    instruments = {key: Instrument(key, "stock") for key in stocks}
    instruments["B"] = Instrument("B", "basket_put", multiplier=config.multiplier,
        strike=config.basket_initial_value, expiry=expiry, coefficients=coefficients)
    opening = Opening("ORIGINATE", initial_capital=config.capital, liability_id="B", liability_quantity=-1)
    stresses = tuple(stress.name for stress in valuator.stress_scenarios)
    rules = Rules(end, stocks, config.safety_buffer,
        funding=Funding(borrow_rates={key: config.borrow_rate for key in stocks}),
        borrow={key: Borrow(True, config.quote_size) for key in stocks},
        expected_stress_scenarios=stresses, start_time=session_time(arrays["dates"][start]))
    events = []
    for index in range(start, exit_index + 1):
        phases = (False,) if index == start else (True,) if index == exit_index else (True, False)
        for is_open in phases:
            date = str(arrays["dates"][index])
            time = session_time(date, opening=is_open)
            label = f"{date}/{'open' if is_open else 'close'}"
            values = arrays["open" if is_open else "close"][index]
            if not np.isfinite(values).all():
                events.append(Event("missing-joint-session/" + label, time, 0))
                return instruments, opening, rules, tuple(events)
            spots = dict(zip(stocks, map(float, values)))
            surface_index = index - 1 if is_open else index
            surface_time = session_time(arrays["dates"][surface_index])
            observations = tuple(Observation(label + "/" + stock, stock, spots[stock], time, time) for stock in stocks)
            fixings = ((Fixing("fixing/B", "B", time, time, spots),) if time == expiry else ())
            packet = None
            if arrays["valid_surface"][surface_index].all() or index == exit_index:
                surfaces = ({stock: IVSurface(PAPER_GRID, arrays["iv"][surface_index, column],
                                surface_time, surface_time, "data-v1/vendor-IV/vega-kernel")
                             for column, stock in enumerate(stocks)} if index != exit_index else {})
                packet = valuator.packet(time, spots, instruments, surfaces,
                    input_ids=tuple(obs.id for obs in observations),
                    observed_at={key: time for key in instruments},
                    fixings={"B": spots} if fixings else None)
            decision = not is_open
            next_close = session_time(arrays["dates"][index + 1]) if decision else None
            if decision and index < expiry_index and packet is not None:
                metadata = dict(packet.metadata)
                try:
                    metadata["historical_state"] = historical_context(arrays, index, next_close)
                except DataGap as error:
                    metadata["conditioning_gap"] = str(error)
                packet = replace(packet, metadata=metadata)
            quotes = tuple(Quote("assumed-execution/" + label + "/" + stock, stock, time, time,
                                 spots[stock] * (1 - config.half_spread), spots[stock] * (1 + config.half_spread),
                                 config.quote_size, config.quote_size) for stock in stocks) if is_open else ()
            dividends = {stock: float(value) for stock, value in zip(stocks, arrays["dividends"][index])
                         if value} if is_open else {}
            events.append(Event(label, time, 0, packet, observations, quotes, fixings,
                decision=decision, execution=is_open,
                earliest_execution=session_time(arrays["dates"][index + 1], opening=True) if decision else None,
                order_deadline=next_close if index < expiry_index else end if decision else None,
                next_rebalance_time=next_close, terminal_decision=time == expiry,
                origination_price=packet.marks["B"] if index == start and packet is not None else None,
                dividends=dividends))
    return instruments, opening, rules, tuple(events)


def _write_rows(path: Path, rows: list[dict], *, headers: tuple[str, ...] = ()) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else headers)
        writer.writeheader()
        writer.writerows(rows)


def decision_schedule(arrays: dict, start: int, config: HistoricalConfig) -> list[dict]:
    """Reporting-only availability screen on the fixed contractual review dates."""
    schedule = []
    for index in range(start, start + config.holding_sessions + 1):
        day = str(arrays["dates"][index])
        terminal = index == start + config.holding_sessions
        issue = None
        if not terminal:
            if not np.isfinite(arrays["close"][index]).all():
                issue = "joint closing prices unavailable"
            else:
                try:
                    historical_context(arrays, index, session_time(arrays["dates"][index + 1]))
                except DataGap as error:
                    issue = str(error)
        schedule.append(dict(event_id=f"{day}/close", time=session_time(day).isoformat(),
                             terminal=terminal, conditioning_available=issue is None,
                             conditioning_issue=issue))
    return schedule


def decision_record(episode_id: str, policy: str, schedule: list[dict],
                    decisions: list[dict], status: str) -> dict:
    """Separate scheduled reviews, actual callbacks and reviews never reached."""
    scheduled_ids = {row["event_id"] for row in schedule}
    observed = [row for row in decisions if row["event_id"] in scheduled_ids]
    reached = {row["event_id"] for row in observed}
    skips = [dict(time=row["time"], reason=row["reason"]) for row in observed if row["status"] == "skipped"]
    return dict(episode=episode_id, policy=policy, status=status,
        scheduled_decisions=len(schedule), decisions_made=sum(row["status"] == "targets" for row in observed),
        decisions_skipped=len(skips), decisions_not_reached=len(scheduled_ids - reached),
        decisions_no_target=sum(row["status"] == "no_decision" for row in observed),
        sufficient_history_by_data=all(row["conditioning_available"] for row in schedule),
        skips=skips, scheduled_records=schedule, records=decisions)


def accounting_record(episode_id: str, policy: str, status: str,
                      valuation_time: datetime | None, components: dict) -> dict:
    """Costs remain known after a stop; stale marks cannot supply current P&L."""
    completed = status == "COMPLETED"
    unavailable = {"net_pnl", "liability_pnl", "hedge_pnl", "reconciliation_error"}
    return dict(episode=episode_id, policy=policy, status=status,
        last_valid_valuation_time=valuation_time.isoformat() if valuation_time else None,
        valuation_status="terminal_current" if completed else "unavailable_current", terminal=completed,
        **{key: value if completed or key not in unavailable else None for key, value in components.items()})


def _run_policy(engine: Engine, events, policy):
    """Retain a numerical failure without changing the engine or solver settings.

    FAILED_OPTIMIZER is a benchmark outcome, not an engine financial status.
    The unfinished account is neither liquidated nor advanced to maturity.
    """
    try:
        return engine.run(iter(events), policy), None
    except SolverConvergenceError as error:
        result = engine.result
        # Mandatory phases commit cloned ledgers; run() normally attaches the
        # current ledger on return, which this exception did not reach.
        result.ledger = ledger = engine.ledger
        failure = dict(status="FAILED_OPTIMIZER", reason=str(error),
            last_valid_checkpoint=result.last_valid_snapshot,
            baseline=ledger.baseline, cash=ledger.cash, positions=dict(ledger.positions),
            marks=dict(ledger.marks), posted_margin_cash=ledger.posted_margin_cash,
            restricted_short_proceeds=ledger.restricted_short_proceeds,
            available_cash=ledger.available_cash, short_lots=ledger.short_lots,
            pending_orders=engine.orders.snapshot(), postings=ledger.postings,
            settlements=ledger.settlements, order_outcomes=result.order_outcomes,
            decisions=result.decisions, terminal_pnl=None, terminal_hedge_error=None)
        return result, failure


def summarize(rows: list[dict]) -> list[dict]:
    """Paired terminal comparisons; failed episodes remain in counts and raw rows."""
    episode_ids = {row["episode"] for row in rows}
    common = {key for key in episode_ids if all(any(row["episode"] == key and row["policy"] == policy
              and row["status"] == "COMPLETED" for row in rows) for policy in POLICIES)}
    output = []
    for policy in POLICIES:
        all_rows = [row for row in rows if row["policy"] == policy]
        complete = [row for row in all_rows if row["episode"] in common]
        errors = np.array([row["hedge_error"] for row in complete])
        pnl = np.array([row["net_pnl"] for row in complete])
        tail_count = max(1, int(np.ceil(.05 * len(pnl))))
        output.append(dict(policy=policy, episodes=len(all_rows), matched_completed=len(complete),
            hedge_error_sd=float(errors.std(ddof=1)) if len(errors) > 1 else None,
            hedge_error_rmse=float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
            net_pnl_sd=float(pnl.std(ddof=1)) if len(pnl) > 1 else None,
            cvar95=float(np.sort(-pnl)[-tail_count:].mean()) if len(pnl) else None,
            cvar95_tail_episodes=tail_count if len(pnl) else 0,
            mean_turnover=float(np.mean([row["turnover"] for row in complete])) if complete else None,
            mean_transaction_costs=float(np.mean([row["trading_costs"] for row in complete])) if complete else None,
            mean_fills=float(np.mean([row["fills"] for row in complete])) if complete else None,
            total_turnover_all_runs=sum(row["turnover"] for row in all_rows),
            total_trading_costs_all_runs=sum(row["trading_costs"] for row in all_rows),
            total_fills_all_runs=sum(row["fills"] for row in all_rows),
            rejected_orders_all_runs=sum(row.get("rejected_orders", 0) for row in all_rows),
            skipped_decisions_all_runs=sum(row.get("skipped_decisions", 0) for row in all_rows),
            margin_breaches=sum(row["status"] == "STOP_WITH_BREACH" for row in all_rows),
            data_gaps=sum(row["status"] == "STOP_DATA_GAP" for row in all_rows),
            optimizer_failures=sum(row["status"] == "FAILED_OPTIMIZER" for row in all_rows),
            completion_rate=sum(row["status"] == "COMPLETED" for row in all_rows) / len(all_rows)))
    return output


def run_historical(prepared: Path, model_directory: Path, output: Path,
                   config: HistoricalConfig = HistoricalConfig()) -> dict:
    from .etf_data import load_prepared
    from .learned_model import ConditionalGaussianModel
    from .reporting import write_reports
    from .volgan import VolGANModel

    arrays, manifest = load_prepared(prepared)
    validate_corporate_actions(arrays)
    from .historical_actions import verified_training_parent
    training_hash = verified_training_parent(prepared, arrays, manifest)
    gaussian = ConditionalGaussianModel.load(model_directory / "gaussian.json")
    volgan = VolGANModel.load(model_directory / "volgan.json")
    if (gaussian.metadata.get("historical_data_manifest", {}).get("prepared_sha256") != training_hash
            or volgan.metadata.get("prepared_sha256") != training_hash):
        raise InputError("checkpoint data hash differs from the prepared historical dataset")
    for model in (gaussian, volgan):
        if model.available_at != MODEL_AVAILABLE or model.trained_through > MODEL_AVAILABLE:
            raise InputError("checkpoint does not respect the declared pre-test cutoff")
    mean, covariance, correlation = training_dynamics(arrays)
    arrays = expand_calendar(arrays, manifest)
    stresses = (StressScenario("unchanged"), StressScenario("spot_down_20", .8),
                StressScenario("spot_up_20", 1.2), StressScenario("vol_up_25", 1, 1.25))
    valuator = SurfaceBasketValuator(tuple(map(str, arrays["symbols"])), correlation, MODEL_AVAILABLE,
        samples=config.pricing_samples, pricing_seed=config.pricing_seed, stress_scenarios=stresses)
    first = int(np.searchsorted(arrays["dates"], TEST_START))
    # Calendar fixed before outcomes. Never filter contracts on future surface quality or P&L.
    starts = range(first, len(arrays["dates"]) - config.holding_sessions - 1, config.holding_sessions + 1)
    output.mkdir(parents=True, exist_ok=True)
    rows, representative, failures = [], [], []
    decisions, accounting, fills, solver_records, order_records = [], [], [], [], []
    for number, start in enumerate(starts):
        episode_id = str(arrays["dates"][start])
        schedule = decision_schedule(arrays, start, config)
        try:
            instruments, opening, rules, events = episode(arrays, start, config, valuator)
        except DataGap as error:
            for name in POLICIES:
                rows.append(dict(episode=episode_id,
                    expiry=session_time(arrays["dates"][start + config.holding_sessions]).isoformat(),
                    policy=name, status="STOP_DATA_GAP", reason=str(error), originated=False,
                    hedge_error=None, net_pnl=None, turnover=0., trading_costs=0., fills=0,
                    rejected_orders=0, skipped_decisions=0, dividend_cashflow=0., borrow_charges=0., seconds=0.))
                decisions.append(decision_record(episode_id, name, schedule, [], "STOP_DATA_GAP"))
            continue
        episode_results = []
        for name in POLICIES:
            if name == "unhedged":
                policy = UnhedgedPolicy()
            elif name == "delta":
                policy = ScheduledDeltaPolicy()
            else:
                provider = HistoricalScenarios(name, valuator, config, mean=mean, covariance=covariance,
                                               model=gaussian if name == "gaussian" else volgan)
                weights = {key: config.half_spread * float(arrays["close"][start, column])
                           for column, key in enumerate(arrays["symbols"])}
                policy = ScenarioHedge(provider, tuple(map(str, arrays["symbols"])), weights, alpha=config.alpha)
                policy.name = name
            started = perf_counter()
            result, failure = _run_policy(
                Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()), events, policy)
            elapsed = perf_counter() - started
            components = result.ledger.components()
            completed = result.status == "COMPLETED"
            if failure is not None:
                failures.append(dict(episode=episode_id, policy=name, **failure))
            rows.append(dict(episode=episode_id, expiry=instruments["B"].expiry.isoformat(),
                policy=name, status=failure["status"] if failure else result.status,
                reason=failure["reason"] if failure else result.reason,
                originated=any(post.kind == "origination" for post in result.ledger.postings),
                hedge_error=-(components["liability_pnl"] + components["hedge_pnl"]) if completed else None,
                net_pnl=components["net_pnl"] if completed else None,
                turnover=components["stock_turnover"], trading_costs=components["trading_costs"],
                fills=len(result.fills), rejected_orders=sum(item.outcome == "rejected" for item in result.order_outcomes),
                skipped_decisions=sum(item["status"] == "skipped" for item in result.decisions),
                dividend_cashflow=sum(p.cash_change for p in result.ledger.postings if p.kind == "dividend"),
                borrow_charges=components["borrow_charges"], seconds=elapsed))
            status = rows[-1]["status"]
            decisions.append(decision_record(episode_id, name, schedule, result.decisions, status))
            accounting.append(accounting_record(episode_id, name, status,
                result.last_valid_snapshot.time if result.last_valid_snapshot else None, components))
            fees = {post.id: -post.cash_change for post in result.ledger.postings if post.kind == "fee"}
            fills.extend(dict(episode=episode_id, policy=name, **asdict(fill),
                              fee=fees.get(fill.id + ":fee", 0.)) for fill in result.fills)
            order_records.extend(dict(episode=episode_id, policy=name, **asdict(row)) for row in result.order_outcomes)
            solver_records.extend(dict(episode=episode_id, policy=name,
                **{key: row[key] for key in ("event_id", "time", "iterations", "kkt_residual",
                    "polish_attempted", "polish_succeeded", "solver_tolerance", "max_iterations")})
                for row in getattr(policy, "diagnostics", ()))
            if result.data_gaps or result.breaches:
                failures.append(dict(episode=episode_id, policy=name, status=status,
                    reason=result.reason, last_valid_checkpoint=result.last_valid_snapshot,
                    data_gaps=result.data_gaps, breaches=result.breaches,
                    positions=dict(result.ledger.positions), settlements=result.ledger.settlements,
                    terminal_pnl=None, terminal_hedge_error=None))
            if failure is None:
                episode_results.append(result)
            if number == 0:
                event_map = {event.id: event for event in events}
                for snapshot in result.snapshots:
                    if snapshot.phase != "event" or snapshot.pnl is None:
                        continue
                    event = event_map[snapshot.event_id]
                    basket = sum(quantity * event.valuation.marks[stock] for stock, quantity in instruments["B"].coefficients)
                    representative.append(dict(policy=name, time=snapshot.time.isoformat(), basket=basket,
                        liability_value=snapshot.positions.get("B", 0) * config.multiplier * event.valuation.marks.get("B", 0),
                        net_pnl=snapshot.pnl["net_pnl"],
                        **{key: snapshot.positions.get(key, 0) for key in arrays["symbols"]}))
        if number == 0:
            replay_path = output / "representative.jsonl"
            with replay_path.open("w", encoding="utf-8") as stream:
                for event in events:
                    stream.write(json.dumps(jsonable(event), allow_nan=False) + "\n")
            case = Case(instruments, opening, rules, replay_path,
                        dict(label=PRICING_DESCRIPTION, representative="first test episode, selected before results"))
            write_reports(episode_results, output / "representative", case=case)
        print(f"Episode {number + 1}/{len(starts)} {episode_id}: "
              + ", ".join(f"{row['policy']}={row['status']}" for row in rows[-len(POLICIES):]), flush=True)
    if not rows:
        raise InputError("no complete contractual windows in the declared test calendar")
    _write_rows(output / "episodes.csv", rows)
    write_json(output / "decisions.json", decisions)
    _write_rows(output / "accounting_components.csv", accounting)
    _write_rows(output / "fills.csv", fills)
    write_json(output / "solver_diagnostics.json", solver_records)
    write_json(output / "order_outcomes.json", order_records)
    write_json(output / "failed_runs.json", failures)
    _write_rows(output / "representative.csv", representative,
                headers=("policy", "time", "basket", "liability_value", "net_pnl", *arrays["symbols"]))
    metrics = summarize(rows)
    _write_rows(output / "metrics.csv", metrics)
    summary = dict(label=PRICING_DESCRIPTION, model="multi-asset VolGAN adaptation",
        split=dict(train_end=TRAIN_END, validation_start=VALIDATION_START, validation_end=VALIDATION_END,
                   test_start=TEST_START, source_end=str(arrays["dates"][-1])),
        config=asdict(config), data_manifest=manifest, metrics=metrics,
        model_hashes={name: sha256((model_directory / f"{name}.json").read_bytes()).hexdigest()
                      for name in ("gaussian", "volgan")},
        source_hashes={path.name: sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob("*.py")},
        parametric_daily_log_mean=mean.tolist(), parametric_daily_covariance=covariance.tolist(),
        frozen_pricing_correlation=correlation,
        definitions=dict(hedge_error="negative liability-plus-hedge P&L, including hedge dividends, before trading/funding costs",
            cvar95="mean of largest ceil(5% * matched episodes) net P&L losses, not a fitted tail estimate",
            turnover="sum of absolute executed stock cash notionals, USD",
            comparison="identical matched completed episodes; all failures retained in episodes.csv and counts"),
        assumptions=["daily bars released at assumed 09:30/16:00 America/New_York clocks; half-day timing not modeled",
            "closing option chains assumed available at close; no certified vendor release timestamp",
            "open fills use observed raw open plus/minus assumed 5bp, not observed stock bid/ask or depth",
            "cash distributions posted at ex-date opening, payment-date financing not modeled",
            "fractional shares; assumed locates and 10000-share quote/borrow capacity",
            "zero cash/collateral rates and 1% annual stock borrow; uncalibrated research margin proxy",
            "European ATM-IV basket proxy with frozen correlation and zero pricing rates/dividend yields",
            "VolGAN predicts one observed trading-session transition, including weekends, without rescaling",
            "Gaussian/parametric baselines predict returns and carry current IV; VolGAN also predicts IV",
            "all optimizers use fixed alpha=0.1 and inception spread cost weights; no test-period tuning",
            "daily scenario objective includes overnight exposure before a next-open fill; execution delay remains approximate"])
    write_json(output / "summary.json", summary)
    return summary
