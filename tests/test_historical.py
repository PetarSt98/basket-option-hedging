"""Historical integration mechanics on explicitly artificial small arrays."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from basket_hedging.contracts import DataGap, InputError
from basket_hedging.engine import Engine
from basket_hedging.historical import (
    HistoricalConfig, MODEL_AVAILABLE, POLICIES, episode, expand_calendar, historical_context,
    session_time, summarize, validate_corporate_actions, _run_policy,
    decision_schedule, decision_record, accounting_record,
)
from basket_hedging.historical_hedge import SurfaceBasketValuator
from basket_hedging.policies import ScheduledDeltaPolicy, UnhedgedPolicy
from basket_hedging.replay import jsonable, parse_event
from basket_hedging.risk import UnnettedLiabilityStressMargin
from basket_hedging.surfaces import PAPER_GRID


def fixture():
    dates = np.busday_offset("2020-01-02", np.arange(50)).astype(str)
    spots = np.tile([100., 80., 120.], (len(dates), 1))
    spots *= np.exp(np.arange(len(dates))[:, None] * .001)
    return dict(dates=dates, symbols=np.array(["SPY", "QQQ", "IWM"]),
        open=spots.copy(), close=spots, dividends=np.zeros_like(spots),
        splits=np.ones_like(spots), iv=np.full((len(dates), 3, *PAPER_GRID.shape), .2),
        valid_surface=np.ones_like(spots, dtype=bool),
        consecutive=np.array([False] + [True] * (len(dates) - 1)))


def valuator():
    return SurfaceBasketValuator(("SPY", "QQQ", "IWM"),
        ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), MODEL_AVAILABLE, samples=128)


def test_missing_actions_block_and_splits_cannot_be_silently_ignored():
    arrays = fixture()
    validate_corporate_actions(arrays)
    arrays["dividends"][:, 2] = np.nan
    with pytest.raises(DataGap, match="unknown dividends"):
        validate_corporate_actions(arrays)
    arrays["dividends"][:] = 0
    arrays["splits"][25, 2] = 2
    with pytest.raises(DataGap, match="splits require"):
        validate_corporate_actions(arrays)


def test_future_data_cannot_change_earlier_packets_conditions_decisions_or_postings():
    arrays = fixture()
    config = HistoricalConfig(holding_sessions=5, scenarios=8, pricing_samples=128)
    instruments, opening, rules, events = episode(arrays, 23, config, valuator())
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["open"][27:] *= .7
    changed["close"][27:] *= .7
    changed["iv"][27:] *= 2
    changed["dividends"][27:, 2] = 1000.  # Future supplement records are not features.
    _, _, _, other = episode(changed, 23, config, valuator())
    cutoff = session_time(arrays["dates"][26])
    assert [jsonable(event) for event in events if event.time <= cutoff] == [
        jsonable(event) for event in other if event.time <= cutoff]
    results = [Engine(instruments, opening, replace(rules, horizon=cutoff),
                      UnnettedLiabilityStressMargin()).run(iter(feed), ScheduledDeltaPolicy())
               for feed in (events, other)]
    assert results[0].decisions == results[1].decisions
    assert results[0].ledger.postings == results[1].ledger.postings
    assert results[0].order_outcomes == results[1].order_outcomes


def test_open_uses_previous_close_surface_and_fills_later_than_decision():
    arrays = fixture()
    arrays["iv"][24] = .9
    config = HistoricalConfig(holding_sessions=3, pricing_samples=128)
    instruments, opening, rules, events = episode(arrays, 23, config, valuator())
    assert events[1].valuation.metadata["surface_observed_at"]["SPY"] == events[0].time.isoformat()
    result = Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()).run(
        iter(events), ScheduledDeltaPolicy())
    assert result.status == "COMPLETED"
    assert result.fills[0].time == events[1].time > events[0].time
    assert result.ledger.posted_margin_cash == 0
    assert result.ledger.restricted_short_proceeds == 0
    assert all(instruments[fill.instrument_id].kind == "stock" for fill in result.fills)
    # Exactly equal initial market contributions: quantities are frozen thereafter.
    assert [q * arrays["close"][23, i] for i, (_, q) in enumerate(instruments["B"].coefficients)] == [100., 100., 100.]
    # The generated replay remains ordinary JSONL-compatible input.
    assert parse_event(jsonable(events[0])) == events[0]


def test_missing_current_surface_stops_without_dropping_the_run():
    arrays = fixture()
    arrays["valid_surface"][24, 1] = False
    instruments, opening, rules, events = episode(arrays, 23, HistoricalConfig(holding_sessions=3), valuator())
    result = Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()).run(
        events, UnhedgedPolicy())
    assert result.status == "STOP_DATA_GAP"
    assert result.ledger.positions == {"B": -1}
    assert result.last_valid_snapshot.time == session_time(arrays["dates"][24], opening=True)


def test_context_uses_three_assets_jointly_and_weekend_clock_is_calendar_based():
    arrays = fixture()
    index = 23
    horizon = session_time(arrays["dates"][index + 1])
    context = historical_context(arrays, index, horizon)
    assert len(context["volgan_features"]) == 306
    assert len(context["gaussian_features"]) == 19
    assert (session_time("2020-01-06") - session_time("2020-01-03")) == timedelta(days=3)
    assert session_time("2020-01-06", opening=True) > session_time("2020-01-03") + timedelta(days=1)
    assert datetime.fromisoformat(context["state_available_at"]) <= horizon


def test_future_missing_chain_date_does_not_shift_expiry_or_earlier_marks():
    full = fixture()
    missing_index = 25
    partial = {key: np.delete(value, missing_index, axis=0)
               if key not in {"symbols"} else value.copy() for key, value in full.items()}
    partial["consecutive"][missing_index] = False
    manifest = dict(common_date_gaps=[dict(previous=full["dates"][24], resumed=full["dates"][26],
                                          missing_dates=[full["dates"][25]])])
    expanded = expand_calendar(partial, manifest)
    config = HistoricalConfig(holding_sessions=5)
    instruments, opening, rules, events = episode(full, 23, config, valuator())
    other_instruments, _, other_rules, altered = episode(expanded, 23, config, valuator())
    assert instruments == other_instruments
    assert rules == other_rules
    assert jsonable(events[0]) == jsonable(altered[0])
    result = Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()).run(
        altered, ScheduledDeltaPolicy())
    assert result.status == "STOP_DATA_GAP"
    assert result.data_gaps[-1]["time"] == session_time(full["dates"][25], opening=True)
    assert result.ledger.positions["B"] == -1


def test_summary_uses_paired_completions_and_independent_tail_calculation():
    rows = []
    for index, error in enumerate([-10., 10., 100.]):
        for policy in POLICIES:
            failed = index == 2 and policy == "volgan"
            rows.append(dict(episode=str(index), policy=policy,
                status="STOP_WITH_BREACH" if failed else "COMPLETED",
                hedge_error=None if failed else error, net_pnl=None if failed else -error - 2,
                turnover=100., trading_costs=2., fills=3))
    metrics = summarize(rows)
    for row in metrics:
        assert row["matched_completed"] == 2
        assert row["hedge_error_sd"] == pytest.approx(np.sqrt(200))
        assert row["hedge_error_rmse"] == 10
        assert row["cvar95"] == 12
    assert metrics[-1]["margin_breaches"] == 1
    assert metrics[-1]["completion_rate"] == pytest.approx(2 / 3)


def test_solver_failure_retains_filled_account_without_invented_terminal_results():
    from basket_hedging.optimization import SolverConvergenceError

    arrays = fixture()
    instruments, opening, rules, events = episode(
        arrays, 23, HistoricalConfig(holding_sessions=3), valuator())

    def fails_after_first_fill(view):
        if view.time == events[0].time:
            return {"SPY": -2.}
        if view.decision_event:
            raise SolverConvergenceError("controlled convergence failure")
        return None

    engine = Engine(instruments, opening, rules, UnnettedLiabilityStressMargin())
    result, failure = _run_policy(engine, events, fails_after_first_fill)
    assert result.status == "RUNNING"  # No fabricated engine financial status.
    assert failure["status"] == "FAILED_OPTIMIZER"
    assert failure["terminal_pnl"] is failure["terminal_hedge_error"] is None
    assert failure["last_valid_checkpoint"].time == events[2].time
    assert failure["positions"] == {"B": -1, "SPY": -2}
    assert failure["settlements"] == {}
    assert len(result.fills) == 1
    spot = float(arrays["open"][24, 0])
    proceeds = 2 * spot * (1 - .0005)
    borrow = 2 * spot * .01 * (6.5 / (365 * 24))
    expected_cash = 100000 + 100 * events[0].origination_price + proceeds - borrow
    assert failure["cash"] == pytest.approx(expected_cash)
    assert failure["restricted_short_proceeds"] == pytest.approx(proceeds)
    assert result.ledger is engine.ledger
    assert failure["postings"] == engine.ledger.postings
    assert failure["pending_orders"] == engine.orders.snapshot()
    assert not result.breaches and not result.data_gaps
    # Another policy still runs the identical episode independently.
    other, other_failure = _run_policy(
        Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()), events, UnhedgedPolicy())
    assert other.status == "COMPLETED" and other_failure is None


def test_benchmark_does_not_suppress_unrelated_errors():
    arrays = fixture()
    instruments, opening, rules, events = episode(
        arrays, 23, HistoricalConfig(holding_sessions=3), valuator())

    def malformed_policy(view):
        raise InputError("unrelated defect")

    with pytest.raises(InputError, match="unrelated defect"):
        _run_policy(Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()),
                    events, malformed_policy)


def test_scheduled_review_reporting_separates_history_skips_and_unreached_reviews():
    arrays = fixture()
    config = HistoricalConfig(holding_sessions=3)
    schedule = decision_schedule(arrays, 23, config)
    assert len(schedule) == 4 and all(row["conditioning_available"] for row in schedule)
    assert schedule[-1]["terminal"]
    # A historical data gap affects conditioning independently of policy outcomes.
    arrays["consecutive"][23] = False
    schedule = decision_schedule(arrays, 23, config)
    assert [row["conditioning_available"] for row in schedule] == [False, False, False, True]
    callbacks = [dict(event_id=schedule[0]["event_id"], time=schedule[0]["time"],
                      status="skipped", reason="joint scenario conditioning is unavailable"),
                 dict(event_id=schedule[1]["event_id"], time=schedule[1]["time"], status="targets"),
                 dict(event_id="irrelevant/open", time=schedule[0]["time"], status="no_decision")]
    row = decision_record("episode", "volgan", schedule, callbacks, "STOP_DATA_GAP")
    assert row["scheduled_decisions"] == 4
    assert row["decisions_made"] == row["decisions_skipped"] == 1
    assert row["decisions_not_reached"] == 2
    assert not row["sufficient_history_by_data"]
    assert row["skips"] == [dict(time=schedule[0]["time"], reason=callbacks[0]["reason"])]
    # No callback is fabricated for a basket that could not be originated.
    unopened = decision_record("episode", "delta", schedule, [], "STOP_DATA_GAP")
    assert unopened["decisions_not_reached"] == 4
    assert unopened["decisions_made"] == unopened["decisions_skipped"] == 0


def test_stopped_reporting_keeps_posted_costs_without_stale_current_pnl():
    time = session_time("2020-01-02")
    components = dict(net_pnl=-12., liability_pnl=-10., hedge_pnl=0.,
                      reconciliation_error=0., trading_costs=1., borrow_charges=1.)
    stopped = accounting_record("episode", "volgan", "STOP_DATA_GAP", time, components)
    assert stopped["last_valid_valuation_time"] == time.isoformat()
    assert stopped["valuation_status"] == "unavailable_current"
    assert stopped["net_pnl"] is stopped["liability_pnl"] is stopped["hedge_pnl"] is None
    assert stopped["reconciliation_error"] is None
    assert stopped["trading_costs"] == stopped["borrow_charges"] == 1
    completed = accounting_record("episode", "delta", "COMPLETED", time, components)
    assert completed["valuation_status"] == "terminal_current"
    assert completed["net_pnl"] == -12
    assert components["net_pnl"] == -12  # Export never changes the ledger data.
