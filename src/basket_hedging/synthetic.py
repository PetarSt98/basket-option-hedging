"""Untrained synthetic market replay, independent of risk-neutral pricing draws."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from .contracts import Funding, InputError, Instrument, Opening, number, timestamp
from .market import Borrow, Event, Fixing, Observation, Quote, Rules
from .replay import write_case
from .valuation import BasketValuator, PricingParameters, StressScenario


def market_path(stock_ids: tuple[str, ...], initial_spots: Sequence[float],
                volatilities: Sequence[float], correlation: Sequence[Sequence[float]],
                times: Sequence[datetime], seed: int) -> Iterator[tuple[datetime, dict[str, float]]]:
    """Zero-drift GBM fixture. The generator never accepts a pricing RNG/sample count."""
    count = len(stock_ids)
    if not count or len(set(stock_ids)) != count or len(initial_spots) != count or len(volatilities) != count:
        raise InputError("market dimensions/stock IDs must match")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise InputError("replay seed must be a nonnegative integer")
    for value in initial_spots:
        number(value, "initial spot")
        if value <= 0:
            raise InputError("initial spots must be positive")
    for value in volatilities:
        number(value, "replay volatility", minimum=0)
    matrix = np.asarray(correlation, dtype=float)
    if (matrix.shape != (count, count) or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)
            or not np.allclose(np.diag(matrix), 1, rtol=0, atol=1e-12)):
        raise InputError("invalid replay correlation matrix")
    # The demonstration generator deliberately accepts positive-definite matrices;
    # it does not modify an invalid/singular input to obtain a Cholesky factor.
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise InputError("replay correlation must be positive definite") from error
    random = np.random.Generator(np.random.PCG64(seed))
    spots = np.array(initial_spots, dtype=float)
    sigma = np.array(volatilities, dtype=float)
    previous = None
    for time in times:
        timestamp(time)
        if previous is not None:
            if time <= previous:
                raise InputError("synthetic market times must strictly increase")
            years = (time - previous).total_seconds() / (365 * 24 * 60 * 60)
            shocks = factor @ random.standard_normal(count)
            spots = spots * np.exp(-0.5 * sigma**2 * years + sigma * np.sqrt(years) * shocks)
        if not np.isfinite(spots).all() or np.any(spots <= 0):
            raise InputError("synthetic path produced nonfinite/nonpositive spots")
        yield time, dict(zip(stock_ids, (float(value) for value in spots)))
        previous = time


def _next_business_date(time: datetime) -> datetime:
    next_day = time + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


def generate_synthetic_case(path: str | Path, *, replay_seed: int = 20260912,
                            pricing_seed: int = 731, samples: int = 8192,
                            constituents: int = 3, days: int = 20, basket_kind: str = "put") -> Path:
    if basket_kind not in {"put", "call"}:
        raise InputError("basket kind must be put or call")
    if (isinstance(constituents, bool) or not isinstance(constituents, int) or constituents < 1
            or isinstance(days, bool) or not isinstance(days, int) or days < 1):
        raise InputError("constituents and toy business days must be positive integers")
    stocks = tuple(f"S{index + 1}" for index in range(constituents))
    initial = (100.0, 80.0, 120.0) if constituents == 3 else tuple(80.0 + 4 * i for i in range(constituents))
    volatility = tuple(0.20 + 0.025 * i for i in range(constituents))
    correlation = tuple(tuple(1.0 if i == j else 0.35 for j in range(constituents)) for i in range(constituents))
    start = datetime(2026, 1, 5, 16, tzinfo=timezone.utc)
    times = [start]
    day = start
    for _ in range(days):
        day = _next_business_date(day)
        times.extend((day.replace(hour=9), day.replace(hour=16)))
    expiry = times[-1]
    horizon = _next_business_date(expiry).replace(hour=9)
    times.append(horizon)
    instruments = {stock: Instrument(stock, "stock") for stock in stocks}
    instruments["BASKET"] = Instrument("BASKET", f"basket_{basket_kind}", multiplier=100,
                                        strike=sum(initial) / constituents, expiry=expiry,
                                        coefficients=tuple((stock, 1 / constituents) for stock in stocks))
    parameters = PricingParameters(stock_ids=stocks, volatilities=volatility, correlation=correlation,
                                   risk_free_rate=0.01, available_at=start, pricing_seed=pricing_seed,
                                   samples=samples, bump_relative=1e-4)
    scenarios = (StressScenario("unchanged"), StressScenario("spot_down_20pct", spot_multiplier=0.8),
                 StressScenario("spot_up_20pct", spot_multiplier=1.2),
                 StressScenario("vol_up_25pct", volatility_multiplier=1.25))
    valuator = BasketValuator(parameters, scenarios)
    opening = Opening("ORIGINATE", initial_capital=10000, liability_id="BASKET", liability_quantity=-1)
    rules = Rules(horizon, stocks, 50, Funding(0.01, 0.005, {stock: 0.02 for stock in stocks}),
                  {stock: Borrow(True, 1000) for stock in stocks}, fee_per_order=0.05,
                  maximum_order_quantities={stock: 1000 for stock in stocks}, start_time=start,
                  expected_stress_scenarios=tuple(scenario.name for scenario in scenarios))

    def valued_events() -> Iterator[Event]:
        # Each valuation call receives one current frame, never the generator or future frames.
        # The already-declared calendar supplies next review times, independently of prices.
        frames = market_path(stocks, initial, volatility, correlation, times, replay_seed)
        for index, (time, spots) in enumerate(frames):
            official = {"BASKET": spots} if time == expiry else None
            observations = tuple(Observation(f"synthetic/{index}/{stock}", stock, value, time, time)
                                 for stock, value in spots.items())
            packet = valuator.packet(time, spots, instruments,
                                     input_ids=tuple(observation.id for observation in observations),
                                     observed_at={stock: time for stock in stocks}, fixings=official)
            decision = time.hour == 16
            execution = time.hour == 9
            quotes = tuple(Quote(f"synthetic/quote/{index}/{stock}", stock, time, time,
                                 value - 0.02, value + 0.02, 1000, 1000)
                           for stock, value in spots.items()) if execution else ()
            yield Event(f"synthetic/{index}", time, index, packet, observations, quotes,
                        (Fixing("synthetic/official/BASKET", "BASKET", expiry, expiry, spots),) if official else (),
                        decision=decision, execution=execution,
                        earliest_execution=times[index + 1] if decision else None,
                        order_deadline=(times[index + 2] if time < expiry else horizon) if decision else None,
                        terminal_decision=time == expiry,
                        next_rebalance_time=times[index + 2] if decision and time < expiry else None,
                        origination_price=packet.marks["BASKET"] if index == 0 else None)

    return write_case(path, instruments, opening, rules, valued_events(),
                      {"source": "synthetic correlated GBM fixture v1; not historical market data",
                       "purpose": "mechanics only, no calibration or profitability claim",
                       "basket_kind": basket_kind,
                       "replay_seed": replay_seed, "replay_rng": "NumPy PCG64",
                       "replay_drift": [0.0] * constituents, "replay_volatilities": volatility,
                       "replay_correlation": correlation, "initial_spots": initial,
                       "pricing_parameters": parameters, "stress_scenarios": scenarios,
                       "margin_label": "uncalibrated research margin proxy", "numpy_version": np.__version__,
                       "calendar": "twenty toy business days by default; weekdays only, no holiday calendar",
                       "days": days, "execution_half_spread": 0.02,
                       "scenario_timing": "Next scheduled business-day close, declared in the current event; no future prices",
                       "funding": "declared toy cash 1%, collateral 0.5%, stock borrow 2%; no proceeds rebate"})
