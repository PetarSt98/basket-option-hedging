"""In-process pricing, independent of the ledger, policies and replay randomness.

The basket is arithmetic. Its Monte Carlo prices and central-difference deltas
share antithetic pricing draws, including across named stress valuations. Prices
are per underlying/basket unit; position quantities and multipliers belong to the
ledger. Correlation eigenvalues in [-1e-12, 0) are treated as numerical roundoff,
and that treatment is disclosed in each packet; larger violations are rejected.
"""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from math import erf, exp, log, sqrt
from typing import Mapping

import numpy as np

from .contracts import DataGap, InputError, Instrument, number, readonly, timestamp, validate_instruments
from .market import PricingView, ValuationPacket


MODEL_VERSION = "arithmetic-basket-crn-v1"
CORRELATION_EIGENVALUE_TOLERANCE = 1e-12
SECONDS_PER_YEAR = 365 * 24 * 60 * 60


@dataclass(frozen=True)
class PricingParameters:
    stock_ids: tuple[str, ...]
    volatilities: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    risk_free_rate: float
    available_at: datetime
    pricing_seed: int
    samples: int = 8192
    bump_relative: float = 1e-4

    def __post_init__(self) -> None:
        for name in ("stock_ids", "volatilities", "correlation"):
            object.__setattr__(self, name, readonly(getattr(self, name)))
            if not isinstance(getattr(self, name), tuple):
                raise InputError(f"pricing {name} must be an explicit sequence")
        timestamp(self.available_at, "pricing-parameter availability")
        if (not self.stock_ids or len(set(self.stock_ids)) != len(self.stock_ids)
                or any(not isinstance(key, str) or not key for key in self.stock_ids)):
            raise InputError("pricing stocks must have nonempty unique IDs")
        if len(self.volatilities) != len(self.stock_ids):
            raise InputError("one pricing volatility is required per stock")
        for volatility in self.volatilities:
            number(volatility, "pricing volatility", minimum=0)
        number(self.risk_free_rate, "risk-neutral rate")
        number(self.bump_relative, "relative spot bump")
        if not 0 < self.bump_relative < 1:
            raise InputError("relative spot bump must be between zero and one")
        if isinstance(self.pricing_seed, bool) or not isinstance(self.pricing_seed, int) or self.pricing_seed < 0:
            raise InputError("pricing seed must be a nonnegative integer")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 2:
            raise InputError("pricing requires an integer sample count of at least two")
        n = len(self.stock_ids)
        if len(self.correlation) != n or any(not isinstance(row, tuple) or len(row) != n for row in self.correlation):
            raise InputError("pricing correlation dimensions must match the stocks")
        for row in self.correlation:
            for value in row:
                number(value, "pricing correlation")
                if not -1 <= value <= 1:
                    raise InputError("correlations must be in [-1, 1]")
        correlation = np.array(self.correlation, dtype=float)
        if not np.array_equal(correlation, correlation.T) or not np.all(np.diag(correlation) == 1):
            raise InputError("correlation must be symmetric with unit diagonal")
        if np.linalg.eigvalsh(correlation).min() < -CORRELATION_EIGENVALUE_TOLERANCE:
            raise InputError("correlation must be positive semidefinite")


@dataclass(frozen=True)
class StressScenario:
    name: str
    spot_multiplier: float = 1.0
    volatility_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise InputError("stress scenarios require nonempty names")
        number(self.spot_multiplier, "stress spot multiplier")
        number(self.volatility_multiplier, "stress volatility multiplier", minimum=0)
        if self.spot_multiplier <= 0:
            raise InputError("stress spot multiplier must be positive")


def black_scholes(spot: float, strike: float, maturity: float, rate: float,
                  volatility: float, kind: str) -> tuple[float, float]:
    """Unit European price/delta before expiry, zero dividends, ACT/365 years.

    At a zero-volatility payoff kink the returned delta uses the midpoint of the
    two one-sided derivatives. Actual expiry instead requires official fixings.
    """
    for value, name in ((spot, "spot"), (strike, "strike"), (maturity, "maturity"),
                        (rate, "rate"), (volatility, "volatility")):
        number(value, name)
    if spot <= 0 or strike < 0 or maturity <= 0 or volatility < 0 or kind not in {"put", "call"}:
        raise InputError("invalid Black-Scholes input")
    if strike == 0:
        return (spot, 1.0) if kind == "call" else (0.0, 0.0)
    discounted_strike = strike * exp(-rate * maturity)
    if volatility == 0:
        signed_intrinsic = spot - discounted_strike
        call_delta = 1.0 if signed_intrinsic > 0 else 0.0 if signed_intrinsic < 0 else 0.5
        return ((max(signed_intrinsic, 0.0), call_delta) if kind == "call"
                else (max(-signed_intrinsic, 0.0), call_delta - 1.0))
    sigma_root_t = volatility * sqrt(maturity)
    d1 = (log(spot / strike) + (rate + volatility * volatility / 2) * maturity) / sigma_root_t
    d2 = d1 - sigma_root_t

    def normal_cdf(value: float) -> float:
        return (1 + erf(value / sqrt(2))) / 2

    if kind == "call":
        return spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2), normal_cdf(d1)
    return discounted_strike * normal_cdf(-d2) - spot * normal_cdf(-d1), normal_cdf(d1) - 1


class BasketValuator:
    def __init__(self, parameters: PricingParameters, scenarios: tuple[StressScenario, ...]):
        self.parameters = parameters
        self.scenarios = tuple(scenarios)
        names = [scenario.name for scenario in self.scenarios]
        if len(set(names)) != len(names) or "unchanged" not in names:
            raise InputError("pricing stress names must be unique and include unchanged")
        unchanged = self.scenarios[names.index("unchanged")]
        if unchanged.spot_multiplier != 1 or unchanged.volatility_multiplier != 1:
            raise InputError("unchanged pricing scenario must have unit multipliers")
        correlation = np.array(parameters.correlation, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        self._minimum_eigenvalue = float(eigenvalues.min())
        # This is explicit floating-point treatment within the validated tolerance.
        self._factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        self._stock_index = {key: index for index, key in enumerate(parameters.stock_ids)}

    def value(self, view: PricingView) -> ValuationPacket:
        """Price from released stock observations, without a portfolio or replay feed.

        Original observation times remain attached to marks: advancing the clock
        does not refresh a stale spot. At expiry, include exact official payoffs
        only when their fixings are present. The engine validates mandatory
        fixings for held contracts before invoking this valuation boundary.
        """
        timestamp(view.time)
        validate_instruments(view.instruments)
        for record in (*view.market.values(), *view.history, *view.quotes.values(), *view.fixings.values()):
            if record.available_at > view.time:
                raise InputError("pricing view contains unreleased observations, quotes or fixings")
        for key, observation in view.market.items():
            if key != observation.instrument_id:
                raise InputError("pricing market observation must match its instrument key")
        if any(key not in view.market for key in self.parameters.stock_ids):
            raise DataGap("missing pricing spot observation")
        observations = {key: view.market[key] for key in self.parameters.stock_ids}
        registry = {key: instrument for key, instrument in view.instruments.items()
                    if not instrument.is_option or instrument.expiry > view.time
                    or (instrument.expiry == view.time and key in view.fixings)}
        fixings = {}
        for key, instrument in registry.items():
            if instrument.is_option and instrument.expiry == view.time:
                fixing = view.fixings[key]
                if fixing.instrument_id != key or fixing.observed_at != instrument.expiry:
                    raise InputError("official pricing fixing must match the contract and expiry")
                fixings[key] = fixing.spots
        input_ids = tuple(observations[key].id for key in sorted(observations)) + tuple(
            view.fixings[key].id for key in sorted(fixings))
        packet = self.packet(view.time, {key: observation.value for key, observation in observations.items()},
                             registry, observed_at={key: observation.observed_at
                                                   for key, observation in observations.items()},
                             fixings=fixings, input_ids=input_ids)
        return replace(packet, calibration_cutoff=self.parameters.available_at)

    def _draws(self, time: datetime) -> tuple[np.ndarray, int]:
        seed_material = json.dumps({
            "seed": self.parameters.pricing_seed,
            "time_utc": time.astimezone(timezone.utc).isoformat(timespec="microseconds"),
            "model": MODEL_VERSION,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:16], "big")
        generator = np.random.Generator(np.random.PCG64(derived_seed))
        half = (self.parameters.samples + 1) // 2
        independent = generator.standard_normal((half, len(self.parameters.stock_ids)))
        antithetic = np.concatenate((independent, -independent), axis=0)[:self.parameters.samples]
        return antithetic @ self._factor.T, derived_seed

    def _basket(self, instrument: Instrument, spots: np.ndarray, draws: np.ndarray,
                maturity: float, scenario: StressScenario, *, deltas: bool) -> tuple[float, dict[str, float]]:
        volatility = np.array(self.parameters.volatilities) * scenario.volatility_multiplier
        rate = self.parameters.risk_free_rate
        with np.errstate(over="raise", invalid="raise"):
            try:
                growth = np.exp((rate - 0.5 * volatility**2) * maturity
                                + volatility * sqrt(maturity) * draws)
                terminal = spots * scenario.spot_multiplier * growth
            except FloatingPointError as error:
                raise InputError("nonfinite simulated pricing values") from error
        basket = sum(coefficient * terminal[:, self._stock_index[key]]
                     for key, coefficient in instrument.coefficients)
        discount = exp(-rate * maturity)
        direction = 1.0 if instrument.kind == "basket_call" else -1.0
        intrinsic = direction * (basket - instrument.strike)
        value = discount * float(np.mean(np.maximum(intrinsic, 0.0)))
        result = {}
        if deltas:
            for key, coefficient in instrument.coefficients:
                index = self._stock_index[key]
                bump = spots[index] * self.parameters.bump_relative
                change = direction * coefficient * bump * growth[:, index]
                plus = np.maximum(intrinsic + change, 0.0)
                minus = np.maximum(intrinsic - change, 0.0)
                result[key] = discount * float(np.mean(plus - minus)) / (2 * bump)
        return value, result

    def packet(self, time: datetime, spots: Mapping[str, float], instruments: Mapping[str, Instrument],
               *, input_ids: tuple[str, ...] = (), observed_at: Mapping[str, datetime] | None = None,
               fixings: Mapping[str, Mapping[str, float]] | None = None) -> ValuationPacket:
        timestamp(time)
        if time < self.parameters.available_at:
            raise InputError("pricing parameters were not available at this event")
        validate_instruments(instruments)
        registered_stocks = {key for key, instrument in instruments.items() if instrument.kind == "stock"}
        if registered_stocks != set(self.parameters.stock_ids):
            raise InputError("pricing stock IDs must match the instrument registry")
        if any(key not in spots for key in self.parameters.stock_ids):
            raise DataGap("missing pricing spot")
        if set(spots) != registered_stocks:
            raise InputError("pricing spots contain unknown instruments")
        for spot in spots.values():
            number(spot, "pricing spot")
            if spot <= 0:
                raise InputError("pricing spots must be positive")
        source_times = dict(observed_at) if observed_at is not None else {key: time for key in spots}
        for key in self.parameters.stock_ids:
            if key not in source_times:
                raise DataGap("missing pricing spot observation timestamp")
            timestamp(source_times[key])
            if source_times[key] > time:
                raise InputError("pricing cannot use future observations")
        spot_values = np.array([spots[key] for key in self.parameters.stock_ids], dtype=float)
        draws, derived_seed = self._draws(time)
        marks: dict[str, float] = {}
        unit_deltas: dict[str, dict[str, float]] = {}
        mark_times: dict[str, datetime] = {}
        stress_marks = {scenario.name: {} for scenario in self.scenarios}
        unchanged = next(s for s in self.scenarios if s.name == "unchanged")
        official_fixings = fixings or {}
        for key, instrument in instruments.items():
            if instrument.kind == "stock":
                marks[key] = spots[key]
                mark_times[key] = source_times[key]
                for scenario in self.scenarios:
                    stress_marks[scenario.name][key] = spots[key] * scenario.spot_multiplier
                continue
            if time > instrument.expiry:
                continue
            if time == instrument.expiry:
                if key not in official_fixings:
                    raise DataGap(f"expiry valuation requires official fixing for {key}")
                marks[key] = instrument.payoff(official_fixings[key])
                mark_times[key] = time
                for scenario in self.scenarios:
                    stress_marks[scenario.name][key] = marks[key]
                continue
            maturity = (instrument.expiry - time).total_seconds() / SECONDS_PER_YEAR
            mark_times[key] = min(source_times[stock] for stock in instrument.risk_stocks)
            if instrument.is_basket:
                marks[key], unit_deltas[key] = self._basket(instrument, spot_values, draws, maturity,
                                                          unchanged, deltas=True)
                for scenario in self.scenarios:
                    stress_marks[scenario.name][key] = (marks[key] if scenario.name == "unchanged" else
                        self._basket(instrument, spot_values, draws, maturity, scenario, deltas=False)[0])
            else:
                stock = instrument.underlying
                volatility = self.parameters.volatilities[self._stock_index[stock]]
                marks[key], delta = black_scholes(spots[stock], instrument.strike, maturity,
                                                  self.parameters.risk_free_rate, volatility, instrument.kind)
                unit_deltas[key] = {stock: delta}
                for scenario in self.scenarios:
                    stress_marks[scenario.name][key] = (marks[key] if scenario.name == "unchanged" else
                        black_scholes(spots[stock] * scenario.spot_multiplier, instrument.strike, maturity,
                                      self.parameters.risk_free_rate,
                                      volatility * scenario.volatility_multiplier, instrument.kind)[0])
        parameters = asdict(self.parameters)
        parameters["available_at"] = self.parameters.available_at.isoformat()
        metadata = {
            "model_version": MODEL_VERSION, "parameters": parameters,
            "pricing_seed_derived": derived_seed, "seed_derivation": "SHA256(seed, UTC event time, model version), first 128 bits",
            "rng": "NumPy PCG64; local generator independent of replay", "antithetic_draws": True,
            "common_random_numbers": "same draws across spot bumps, options and named stresses",
            "correlation_eigenvalue_tolerance": CORRELATION_EIGENVALUE_TOLERANCE,
            "correlation_minimum_eigenvalue": self._minimum_eigenvalue,
            "correlation_roundoff_clipped": self._minimum_eigenvalue < 0,
            "scenarios": [asdict(scenario) for scenario in self.scenarios],
            "input_ids": tuple(input_ids), "conventions": "zero dividends; ACT/365; unit marks/deltas; exact official expiry fixing",
        }
        return ValuationPacket(time, marks, stress_marks, unit_deltas, mark_times,
                               time, tuple(input_ids), metadata)
