"""Conditional toy GBM scenarios for hedge regression, separate from market replay.

The fixed physical parameters are declared available with the pricing parameters;
they are not fitted forecasts. One step ends at the declared next rebalance,
shortened to the first relevant option expiry, strictly after execution eligibility.
Hypothetical expiry spots are scenario payoffs,
never actual fixings or ledger settlements. No funding/cost cashflows are modeled
in these gross instrument changes; the engine still accounts for actual flows.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Mapping

import numpy as np

from .contracts import DataGap, InputError, number, timestamp, validate_instruments
from .engine import PolicyView
from .synthetic import market_path
from .valuation import BasketValuator, PricingParameters, StressScenario


SCENARIO_MODEL = "conditional-zero-drift-gbm-v1"


def scenario_seed(seed: int, time: datetime, index: int) -> int:
    """Stable per-row shocks; both providers use common random numbers."""
    payload = dict(seed=seed, time_utc=time.astimezone(timezone.utc).isoformat(timespec="microseconds"),
                   model=SCENARIO_MODEL, scenario_index=index)
    return int.from_bytes(hashlib.sha256(json.dumps(payload, sort_keys=True,
                                                    separators=(",", ":")).encode()).digest()[:16], "big")


@dataclass(frozen=True)
class ScenarioChanges:
    hedge_ids: tuple[str, ...]
    delta_v: np.ndarray
    delta_h: np.ndarray
    horizon: datetime
    seed: int
    input_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        hedge_ids = tuple(self.hedge_ids)
        if (any(not isinstance(key, str) or not key for key in hedge_ids)
                or len(set(hedge_ids)) != len(hedge_ids)):
            raise InputError("scenario hedge IDs must be unique nonempty strings")
        timestamp(self.horizon, "scenario horizon")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise InputError("scenario seed must be a nonnegative integer")
        try:
            response = np.asarray(self.delta_v, dtype=float)
            hedges = np.asarray(self.delta_h, dtype=float)
        except (TypeError, ValueError) as error:
            raise InputError("scenario changes must be numeric arrays") from error
        if (response.ndim != 1 or len(response) < 2
                or hedges.shape != (len(response), len(hedge_ids))
                or not np.isfinite(response).all() or not np.isfinite(hedges).all()):
            raise InputError("scenario changes require finite shapes (N,) and (N, hedges), N >= 2")
        object.__setattr__(self, "hedge_ids", hedge_ids)
        object.__setattr__(self, "input_ids", tuple(self.input_ids))
        # Bytes-backed copies cannot regain write permission or alias input arrays.
        for name, values in (("delta_v", response), ("delta_h", hedges)):
            immutable = np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)
            object.__setattr__(self, name, immutable)


@dataclass(frozen=True)
class ScenarioInputs:
    """Validated decision inputs shared by simulator and learned scenarios."""

    fixed: Mapping[str, float]
    relevant: frozenset[str]
    observations: tuple
    registry: Mapping
    horizon: datetime


def prepare_scenarios(view: PolicyView, fixed_positions: Mapping[str, float],
                      hedge_ids: tuple[str, ...], pricing: PricingParameters,
                      max_observation_age_seconds: float) -> ScenarioInputs:
    timestamp(view.time)
    validate_instruments(view.instruments)
    if pricing.available_at > view.time:
        raise InputError("scenario/pricing parameters were not available at decision time")
    if view.valuation.as_of != view.time:
        raise DataGap("scenario base valuation must match decision time")
    hedge_ids = tuple(hedge_ids)
    if (any(not isinstance(key, str) or not key for key in hedge_ids)
            or len(set(hedge_ids)) != len(hedge_ids)):
        raise InputError("candidate hedge IDs must be unique nonempty strings")
    for key in hedge_ids:
        if (key not in view.hedge_universe or key not in view.instruments
                or view.instruments[key].is_basket):
            raise InputError("scenario candidates must be permitted tradable hedges, never the basket")
    fixed = {}
    for key, quantity in fixed_positions.items():
        if key not in view.instruments or key in hedge_ids:
            raise InputError("fixed positions must be known and outside candidate hedges")
        number(quantity, "fixed position quantity")
        if quantity:
            fixed[key] = quantity
    relevant = set(fixed) | set(hedge_ids)
    for key in relevant:
        instrument = view.instruments[key]
        if instrument.expiry is not None and instrument.expiry <= view.time:
            raise InputError("scenario instruments must not be at or past expiry")
        if key not in view.valuation.marks or key not in view.valuation.observed_at:
            raise DataGap(f"missing current scenario mark/timestamp for {key}")
        mark = view.valuation.marks[key]
        number(mark, "current scenario mark", minimum=0)
        if instrument.kind == "stock" and mark <= 0:
            raise InputError("current stock mark must be positive")
        age = (view.time - view.valuation.observed_at[key]).total_seconds()
        if age < 0:
            raise InputError("scenario base mark uses future observations")
        if age > max_observation_age_seconds:
            raise DataGap(f"stale current scenario mark for {key}")
    observations = []
    for key in pricing.stock_ids:
        observation = view.market.get(key)
        if observation is None:
            raise DataGap(f"missing conditional spot for {key}")
        if observation.instrument_id != key or observation.available_at > view.time:
            raise InputError("conditional spots must match their IDs and be released")
        if observation.value <= 0:
            raise InputError("conditional stock spots must be positive")
        if (view.time - observation.observed_at).total_seconds() > max_observation_age_seconds:
            raise DataGap(f"stale conditional spot for {key}")
        observations.append(observation)
    # Only schedule metadata released with this decision is needed. Never scan
    # the replay for its next row, executable quote or realized fixing.
    earliest_execution = view.rules.get("earliest_execution")
    next_rebalance = view.rules.get("next_rebalance_time")
    if earliest_execution is None or next_rebalance is None:
        raise DataGap("scenario decision requires earliest execution and next scheduled rebalance")
    timestamp(earliest_execution, "earliest execution")
    timestamp(next_rebalance, "next scheduled rebalance")
    if earliest_execution < view.time or next_rebalance <= view.time:
        raise InputError("scenario timing must not precede the decision")
    horizon = min([next_rebalance] + [
        view.instruments[key].expiry for key in relevant if view.instruments[key].expiry is not None])
    if horizon <= earliest_execution:
        raise DataGap("scenario horizon leaves no interval after earliest execution")
    registry = {key: instrument for key, instrument in view.instruments.items()
                if instrument.kind == "stock" or key in relevant}
    return ScenarioInputs(fixed, frozenset(relevant), tuple(observations), registry, horizon)


def reprice_scenarios(view: PolicyView, inputs: ScenarioInputs, hedge_ids: tuple[str, ...],
                      future_spots: np.ndarray, valuator: BasketValuator, seed: int,
                      input_ids: tuple[str, ...]) -> ScenarioChanges:
    """One common pricing path: fixed-book loss and per-share/contract hedge gains."""
    future_spots = np.asarray(future_spots, dtype=float)
    if (future_spots.ndim != 2 or future_spots.shape[0] < 2
            or future_spots.shape[1] != len(valuator.parameters.stock_ids)
            or not np.isfinite(future_spots).all() or np.any(future_spots <= 0)):
        raise InputError("scenario prices must be a finite positive (N, constituents) array, N >= 2")
    delta_v = np.empty(len(future_spots))
    delta_h = np.empty((len(future_spots), len(hedge_ids)))
    for row, prices in enumerate(future_spots):
        spots = dict(zip(valuator.parameters.stock_ids, prices))
        hypothetical_fixings = {key: spots for key in inputs.relevant
                                if view.instruments[key].expiry == inputs.horizon}
        future = valuator.packet(inputs.horizon, spots, inputs.registry, fixings=hypothetical_fixings,
                                  input_ids=input_ids)
        delta_v[row] = -sum(quantity * view.instruments[key].multiplier
                            * (future.marks[key] - view.valuation.marks[key])
                            for key, quantity in inputs.fixed.items())
        delta_h[row] = [view.instruments[key].multiplier
                        * (future.marks[key] - view.valuation.marks[key]) for key in hedge_ids]
    return ScenarioChanges(hedge_ids, delta_v, delta_h, inputs.horizon, seed, input_ids)


class ConditionalGBMScenarios:
    """Zero-drift physical draws, then existing risk-neutral instrument pricing.

    Scenario seeds use only root seed, current UTC time, model tag and row index.
    Draws do not depend on holdings, strategy order, future replay or pricing
    sample count. The common future time gives common inner pricing draws across
    all scenarios. ``ScenarioChanges.seed`` records the root scenario seed.
    """

    def __init__(self, pricing: PricingParameters, *, replay_volatilities: tuple[float, ...],
                 replay_correlation: tuple[tuple[float, ...], ...], samples: int = 256,
                 seed: int = 9173,
                 max_observation_age_seconds: float = 0):
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
            raise InputError("conditional scenarios require integer samples >= 2")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise InputError("scenario seed must be a nonnegative integer")
        number(max_observation_age_seconds, "scenario input age", minimum=0)
        self.pricing = pricing
        try:
            self.replay_volatilities = tuple(replay_volatilities)
            self.replay_correlation = tuple(tuple(row) for row in replay_correlation)
        except TypeError as error:
            raise InputError("scenario volatilities/correlation must be explicit sequences") from error
        self.samples = samples
        self.seed = seed
        self.max_observation_age_seconds = max_observation_age_seconds
        # Reuse the replay model's dimension/correlation validation and sampler.
        next(market_path(pricing.stock_ids, (1.0,) * len(pricing.stock_ids),
                         self.replay_volatilities, self.replay_correlation,
                         (pricing.available_at,), seed))
        self._valuator = BasketValuator(pricing, (StressScenario("unchanged"),))

    def _seed(self, time: datetime, index: int) -> int:
        return scenario_seed(self.seed, time, index)

    def generate(self, view: PolicyView, fixed_positions: Mapping[str, float],
                 hedge_ids: tuple[str, ...]) -> ScenarioChanges:
        """Return loss response of the fixed book and long-unit hedge changes.

        ``delta_v = -sum(q*m*(future_mark-current_mark))`` for signed fixed
        positions outside the optimized set. Columns ``delta_h = m*mark_change``
        are per share/contract; regression coefficients are absolute holdings.
        The caller supplies actual fixed positions, excluding optimized hedges.
        """
        inputs = prepare_scenarios(view, fixed_positions, hedge_ids, self.pricing,
                                   self.max_observation_age_seconds)
        future_spots = np.empty((self.samples, len(self.pricing.stock_ids)))
        for row in range(self.samples):
            path = market_path(self.pricing.stock_ids, tuple(o.value for o in inputs.observations),
                               self.replay_volatilities, self.replay_correlation,
                               (view.time, inputs.horizon), self._seed(view.time, row))
            next(path)
            _, spots = next(path)
            future_spots[row] = [spots[key] for key in self.pricing.stock_ids]
        return reprice_scenarios(view, inputs, hedge_ids, future_spots, self._valuator, self.seed,
                                 tuple(o.id for o in inputs.observations))
