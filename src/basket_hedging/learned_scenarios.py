"""Learned physical scenarios, followed by the unchanged risk-neutral repricer.

This provider cannot trade or write to the portfolio. It receives the same released
view and returns the same signed ScenarioChanges as ConditionalGBMScenarios.
Training is offline; inference needs NumPy and the frozen JSON model only.
"""

from typing import Mapping

import numpy as np

from .contracts import InputError, number
from .engine import PolicyView
from .learned_model import ConditionalGaussianModel
from .learning_data import conditioning_features
from .scenarios import ScenarioChanges, prepare_scenarios, reprice_scenarios, scenario_seed
from .valuation import BasketValuator, PricingParameters, StressScenario


class LearnedScenarios:
    def __init__(self, model: ConditionalGaussianModel, pricing: PricingParameters, *,
                 samples: int = 256, seed: int = 9173, max_observation_age_seconds: float = 0):
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
            raise InputError("learned scenarios require integer samples >= 2")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise InputError("scenario seed must be a nonnegative integer")
        number(max_observation_age_seconds, "scenario input age", minimum=0)
        if model.stock_ids != tuple(pricing.stock_ids):
            raise InputError("learned model and pricing must use the same ordered constituents")
        self.model, self.pricing = model, pricing
        self.samples, self.seed = samples, seed
        self.max_observation_age_seconds = max_observation_age_seconds
        self._valuator = BasketValuator(pricing, (StressScenario("unchanged"),))

    def generate(self, view: PolicyView, fixed_positions: Mapping[str, float],
                 hedge_ids: tuple[str, ...]) -> ScenarioChanges:
        if self.model.available_at > view.time or self.model.trained_through > view.time:
            raise InputError("learned model was not available at decision time")
        inputs = prepare_scenarios(view, fixed_positions, hedge_ids, self.pricing,
                                   self.max_observation_age_seconds)
        current_spots = {observation.instrument_id: observation.value for observation in inputs.observations}
        features, history_ids = conditioning_features(
            self.model.stock_ids, view.history, current_spots, view.time, inputs.horizon,
            lags=self.model.lags)
        days = (inputs.horizon - view.time).total_seconds() / 86400
        # Same independent base normals as the true provider. Changing the number
        # of scenarios preserves earlier rows; no replay or pricing RNG is exposed.
        normals = np.stack([np.random.Generator(np.random.PCG64(scenario_seed(self.seed, view.time, row)))
                            .standard_normal(len(self.model.stock_ids)) for row in range(self.samples)])
        log_returns = self.model.sample_log_returns(features, days, normals)
        with np.errstate(over="ignore", invalid="ignore"):
            future_spots = np.array([current_spots[key] for key in self.model.stock_ids]) * np.exp(log_returns)
        input_ids = tuple(dict.fromkeys((*[o.id for o in inputs.observations], *history_ids)))
        return reprice_scenarios(view, inputs, hedge_ids, future_spots, self._valuator, self.seed, input_ids)
