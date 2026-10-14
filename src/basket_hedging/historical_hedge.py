"""Surface-aware basket repricing and generated-sample alpha selection.

The basket is a model-valued, non-traded European contract. Its declared proxy
uses each constituent's ATM (K/S=1) surface IV at the remaining basket tenor,
a frozen historical correlation, zero rates/dividend yields, and the existing
lognormal arithmetic-basket Monte Carlo pricer. It is not a smile-consistent
multi-asset model. Actual stock dividends belong to the engine cash ledger.
"""

from copy import copy
from dataclasses import dataclass, replace
from datetime import datetime
from math import log
from typing import Mapping

import numpy as np

from .contracts import DataGap, InputError, Instrument, number, timestamp, validate_instruments
from .engine import PolicyView
from .market import ValuationPacket
from .optimization import solve_hedge
from .scenario_hedge import ScenarioHedge, ScenarioSource
from .scenarios import ScenarioChanges
from .surfaces import IVSurface, SurfaceGrid, interpolate_surface
from .valuation import BasketValuator, PricingParameters, SECONDS_PER_YEAR, StressScenario


PAPER_ALPHAS = tuple(i / 100 for i in range(1, 21))
PRICING_DESCRIPTION = (
    "Real historical constituent prices and option surfaces; "
    "model-valued non-traded basket option. "
    "European lognormal ATM-IV basket proxy; frozen historical correlation; "
    "zero pricing rates and dividend yields; ACT/365."
)


class SurfaceBasketValuator:
    """Use the same basket pricing function for marks and generated scenarios.

    Spot bumps for delta keep the selected ATM IVs fixed. Surface timestamps are
    reported separately from the new as-of mark: an opening mark may legitimately
    use the previous close's surface, with its age visible in metadata.
    """

    def __init__(self, stock_ids: tuple[str, ...], correlation: tuple[tuple[float, ...], ...],
                 available_at: datetime, *, pricing_seed: int = 731, samples: int = 512,
                 stress_scenarios: tuple[StressScenario, ...] = (StressScenario("unchanged"),)):
        self.parameters = PricingParameters(tuple(stock_ids), (0.0,) * len(stock_ids),
                                            correlation, 0.0, available_at, pricing_seed, samples)
        self.stress_scenarios = tuple(stress_scenarios)
        self._base = BasketValuator(self.parameters, self.stress_scenarios)

    def _volatilities(self, time: datetime, basket: Instrument,
                      surfaces: Mapping[str, IVSurface]) -> tuple[float, ...]:
        if basket.expiry is None or basket.expiry < time:
            raise InputError("basket valuation cannot run past contract expiry")
        maturity = (basket.expiry - time).total_seconds() / SECONDS_PER_YEAR
        if maturity == 0:
            return (0.0,) * len(self.parameters.stock_ids)
        if set(surfaces) != set(self.parameters.stock_ids):
            raise DataGap("one IV surface per basket constituent is required")
        for surface in surfaces.values():
            if surface.available_at > time or surface.observed_at > time:
                raise InputError("basket pricing cannot use unreleased IV surfaces")
        return tuple(float(surfaces[key].volatility(1.0, maturity))
                     for key in self.parameters.stock_ids)

    def packet(self, time: datetime, spots: Mapping[str, float],
               instruments: Mapping[str, Instrument], surfaces: Mapping[str, IVSurface], *,
               input_ids: tuple[str, ...] = (), observed_at: Mapping[str, datetime] | None = None,
               fixings: Mapping[str, Mapping[str, float]] | None = None) -> ValuationPacket:
        baskets = [item for item in instruments.values() if item.is_basket and item.expiry >= time]
        if any(item.is_option and not item.is_basket for item in instruments.values()):
            raise InputError("historical basket benchmark executes stock hedges only")
        if len(baskets) > 1:
            raise InputError("the historical pricing proxy supports one basket")
        vols = self._volatilities(time, baskets[0], surfaces) if baskets else self.parameters.volatilities
        valuator = BasketValuator(replace(self.parameters, volatilities=vols), self.stress_scenarios)
        packet = valuator.packet(time, spots, instruments, input_ids=input_ids,
                                 observed_at=observed_at, fixings=fixings)
        metadata = dict(packet.metadata)
        metadata.update(historical_basket_proxy=PRICING_DESCRIPTION,
                        surface_observed_at={key: surface.observed_at.isoformat()
                                             for key, surface in surfaces.items()},
                        surface_available_at={key: surface.available_at.isoformat()
                                              for key, surface in surfaces.items()},
                        surface_sources={key: surface.source for key, surface in surfaces.items()},
                        pricing_volatility_rule="surface IV at K/S=1 and remaining basket ACT/365 tenor")
        return replace(packet, metadata=metadata, calibration_cutoff=self.parameters.available_at)


def reprice_basket_scenarios(view: PolicyView, fixed_positions: Mapping[str, float],
                             hedge_ids: tuple[str, ...], *, future_spots: np.ndarray,
                             future_surfaces: np.ndarray, grid: SurfaceGrid, horizon: datetime,
                             valuator: SurfaceBasketValuator, seed: int,
                             input_ids: tuple[str, ...] = ()) -> ScenarioChanges:
    """Convert joint future spots AND surfaces to the unchanged optimizer inputs.

    No transition rescaling occurs here. The caller must supply the next observed
    trading-session horizon, including weekends, from known calendar metadata.
    Hypothetical expiry spots price scenario payoffs; they never settle the book.
    """
    timestamp(horizon, "scenario horizon")
    validate_instruments(view.instruments)
    if valuator.parameters.available_at > view.time or view.valuation.as_of != view.time:
        raise InputError("scenario pricing requires available calibration and as-of marks")
    earliest = view.rules.get("earliest_execution")
    next_review = view.rules.get("next_rebalance_time")
    if earliest is None or next_review is None:
        raise DataGap("historical scenarios require released execution and review times")
    timestamp(earliest, "earliest execution")
    timestamp(next_review, "next review")
    if horizon != next_review:
        raise InputError("daily generated scenarios must end at the next scheduled close")
    if earliest < view.time or horizon <= view.time or horizon <= earliest:
        raise DataGap("daily scenario endpoint must be strictly after execution eligibility")
    stocks = valuator.parameters.stock_ids
    if {key for key, item in view.instruments.items() if item.kind == "stock"} != set(stocks):
        raise InputError("historical scenario/pricing constituents must match the book")
    hedge_ids = tuple(hedge_ids)
    if len(set(hedge_ids)) != len(hedge_ids):
        raise InputError("duplicate hedge IDs")
    if any(key not in stocks or key not in view.hedge_universe
           or view.instruments.get(key) is None or view.instruments[key].kind != "stock"
           for key in hedge_ids):
        raise InputError("historical basket hedges must be permitted constituent stocks")
    fixed = {}
    for key, quantity in fixed_positions.items():
        number(quantity, "fixed position")
        if key not in view.instruments or key in hedge_ids:
            raise InputError("fixed positions must be known and outside the hedge candidates")
        if quantity:
            fixed[key] = quantity
    relevant = set(fixed) | set(hedge_ids)
    for key in relevant:
        item = view.instruments[key]
        if item.is_option and (not item.is_basket or item.expiry < horizon or item.expiry <= view.time):
            raise InputError("scenario options must be the basket and survive to the next close")
        if key not in view.valuation.marks or key not in view.valuation.observed_at:
            raise DataGap(f"missing current mark for {key}")
        if view.valuation.observed_at[key] != view.time:
            raise DataGap("historical scenario decisions require current close marks")
    if np.iscomplexobj(future_spots) or np.iscomplexobj(future_surfaces):
        raise InputError("generated market states must be real")
    try:
        spots = np.asarray(future_spots, dtype=float)
        surfaces = np.asarray(future_surfaces, dtype=float)
    except (TypeError, ValueError) as error:
        raise InputError("generated market states must be numeric") from error
    if (spots.ndim != 2 or len(spots) < 2 or spots.shape[1] != len(stocks)
            or surfaces.shape != (len(spots), len(stocks), len(grid.moneyness), len(grid.maturities))
            or not np.isfinite(spots).all() or not np.isfinite(surfaces).all()
            or np.any(spots <= 0) or np.any(surfaces <= 0)):
        raise InputError("generated states need positive finite spots[N,A] and IV[N,A,M,T], N>=2")
    delta_h = np.column_stack([spots[:, stocks.index(key)] - view.valuation.marks[key]
                               for key in hedge_ids]) if hedge_ids else np.empty((len(spots), 0))
    delta_v = np.zeros(len(spots))
    # Draw ONCE at the common future endpoint. Reuse the authoritative basket
    # payoff/pricing method, omitting unnecessary deltas, stresses and report data.
    draws, _ = valuator._base._draws(horizon)
    unchanged = StressScenario("unchanged")
    scenario_vols = {
        key: interpolate_surface(surfaces, grid, 1.0,
                                 (view.instruments[key].expiry - horizon).total_seconds() / SECONDS_PER_YEAR)
        for key in fixed if view.instruments[key].is_basket and view.instruments[key].expiry > horizon}
    for row in range(len(spots)):
        spot_map = dict(zip(stocks, map(float, spots[row])))
        for key, quantity in fixed.items():
            item = view.instruments[key]
            if item.kind == "stock":
                future_mark = spot_map[key]
            elif item.expiry == horizon:
                future_mark = item.payoff(spot_map)
            else:
                vols = tuple(map(float, scenario_vols[key][row]))
                # A shallow copy retains the constant stock index and shared
                # correlation factor, changing only this scenario's IV inputs.
                scenario_pricer = copy(valuator._base)
                scenario_pricer.parameters = replace(valuator.parameters, volatilities=vols)
                maturity = (item.expiry - horizon).total_seconds() / SECONDS_PER_YEAR
                future_mark = scenario_pricer._basket(item, spots[row], draws, maturity,
                                                       unchanged, deltas=False)[0]
            delta_v[row] -= quantity * item.multiplier * (future_mark - view.valuation.marks[key])
    return ScenarioChanges(hedge_ids, delta_v, delta_h, horizon, seed, input_ids)


@dataclass(frozen=True)
class AlphaSelection:
    alpha: float
    scores: tuple[dict, ...]
    fit_seed: int
    validation_seed: int


def select_paper_alpha(fit: ScenarioChanges, validation: ScenarioChanges,
                       previous_holdings, cost_weights, *, gross_scale: float,
                       alphas: tuple[float, ...] = PAPER_ALPHAS,
                       tolerance: float = 1e-9, max_iterations: int = 20000) -> AlphaSelection:
    """Section 4.2 Eq.24: select once using independent GENERATED validation.

    The reference protocol supplies 1000 fitting and 100 validation draws. These
    counts remain caller-controlled for tiny unit tests. Distinct seed metadata
    is required; callers remain responsible for independent random streams.
    Nonzero *holdings*, plus the intercept, count as fitted parameters. The
    intercept is never refitted on validation data and never added to cash.
    """
    if fit.hedge_ids != validation.hedge_ids or fit.horizon != validation.horizon:
        raise InputError("alpha fitting/validation must describe the same hedge problem")
    if fit.seed == validation.seed:
        raise InputError("alpha validation requires an independent generated seed")
    if not alphas or len(set(alphas)) != len(alphas):
        raise InputError("alpha candidates must be a nonempty unique sequence")
    scores = []
    for alpha in alphas:
        number(alpha, "alpha candidate", minimum=0)
        solution = solve_hedge(fit.delta_v, fit.delta_h, previous_holdings, cost_weights,
                               alpha=alpha, gross_scale=gross_scale, tolerance=tolerance,
                               max_iterations=max_iterations)
        residual = validation.delta_v - solution.intercept - validation.delta_h @ solution.holdings
        rss = float(residual @ residual)
        parameters = 1 + int(np.count_nonzero(solution.holdings))
        count = len(residual)
        aic = count * log(rss / count) + 2 * parameters if rss > 0 else float("-inf")
        scores.append(dict(alpha=float(alpha), aic=aic, rss=rss, parameters=parameters,
                           validation_count=count, fit_count=len(fit.delta_v)))
    # Predeclared numeric order resolves equal scores; never historical P&L.
    best = min(scores, key=lambda row: (row["aic"], row["alpha"]))
    return AlphaSelection(best["alpha"], tuple(scores), fit.seed, validation.seed)


class PaperAICHedge(ScenarioHedge):
    """The existing policy with one inception-only generated-sample AIC choice.

    Fixed-alpha operation remains the original ScenarioHedge. A new instance is
    required for each basket period. The original solver, diagnostics, targets
    and all account/execution logic remain authoritative.
    """

    def __init__(self, source: ScenarioSource, validation_source: ScenarioSource,
                 hedge_ids: tuple[str, ...], cost_weights: Mapping[str, float], **kwargs):
        if "alpha" in kwargs:
            raise InputError("use ScenarioHedge for a fixed alpha")
        super().__init__(source, hedge_ids, cost_weights, **kwargs)
        self.validation_source = validation_source
        self.selection: AlphaSelection | None = None
        self.name = "paper_aic_l1"

    def __call__(self, view: PolicyView) -> Mapping[str, float] | None:
        if view.decision_event and self.selection is None:
            # Existing policy owns candidate validation and initial gross scale.
            super().__call__(replace(view, decision_event=False))
            fixed = {key: quantity for key, quantity in view.positions.items()
                     if key not in self.hedge_ids and quantity}
            fit = self.source.generate(view, fixed, self.hedge_ids)
            validation = self.validation_source.generate(view, fixed, self.hedge_ids)
            self.selection = select_paper_alpha(
                fit, validation, [view.positions.get(key, 0.0) for key in self.hedge_ids],
                [self.cost_weights[key] for key in self.hedge_ids], gross_scale=self.gross_scale,
                tolerance=self.tolerance, max_iterations=self.max_iterations)
            self.alpha = self.selection.alpha
        return super().__call__(view)
