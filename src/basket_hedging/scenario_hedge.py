"""Scenario regression returns holdings; the engine alone trades and funds them."""

from typing import Mapping, Protocol

from .contracts import InputError, number
from .engine import PolicyView
from .optimization import solve_hedge
from .scenarios import ScenarioChanges


class ScenarioSource(Protocol):
    def generate(self, view: PolicyView, fixed_positions: Mapping[str, float],
                 hedge_ids: tuple[str, ...]) -> ScenarioChanges: ...


class ScenarioHedge:
    """One-run, scheduled Eq. (20) policy with actual holdings as turnover origin.

    The regression target is the negative signed change of positions outside the
    candidate set. Candidate positions enter only through previous_holdings.
    Gross scaling is fixed at the first callback, including non-decision events.
    """

    def __init__(self, source: ScenarioSource, hedge_ids: tuple[str, ...],
                 cost_weights: Mapping[str, float], *, alpha: float = 0.0,
                 gross_scale: float | None = None, tolerance: float = 1e-9,
                 max_iterations: int = 20000):
        self.hedge_ids = tuple(hedge_ids)
        if (not self.hedge_ids or len(set(self.hedge_ids)) != len(self.hedge_ids)
                or any(not isinstance(key, str) or not key for key in self.hedge_ids)):
            raise InputError("hedge candidates must be nonempty unique instrument IDs")
        if set(cost_weights) != set(self.hedge_ids):
            raise InputError("one transaction-cost weight is required per hedge candidate")
        for cost in cost_weights.values():
            number(cost, "cost per share/contract", minimum=0)
        number(alpha, "alpha", minimum=0)
        if gross_scale is not None:
            number(gross_scale, "initial marked gross position")
            if gross_scale <= 0:
                raise InputError("initial marked gross position must be positive")
        self.source = source
        self.cost_weights = dict(cost_weights)
        self.alpha = alpha
        self.gross_scale = gross_scale
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.name = "variance" if alpha == 0 else "cont_vuletic_l1"
        self.diagnostics: list[dict] = []

    def __call__(self, view: PolicyView) -> Mapping[str, float] | None:
        for key in self.hedge_ids:
            if (key not in view.hedge_universe or key not in view.instruments
                    or view.instruments[key].is_basket):
                raise InputError("scenario candidates must be permitted hedges, never the basket")
        fixed = {key: quantity for key, quantity in view.positions.items()
                 if key not in self.hedge_ids and quantity != 0}
        if self.gross_scale is None:
            self.gross_scale = sum(abs(quantity * view.instruments[key].multiplier * view.valuation.marks[key])
                                   for key, quantity in fixed.items())
            if self.gross_scale <= 0:
                raise InputError("automatic gross scaling requires a positive opening marked target value")
        if not view.decision_event:
            return None
        active = tuple(key for key in self.hedge_ids if view.instruments[key].expiry is None
                       or view.instruments[key].expiry > view.time)
        if not active:
            return None
        changes = self.source.generate(view, fixed, active)
        if changes.hedge_ids != active:
            raise InputError("scenario columns must match the requested hedge order")
        previous = [view.positions.get(key, 0.0) for key in active]
        solution = solve_hedge(changes.delta_v, changes.delta_h, previous,
                               [self.cost_weights[key] for key in active], alpha=self.alpha,
                               gross_scale=self.gross_scale, tolerance=self.tolerance,
                               max_iterations=self.max_iterations)
        targets = dict(zip(active, map(float, solution.holdings)))
        self.diagnostics.append(dict(
            event_id=view.event_id, time=view.time, horizon=changes.horizon,
            earliest_execution=view.rules.get("earliest_execution"),
            next_rebalance_time=view.rules.get("next_rebalance_time"),
            scenario_seconds=(changes.horizon - view.time).total_seconds(),
            scenario_seed=changes.seed, scenario_count=len(changes.delta_v),
            input_ids=changes.input_ids, solver_tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            fixed_positions=fixed, previous_holdings=dict(zip(active, previous)),
            targets=targets.copy(), alpha=self.alpha, gross_scale=self.gross_scale,
            cost_weights={key: self.cost_weights[key] for key in active},
            intercept=solution.intercept, scenario_residual_mse=solution.mse,
            turnover_penalty=solution.penalty, objective=solution.objective,
            iterations=solution.iterations, kkt_residual=solution.kkt_residual,
            polish_attempted=solution.polish_attempted, polish_succeeded=solution.polish_succeeded,
        ))
        return targets
