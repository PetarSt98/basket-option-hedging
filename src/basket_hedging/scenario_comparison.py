"""Opt-in synthetic comparison, using the existing engine and report writer."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .contracts import InputError, number
from .engine import Engine, Result
from .optimization import solve_hedge
from .policies import ScheduledDeltaPolicy, UnhedgedPolicy
from .replay import load_case, write_json
from .reporting import write_reports
from .risk import UnnettedLiabilityStressMargin
from .scenario_hedge import ScenarioHedge
from .scenarios import SCENARIO_MODEL, ConditionalGBMScenarios
from .valuation import PricingParameters


def _metrics(result: Result) -> dict:
    current = result.snapshots[-1] if result.snapshots else None
    pnl = current.pnl if current is not None else None
    accumulated = result.ledger.components()
    return dict(
        policy=result.policy, status=result.status, reason=result.reason,
        observed_at=current.time if current else None,
        net_pnl=pnl["net_pnl"] if pnl is not None else None,
        gross_hedge_error_currency=-(pnl["liability_pnl"] + pnl["hedge_pnl"]) if pnl is not None else None,
        stock_turnover_currency=accumulated["stock_turnover"],
        option_premium_turnover_currency=accumulated["option_premium_turnover"],
        transaction_costs=accumulated["trading_costs"], financing_costs=accumulated["financing_costs"],
        hedge_instruments_used=len({fill.instrument_id for fill in result.fills}),
        accepted_hedge_fill_count=len(result.fills), breach_count=len(result.breaches),
        margin_label=result.margin_label, current_risk=current.risk if current else None,
        residual_positions=dict(result.ledger.positions),
    )


def _hand_examples() -> dict:
    x = np.array([-1., 1.])
    perfect = solve_hedge(2 * x, x[:, None], [0], [1], alpha=0)
    redundant = solve_hedge(2 * x, np.column_stack((x, x)), [0, 0], [1, 2], alpha=1, gross_scale=1)
    return dict(
        perfect=dict(delta_v=[-2, 2], delta_h=[[-1], [1]], expected_holdings=[2],
                     solution=asdict(perfect)),
        redundant=dict(delta_v=[-2, 2], delta_h=[[-1, -1], [1, 1]], cost_weights=[1, 2],
                       alpha=1, gross_scale=1, expected_holdings=[1.5, 0],
                       explanation="For total b>=0, use the cheaper identical instrument: (2-b)^2+b minimizes at b=1.5.",
                       solution=asdict(redundant)),
    )


def run_scenario_comparison(case_path: str | Path, output_directory: str | Path, *,
                            samples: int = 256, scenario_seed: int = 9173,
                            pricing_samples: int = 1024,
                            alpha: float = .1, cost_proxy: float = .02) -> dict:
    """Four strategies, identical saved replay and engine configuration; no tuning."""
    number(alpha, "regularized comparison alpha")
    number(cost_proxy, "per-share/contract cost proxy", minimum=0)
    if alpha <= 0:
        raise InputError("the regularized comparison requires alpha > 0; variance already uses zero")
    case = load_case(case_path)
    try:
        raw = dict(case.metadata["pricing_parameters"])
        raw["available_at"] = datetime.fromisoformat(raw["available_at"])
        raw["samples"] = pricing_samples
        parameters = PricingParameters(**raw)
        replay_volatility = tuple(case.metadata["replay_volatilities"])
        replay_correlation = tuple(tuple(row) for row in case.metadata["replay_correlation"])
        if any(drift != 0 for drift in case.metadata["replay_drift"]):
            raise InputError("conditional scenarios currently use the existing zero-drift GBM fixture")
    except (KeyError, TypeError, ValueError) as error:
        raise InputError(f"scenario comparison needs declared synthetic pricing/replay parameters: {error}") from error
    candidates = tuple(case.rules.hedge_universe)
    costs = {key: cost_proxy for key in candidates}
    def source():
        return ConditionalGBMScenarios(
            parameters, replay_volatilities=replay_volatility, replay_correlation=replay_correlation,
            samples=samples, seed=scenario_seed,
            max_observation_age_seconds=case.rules.max_mark_age_seconds)
    policies = [UnhedgedPolicy(), ScheduledDeltaPolicy(),
                ScenarioHedge(source(), candidates, costs),
                ScenarioHedge(source(), candidates, costs, alpha=alpha)]
    results = [Engine(case.instruments, case.opening, case.rules,
                      UnnettedLiabilityStressMargin(case.rules.expected_stress_scenarios)).run(case.events(), policy)
               for policy in policies]
    output = Path(output_directory)
    comparison = write_reports(results, output, case_path=case_path, case=case)
    matched = comparison["matched_checkpoint"]
    summary = dict(
        stage="Synthetic-scenario prototype of the Eq. (20) optimization layer; no generative model",
        configuration=dict(scenarios=samples, scenario_seed=scenario_seed, pricing_parameters=parameters,
                           scenario_model=SCENARIO_MODEL,
                           scenario_seed_derivation="SHA256(root seed, UTC decision time, model tag, scenario index), first 128 bits",
                           replay_volatilities=replay_volatility, replay_correlation=replay_correlation,
                           replay_drift="zero", horizon_rule="Next declared rebalance capped by relevant expiry; strictly after execution eligibility",
                           alpha=alpha, cost_weights=costs,
                           gross_scale="Initial sum of absolute marked fixed target legs, captured at the first policy callback",
                           candidate_ids=candidates),
        definitions=dict(
            gross_hedge_error_currency="-(liability P&L + hedge P&L); actual gross tracking residual, excluding trading/financing costs",
            hedge_instruments_used="Number of distinct instruments with accepted hedge fills, including exits",
            turnover="Currency traded on both sides; stock and option-premium amounts reported separately",
            scenario_residual_mse="In-sample conditional fit diagnostic, not observed or out-of-sample hedge performance",
        ),
        outcomes=[_metrics(result) for result in results], matched_checkpoint=matched,
        diagnostics={policy.name: policy.diagnostics for policy in policies if isinstance(policy, ScenarioHedge)},
        provenance=comparison["provenance"],
        interpretation="One synthetic path; no distributional or profitability claim. Compare different stopping times only at matched valid checkpoints.",
    )
    write_json(output / "research_summary.json", summary)
    # NumPy arrays in the numerical examples are deliberately converted at this boundary.
    examples = _hand_examples()
    for example in examples.values():
        for name, value in example["solution"].items():
            if isinstance(value, np.ndarray):
                example["solution"][name] = value.tolist()
    write_json(output / "optimization_examples.json", examples)
    return summary
