"""Show generated next-day scenarios, chosen hedges and their actual backtest.

Run: python examples/paper_demo.py. Uses the saved model; no training required.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from basket_hedging.engine import Engine, PolicyView, Result
from basket_hedging.learned_model import ConditionalGaussianModel
from basket_hedging.learned_scenarios import LearnedScenarios
from basket_hedging.policies import ScheduledDeltaPolicy, UnhedgedPolicy
from basket_hedging.replay import load_case, write_json
from basket_hedging.reporting import write_reports
from basket_hedging.risk import UnnettedLiabilityStressMargin
from basket_hedging.scenario_hedge import ScenarioHedge
from basket_hedging.scenarios import ScenarioChanges
from basket_hedging.valuation import PricingParameters


ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#176477", "#bf7636", "#7874aa")


class CaptureFirstDecision:
    """Observe one provider call for the chart; return its output unchanged."""

    def __init__(self, source: LearnedScenarios):
        self.source = source
        self.first: tuple[PolicyView, ScenarioChanges] | None = None

    def generate(self, view: PolicyView, fixed_positions: Mapping[str, float],
                 hedge_ids: tuple[str, ...]) -> ScenarioChanges:
        changes = self.source.generate(view, fixed_positions, hedge_ids)
        if self.first is None:
            self.first = (view, changes)
        return changes


def plot_decision(spots: np.ndarray, generated: np.ndarray, realized: np.ndarray,
                  decision: dict, stock_ids: tuple[str, ...], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    figure.subplots_adjust(top=.76, bottom=.23, left=.08, right=.97, wspace=.30)
    figure.suptitle("Generate possible next days, then choose a hedge", x=.08,
                   ha="left", fontsize=17, fontweight="bold", y=.96)
    figure.text(.08, .86, f"Decision: {decision['time']:%d %b %H:%M} UTC  |  "
                f"Earliest execution: {decision['earliest_execution']:%d %b %H:%M} UTC\n"
                f"Scenario endpoint / next review: {decision['horizon']:%d %b %H:%M} UTC",
                fontsize=10, color="#526070")
    x = np.arange(len(stock_ids))
    returns = 100 * (generated / spots - 1)
    violins = axes[0].violinplot(returns, positions=x, showmedians=True, widths=.7)
    for body, color in zip(violins['bodies'], COLORS):
        body.set_facecolor(color)
        body.set_alpha(.4)
    axes[0].scatter(x, 100 * (realized / spots - 1), marker="D", s=36,
                    color="#18212b", label="Later replay outcome", zorder=3)
    axes[0].set_title(f"{len(generated)} joint scenarios", loc="left", fontsize=12)
    axes[0].set_ylabel("Return to next review (%)")
    axes[0].legend(fontsize=9, loc="best")
    previous = [decision['previous_holdings'][key] for key in stock_ids]
    targets = [decision['targets'][key] for key in stock_ids]
    axes[1].bar(x - .18, previous, width=.36, color="#cbd5e1", label="Held before decision")
    bars = axes[1].bar(x + .18, targets, width=.36, color=COLORS, label="New absolute target")
    axes[1].bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    axes[1].set_ylim(min(0, *previous, *targets) * 1.25 - 1, max(0, *previous, *targets) + 5)
    axes[1].set_title("Chosen stock positions", loc="left", fontsize=12)
    axes[1].set_ylabel("Shares (negative = short)")
    axes[1].legend(fontsize=9, loc="best")
    for axis in axes:
        axis.set_xticks(x, stock_ids)
        axis.axhline(0, color="#94a3b8", linewidth=.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
    figure.text(.08, .07, "Widths show simulated marginal densities; scenario rows jointly contain all three stocks.\n"
                "The later replay outcome is added after the run, never supplied to the decision.\n"
                "Frozen conditional Gaussian model trained on synthetic data; each draw is one possible future market state.",
                fontsize=9, color="#526070")
    figure.savefig(output / "next_day.png", dpi=160)
    plt.close(figure)


def plot_backtest(results: list[Result], market_events: list, stock_ids: tuple[str, ...],
                  output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    figure.subplots_adjust(top=.91, bottom=.12, left=.09, right=.97, hspace=.38)
    figure.suptitle("Replay the market and execute the hedges", x=.09, ha="left",
                   fontsize=17, fontweight="bold")
    for key, color in zip(stock_ids, COLORS):
        observations = [o for event in market_events for o in event.observations if o.instrument_id == key]
        axes[0].plot([o.observed_at for o in observations],
                     [100 * o.value / observations[0].value for o in observations], color=color, label=key)
    axes[0].set_title("Independent synthetic market path", loc="left", fontsize=11)
    axes[0].set_ylabel("Start = 100")
    labels = {"unhedged": "Unhedged", "delta": "Mechanical delta", "learned_generator": "Generated-scenario hedge"}
    for result, color in zip(results, ("#94a3b8", "#bf7636", "#176477")):
        points = list({s.event_id: s for s in result.snapshots}.values())
        axes[1].plot([s.time for s in points],
                     [s.pnl['net_pnl'] if s.pnl is not None else np.nan for s in points],
                     color=color, label=f"{labels[result.policy]} ({result.status})")
    axes[1].set_title("Account P&L after execution costs and financing", loc="left", fontsize=11)
    axes[1].set_ylabel("USD")
    axes[1].axhline(0, color="#94a3b8", linewidth=.8)
    learned = next(result for result in results if result.policy == "learned_generator")
    points = list({s.event_id: s for s in learned.snapshots}.values())
    for key, color in zip(stock_ids, COLORS):
        axes[2].step([s.time for s in points], [s.positions.get(key, 0) for s in points],
                     where="post", color=color, label=key)
    axes[2].set_title("Generated-scenario hedge: actual holdings after fills", loc="left", fontsize=11)
    axes[2].set_ylabel("Shares")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7, tz=timezone.utc)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timezone.utc))
    axes[2].set_xlabel("Time (UTC)")
    for axis in axes:
        axis.legend(fontsize=8, loc="best", ncol=3 if axis != axes[1] else 1)
        axis.grid(axis="y", alpha=.15)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(.09, .035, "One saved synthetic path: a mechanics demonstration, not a performance ranking.\n"
                "Prices/P&L connect recorded events; holdings change at fills. All policies use the same replay and account rules.",
                fontsize=9, color="#526070")
    figure.savefig(output / "backtest.png", dpi=160)
    plt.close(figure)


def main() -> None:
    case_path = ROOT / "fixtures" / "synthetic.json"
    model_path = ROOT / "reports" / "stage2" / "model.json"
    output = ROOT / "reports" / "paper_demo"
    case = load_case(case_path)
    parameters = dict(case.metadata['pricing_parameters'])
    parameters['available_at'] = datetime.fromisoformat(parameters['available_at'])
    parameters['samples'] = 1024  # Existing Stage 2 scenario repricing setting.
    pricing = PricingParameters(**parameters)
    model = ConditionalGaussianModel.load(model_path)
    source = CaptureFirstDecision(LearnedScenarios(model, pricing, samples=256, seed=9173))
    hedge_ids = case.rules.hedge_universe
    if any(case.instruments[key].kind != 'stock' for key in hedge_ids):
        raise ValueError("This presentation chart requires stock hedge candidates")
    learned = ScenarioHedge(source, hedge_ids, {key: .02 for key in hedge_ids}, alpha=.1)
    learned.name = "learned_generator"

    results = []
    for policy in (UnhedgedPolicy(), ScheduledDeltaPolicy(), learned):
        print(f"Running {policy.name} ...", flush=True)
        engine = Engine(case.instruments, case.opening, case.rules,
                        UnnettedLiabilityStressMargin(case.rules.expected_stress_scenarios))
        results.append(engine.run(case.events(), policy))
    comparison = write_reports(results, output, case_path=case_path, case=case)
    if source.first is None:
        raise RuntimeError("No scenario decision was made; inspect the saved reports")
    view, changes = source.first
    decision = learned.diagnostics[0]
    spots = np.array([view.valuation.marks[key] for key in hedge_ids])
    multipliers = np.array([case.instruments[key].multiplier for key in hedge_ids])
    generated = spots + changes.delta_h / multipliers  # Stock columns are multiplier * price change.

    # Read later observations ONLY after all backtests, for the chart's outcome markers.
    market_events = list(case.events())
    endpoint = {o.instrument_id: o.value for event in market_events for o in event.observations
                if o.observed_at == changes.horizon}
    realized = np.array([endpoint[key] for key in hedge_ids])
    first_fills = [fill for fill in results[-1].fills if view.time < fill.time < changes.horizon]
    write_json(output / 'first_decision.json', dict(
        decision=decision, stock_ids=hedge_ids, current_spots=spots.tolist(),
        generated_spots=generated.tolist(), later_replay_spots=realized.tolist(),
        actual_fills_before_next_review=first_fills))
    write_json(output / 'model_and_decisions.json', dict(
        model_file=str(model_path.relative_to(ROOT)), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), pricing_parameters=pricing,
        generator='frozen conditional Gaussian; synthetic training', decisions=learned.diagnostics))
    plot_decision(spots, generated, realized, decision, hedge_ids, output)
    plot_backtest(results, market_events, hedge_ids, output)
    print(f"\nFirst decision: {view.time}; eligible: {decision['earliest_execution']}; endpoint: {changes.horizon}")
    print("Stock targets:", {key: round(value, 4) for key, value in decision['targets'].items()})
    for fill in first_fills:
        print(f"  Filled {fill.time}: {fill.quantity:+.4f} {fill.instrument_id} @ {fill.price:.4f}")
    for result, outcome in zip(results, comparison['outcomes']):
        pnl = outcome['terminal_pnl']
        displayed = 'unavailable' if pnl is None else f"{pnl:.2f}"
        print(f"{result.policy:18} {result.status:18} terminal P&L={displayed:>11}  "
              f"costs={result.ledger.components()['trading_costs']:.2f}  fills={len(result.fills)}")
    print(f"Charts and reports: {output}")


if __name__ == '__main__':
    main()
