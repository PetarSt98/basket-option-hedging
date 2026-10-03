"""Run three saved examples and plot their actual account records.

From the repository: python examples/demo.py
Outputs: reports/demo/{deterministic,synthetic,gap}/
"""

from datetime import timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save charts without requiring a desktop plotting window.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from basket_hedging.engine import Engine, Result
from basket_hedging.policies import ScheduledDeltaPolicy, UnhedgedPolicy
from basket_hedging.replay import Case, load_case
from basket_hedging.reporting import write_reports
from basket_hedging.risk import UnnettedLiabilityStressMargin


ROOT = Path(__file__).resolve().parents[1]


def event_snapshots(result: Result) -> list:
    """Keep the last recorded state of each event, including a stopping event."""
    latest = {}
    for snapshot in result.snapshots:
        latest[snapshot.event_id] = snapshot
    return list(latest.values())


def plot_run(case: Case, results: list[Result], title: str, output: Path) -> None:
    currency = next(iter(case.instruments.values())).currency
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    figure.subplots_adjust(top=.90, bottom=.12, left=.10, right=.97, hspace=.35)
    figure.suptitle(title, fontsize=18, fontweight="bold", x=.10, ha="left")
    for result, color in zip(results, ("#64748b", "#167582")):
        points = event_snapshots(result)
        pnl = [p.pnl["net_pnl"] if p.pnl is not None else float("nan") for p in points]
        axes[0].plot([p.time for p in points], pnl, marker=".", color=color,
                     label=f"{result.policy}: {result.status}")
    axes[0].set_title("Account P&L after costs", loc="left", fontsize=11)
    axes[0].set_ylabel(currency)
    axes[0].legend(fontsize=9, loc="best")

    delta = next(result for result in results if result.policy == "delta")
    points = event_snapshots(delta)
    times = [p.time for p in points]
    stocks = [key for key, instrument in case.instruments.items() if instrument.kind == "stock"]
    for index, stock in enumerate(stocks):
        axes[1].step(times, [p.positions.get(stock, 0) for p in points], where="post",
                     linestyle=("-", "--", ":")[index % 3], label=stock)
    axes[1].set_title("Delta policy: actual stock holdings (negative means short)", loc="left", fontsize=11)
    axes[1].set_ylabel("Shares")
    axes[1].legend(ncol=len(stocks), fontsize=9, loc="best")

    # Display balances relative to the engine's recorded requirement; no new risk model.
    equity_room, cash_room = [], []
    for point in points:
        risk = point.risk
        known = risk.required_margin is not None and risk.equity is not None
        required = risk.required_margin + risk.safety_buffer if known else float("nan")
        equity_room.append(risk.equity - required if known else float("nan"))
        cash_room.append(risk.cash - risk.restricted_short_proceeds - required)
    axes[2].plot(times, equity_room, marker=".", color="#167582", label="Equity above requirement")
    axes[2].plot(times, cash_room, marker=".", color="#b7791f", label="Cash backing above requirement")
    axes[2].set_title("Delta policy: margin plus buffer coverage", loc="left", fontsize=11)
    axes[2].set_ylabel(currency)
    axes[2].legend(fontsize=9, loc="best")
    if delta.breaches:
        axes[2].annotate("Breach before fills\nRun stops here", xy=(times[-1], equity_room[-1]),
                         xytext=(-155, 40), textcoords="offset points", color="#b42318",
                         arrowprops=dict(arrowstyle="->", color="#b42318"), fontsize=10)
    expiry = case.instruments[case.opening.liability_id].expiry
    if expiry <= times[-1]:
        for axis in axes:
            axis.axvline(expiry, color="#94a3b8", linestyle=":", linewidth=1)
        axes[0].text(expiry, 1.02, "Option expiry", transform=axes[0].get_xaxis_transform(),
                     ha="right", fontsize=9, color="#526070")
    for axis in axes:
        axis.axhline(0, color="#94a3b8", linewidth=.8)
        axis.grid(axis="y", alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7, tz=timezone.utc)
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timezone.utc))
    axes[2].set_xlabel("Time (UTC)")
    figure.text(.10, .035, "Dots are observed events; lines connect recorded values. Inventory steps change at fills.\n"
                "Single demonstration path, not a performance ranking. Margin is an uncalibrated research proxy.",
                fontsize=9, color="#526070")
    figure.savefig(output / "run.png", dpi=160)
    plt.close(figure)


def main() -> None:
    examples = (
        ("deterministic", "deterministic_complete.json", "Hand-calculated example | trading, settlement and exit"),
        ("synthetic", "synthetic.json", "Synthetic replay | one three-stock market path"),
        ("gap", "deterministic_gap.json", "Margin breach | stopped before a pending hedge can fill"),
    )
    for name, filename, title in examples:
        case_path = ROOT / "fixtures" / filename
        case = load_case(case_path)
        results = []
        for policy in (UnhedgedPolicy(), ScheduledDeltaPolicy()):
            margin = UnnettedLiabilityStressMargin(case.rules.expected_stress_scenarios)
            engine = Engine(case.instruments, case.opening, case.rules, margin)
            results.append(engine.run(case.events(), policy))
        output = ROOT / "reports" / "demo" / name
        report = write_reports(results, output, case_path=case_path, case=case)
        plot_run(case, results, title, output)
        for result, outcome in zip(results, report["outcomes"]):
            pnl = outcome["marked_pnl"]
            displayed = "unknown" if pnl is None else f"{pnl:.2f}"
            print(f"{name:13} {result.policy:8} {result.status:18} P&L={displayed:>8}  fills={len(result.fills)}")
        print(f"Chart: {output / 'run.png'}")


if __name__ == "__main__":
    main()
