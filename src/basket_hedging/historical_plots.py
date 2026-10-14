"""Four static figures from saved results, with identical completed episode sets."""

import csv
import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from .contracts import InputError
from .historical import POLICIES


LABELS = {"unhedged": "Unhedged", "delta": "Delta", "parametric": "Parametric",
          "gaussian": "Gaussian", "volgan": "VolGAN"}


def draw_figures(directory: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    directory = Path(directory)
    blocked = directory / "blocked.json"
    summary_path = directory / "summary.json"
    if blocked.exists() and (not summary_path.exists() or blocked.stat().st_mtime_ns >= summary_path.stat().st_mtime_ns):
        raise InputError("latest financial run was blocked; stale results must not be plotted")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with (directory / "episodes.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (directory / "representative.csv").open(encoding="utf-8") as stream:
        representative = list(csv.DictReader(stream))
    version_path = directory / "benchmark_version.json"
    version = json.loads(version_path.read_text(encoding="utf-8")) if version_path.exists() else {}
    labels = LABELS | version.get("policy_labels", {})
    keys = {row["episode"] for row in rows}
    common = {key for key in keys if all(any(row["episode"] == key and row["policy"] == policy
              and row["status"] == "COMPLETED" for row in rows) for policy in POLICIES)}
    if len(common) < 2:
        raise InputError("at least two matched completed historical episodes are required for distribution plots")
    failed_runs = sum(row["status"] != "COMPLETED" for row in rows)
    scheduled = sorted(keys)
    excluded = sorted(keys - common)
    caption = (f"Historical constituents; model-valued basket. Matched episodes: {len(common)}/{len(keys)}. "
               f"Excluded episodes: {len(excluded)}. Failed runs: {failed_runs}/{len(rows)}.")
    if version.get("label"):
        caption += "\n" + version["label"]
    colors = dict(zip(POLICIES, ("#697782", "#176477", "#cd853f", "#6c5891", "#217352")))
    output = directory / "figures"
    output.mkdir(exist_ok=True)
    paths = []

    def save(fig, name):
        fig.text(.01, .005, caption, fontsize=8, color="#526070")
        fig.tight_layout(rect=(0, .065 if version.get("label") else .04, 1, .98))
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=170)
            paths.append(path)
        plt.close(fig)

    selected = {policy: sorted((row for row in rows if row["policy"] == policy and row["episode"] in common),
                               key=lambda row: row["episode"]) for policy in POLICIES}
    time_values = {policy: [next((float(row["hedge_error"]) for row in selected[policy]
                          if row["episode"] == key), None) for key in scheduled] for policy in POLICIES}
    inputs = ["episodes.csv", "summary.json", "representative.csv"]
    if version_path.exists():
        inputs.append(version_path.name)
    figure_data = dict(
        input_sha256={name: sha256((directory / name).read_bytes()).hexdigest() for name in inputs},
        caption=caption, policy_labels=labels, scheduled_episode_ids=scheduled,
        common_completed_episode_ids=sorted(common), excluded_episode_ids=excluded,
        excluded_statuses={key: {row["policy"]: row["status"] for row in rows if row["episode"] == key}
                           for key in excluded},
        distribution={policy: [float(row["hedge_error"]) for row in selected[policy]] for policy in POLICIES},
        time=dict(dates=scheduled, values=time_values, missing_value="null is rendered as NaN; line breaks"),
        risk_cost=[{key: row[key] for key in ("policy", "mean_turnover", "mean_transaction_costs", "hedge_error_sd")}
                   for row in summary["metrics"]],
        representative=dict(rule="first scheduled test episode, selected before results", episode=scheduled[0],
                            rows=[{key: value if key in ("policy", "time") else float(value)
                                   for key, value in row.items()} for row in representative]))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    errors = [[float(row["hedge_error"]) for row in selected[policy]] for policy in POLICIES]
    boxes = ax.boxplot(errors, patch_artist=True)
    ax.set_xticks(range(1, len(POLICIES) + 1), [labels[p] for p in POLICIES])
    for box, color in zip(boxes["boxes"], colors.values()):
        box.set_facecolor(color)
        box.set_alpha(.55)
    ax.axhline(0, color="#aaaaaa", lw=.7)
    ax.set(title="Gross hedge error across historical basket episodes", ylabel="Hedge error (USD)")
    save(fig, "01_error_distribution")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    dates = [datetime.fromisoformat(key) for key in scheduled]
    excluded_dates = [datetime.fromisoformat(key) for key in excluded]
    for policy in POLICIES:
        axis = axes[0] if policy == "unhedged" else axes[1]
        axis.plot(dates, [float("nan") if value is None else value for value in time_values[policy]],
                  label=labels[policy], color=colors[policy], lw=1)
    for ax in axes:
        if excluded_dates:
            for date in excluded_dates:
                ax.axvline(date, color="#b04747", alpha=.22, lw=.8)
            ax.scatter(excluded_dates, [.025] * len(excluded_dates), transform=ax.get_xaxis_transform(),
                       color="#b04747", marker="x", s=25, label=f"Excluded episodes ({len(excluded)})")
        ax.axhline(0, color="#aaaaaa", lw=.7)
        ax.set_ylabel("Hedge error (USD)")
        ax.legend(frameon=False, ncol=4)
    axes[0].set_title("Hedge error by basket inception date")
    save(fig, "02_error_through_time")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for row in summary["metrics"]:
        policy = row["policy"]
        for ax, key, label in zip(axes, ("mean_turnover", "mean_transaction_costs"),
                                 ("Mean turnover (USD)", "Mean trading costs (USD)")):
            ax.scatter(row[key], row["hedge_error_sd"], color=colors[policy], s=42, label=labels[policy])
            ax.set(xlabel=label, ylabel="Hedge-error SD (USD)")
    for ax in axes:
        ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Risk and trading activity")
    save(fig, "03_risk_cost")

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    chosen = [row for row in representative if row["policy"] == "volgan"]
    if chosen:
        dates = [datetime.fromisoformat(row["time"]) for row in chosen]
        axes[0].plot(dates, [float(row["basket"]) for row in chosen], color="#173047")
        axes[1].plot(dates, [float(row["liability_value"]) for row in chosen], color="#173047")
        for stock in ("SPY", "QQQ", "IWM"):
            axes[2].step(dates, [float(row[stock]) for row in chosen], where="post", label=stock)
        axes[2].legend(frameon=False, ncol=3)
    else:
        axes[0].text(.5, .5, "No valid first-episode valuation; see failure report", transform=axes[0].transAxes, ha="center")
    for policy in POLICIES:
        subset = [row for row in representative if row["policy"] == policy]
        axes[3].plot([datetime.fromisoformat(row["time"]) for row in subset],
                     [float(row["net_pnl"]) for row in subset], label=labels[policy], color=colors[policy])
    for ax, ylabel in zip(axes, ("Basket value (USD)", "Signed liability (USD)", "VolGAN shares", "Net P&L (USD)")):
        ax.set_ylabel(ylabel)
    axes[3].legend(frameon=False, ncol=5)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    axes[3].xaxis.set_major_locator(locator)
    axes[3].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[0].set_title("First test episode, selected before results")
    save(fig, "04_representative_episode")
    (output / "figure_data.json").write_text(json.dumps(figure_data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths
