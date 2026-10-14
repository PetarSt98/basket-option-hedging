"""Matched historical comparisons from saved financial results, without rerunning policies."""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .contracts import InputError
from .historical import POLICIES


REFERENCES = ("unhedged", "delta", "gaussian")
LABELS = {"unhedged": "Unhedged", "delta": "Mechanical delta", "parametric": "Parametric",
          "gaussian": "Gaussian", "volgan": "VolGAN"}
CALIBRATION_COVERAGE = {"SPY": 49.12, "QQQ": 60.79, "IWM": 65.86}
NUMERIC_COLUMNS = ("hedge_error", "net_pnl", "turnover", "trading_costs", "fills")


def _load_rows(directory: Path) -> tuple[list[dict], dict]:
    summary_path = directory / "summary.json"
    blocked = directory / "blocked.json"
    if blocked.exists() and (not summary_path.exists()
            or blocked.stat().st_mtime_ns >= summary_path.stat().st_mtime_ns):
        raise InputError("latest financial run was blocked; stale results must not be reported")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with (directory / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        key = row["episode"], row["policy"]
        if key in seen or row["policy"] not in POLICIES:
            raise InputError("duplicate or unknown policy in historical results")
        seen.add(key)
        if row["status"] == "COMPLETED":
            for name in NUMERIC_COLUMNS:
                try:
                    value = float(row[name])
                except (TypeError, ValueError, KeyError) as error:
                    raise InputError("completed historical result lacks a numeric value") from error
                if not np.isfinite(value):
                    raise InputError("nonfinite completed historical result")
                row[name] = value
    for episode in {row["episode"] for row in rows}:
        if any((episode, policy) not in seen for policy in POLICIES):
            raise InputError("historical results omit a policy/episode outcome")
    if not rows:
        raise InputError("historical results contain no episodes")
    return rows, summary


def _sd(values: np.ndarray) -> float | None:
    return float(values.std(ddof=1)) if len(values) > 1 else None


def _cvar95(pnl: np.ndarray) -> float:
    count = max(1, int(np.ceil(.05 * len(pnl))))
    return float(np.sort(-pnl)[-count:].mean())


def _paired_comparisons(indexed: dict, common: list[str]) -> list[dict]:
    values = {policy: {name: np.array([indexed[episode, policy][name] for episode in common])
                       for name in NUMERIC_COLUMNS} for policy in POLICIES}
    comparisons = []
    for reference in REFERENCES:
        for policy in POLICIES:
            if policy == reference:
                continue
            item = dict(policy=policy, reference=reference, matched_completed=len(common))
            current, base = values[policy], values[reference]
            errors, reference_errors = current["hedge_error"], base["hedge_error"]
            pnl, reference_pnl = current["net_pnl"], base["net_pnl"]
            for source, target in (("hedge_error", "mean_hedge_error_difference"),
                                   ("net_pnl", "mean_net_pnl_difference"),
                                   ("turnover", "mean_turnover_difference"),
                                   ("trading_costs", "mean_trading_cost_difference"),
                                   ("fills", "mean_fills_difference")):
                item[target] = float(np.mean(current[source] - base[source])) if common else None
            item.update(
                mean_absolute_error_difference=float(np.mean(np.abs(errors) - np.abs(reference_errors))) if common else None,
                mean_squared_error_difference=float(np.mean(errors ** 2 - reference_errors ** 2)) if common else None,
                hedge_error_sd_difference=_sd(errors) - _sd(reference_errors) if len(common) > 1 else None,
                hedge_error_rmse_difference=float(np.sqrt(np.mean(errors ** 2))
                                                  - np.sqrt(np.mean(reference_errors ** 2))) if common else None,
                net_pnl_sd_difference=_sd(pnl) - _sd(reference_pnl) if len(common) > 1 else None,
                cvar95_difference=_cvar95(pnl) - _cvar95(reference_pnl) if common else None)
            comparisons.append(item)
    return comparisons


def paired_report(report_dir: Path) -> dict:
    """Write descriptive paired differences; stopped runs remain visible in all counts.

    Every pair uses the same intersection of episodes completed by all five
    policies. Differences are policy minus reference. SD, RMSE and CVaR differences
    compare sample statistics on this common set, not uncertainty estimates.
    """
    directory = Path(report_dir)
    rows, summary = _load_rows(directory)
    episodes = sorted({row["episode"] for row in rows})
    indexed = {(row["episode"], row["policy"]): row for row in rows}
    common = [episode for episode in episodes
              if all(indexed[episode, policy]["status"] == "COMPLETED" for policy in POLICIES)]
    comparisons = _paired_comparisons(indexed, common)
    counts = {policy: dict(Counter(row["status"] for row in rows if row["policy"] == policy))
              for policy in POLICIES}
    result = dict(
        matched_completed=len(common), total_episodes=len(episodes), matched_episode_ids=common,
        excluded_episode_ids=[episode for episode in episodes if episode not in common],
        status_counts=counts,
        optimizer_failures={policy: counts[policy].get("FAILED_OPTIMIZER", 0) for policy in POLICIES},
        comparisons=comparisons,
        definitions={
            "difference": "policy minus reference, on the same episodes completed by all five policies",
            "units": "USD, except squared-error differences in USD squared and fills in counts",
            "risk_differences": "sample SD, RMSE and empirical CVaR95 differences on the matched set; no confidence intervals",
            "cvar95": "mean largest ceil(0.05 * matched episodes) net P&L losses",
            "failures": "all statuses retained; no stop-time P&L compared with another policy's terminal P&L",
            "FAILED_OPTIMIZER": "benchmark-level numerical solver failure; not an engine margin breach or data gap"},
        volgan_nominal_90_percent_return_coverage=CALIBRATION_COVERAGE)
    (directory / "paired_differences.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (directory / "paired_differences.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    _write_markdown(directory, summary, result)
    return result


UNCERTAINTY_METRICS = (
    "mean_absolute_error_difference", "hedge_error_sd_difference",
    "hedge_error_rmse_difference", "cvar95_difference",
    "mean_trading_cost_difference", "mean_turnover_difference")


def _population_statistics(indexed: dict, members: dict[str, list[str]],
                           scheduled: int) -> list[dict]:
    """Terminal metrics use only the explicitly named completed population."""
    output = []
    for policy in POLICIES:
        rows = [indexed[episode, policy] for episode in members[policy]]
        error = np.array([row["hedge_error"] for row in rows])
        pnl = np.array([row["net_pnl"] for row in rows])
        count = len(rows)
        output.append(dict(policy=policy, completed_episodes=count, scheduled_episodes=scheduled,
            episode_ids=members[policy], mean_hedge_error=float(error.mean()) if count else None,
            mean_absolute_hedge_error=float(np.abs(error).mean()) if count else None,
            hedge_error_sd=_sd(error),
            hedge_error_rmse=float(np.sqrt(np.mean(error ** 2))) if count else None,
            mean_net_pnl=float(pnl.mean()) if count else None, net_pnl_sd=_sd(pnl),
            cvar95=_cvar95(pnl) if count else None,
            cvar95_tail_episodes=int(np.ceil(.05 * count)),
            mean_turnover=float(np.mean([row["turnover"] for row in rows])) if count else None,
            mean_transaction_costs=float(np.mean([row["trading_costs"] for row in rows])) if count else None,
            mean_fills=float(np.mean([row["fills"] for row in rows])) if count else None))
    return output


def _moving_block_indices(size: int, repetitions: int, block_length: int,
                          seed: int) -> np.ndarray:
    if (any(not isinstance(value, int) or isinstance(value, bool)
            for value in (size, repetitions, block_length, seed))
            or not 1 <= block_length <= size or repetitions <= 0 or seed < 0):
        raise InputError("invalid moving-block bootstrap settings")
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(0, size - block_length + 1,
                         size=(repetitions, int(np.ceil(size / block_length))))
    return (starts[:, :, None] + np.arange(block_length)).reshape(repetitions, -1)[:, :size]


def _bootstrap_statistics(sample: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Vectorized sample statistics; NaN slots retain missing scheduled episodes."""
    valid = np.isfinite(sample[:, :, 0])
    count = valid.sum(axis=1)
    values = np.where(valid[:, :, None], sample, 0.)

    def mean(value: np.ndarray) -> np.ndarray:
        return np.divide(value.sum(axis=1), count, out=np.full(len(count), np.nan), where=count > 0)

    errors, pnl = values[:, :, 0], values[:, :, 1]
    error_mean = mean(errors)
    centered = np.where(valid, errors - error_mean[:, None], 0.)
    sd = np.sqrt(np.divide(np.sum(centered ** 2, axis=1), count - 1,
                          out=np.full(len(count), np.nan), where=count > 1))
    tails = np.ceil(.05 * count).astype(int)
    losses = -np.sort(np.where(valid, pnl, np.inf), axis=1)
    worst = np.arange(sample.shape[1])[None, :] < tails[:, None]
    cvar = np.divide(np.where(worst, losses, 0.).sum(axis=1), tails,
                     out=np.full(len(count), np.nan), where=tails > 0)
    return dict(mean_absolute_error_difference=mean(np.abs(errors)),
        hedge_error_sd_difference=sd, hedge_error_rmse_difference=np.sqrt(mean(errors ** 2)),
        cvar95_difference=cvar, mean_turnover_difference=mean(values[:, :, 2]),
        mean_trading_cost_difference=mean(values[:, :, 3])), count


def _bootstrap_comparisons(indexed: dict, episodes: list[str], members: list[str],
                          draws: np.ndarray) -> list[dict]:
    statistics = {}
    member_set = set(members)
    for policy in POLICIES:
        matrix = np.full((len(episodes), len(NUMERIC_COLUMNS)), np.nan)
        for index, episode in enumerate(episodes):
            if episode in member_set:
                matrix[index] = [indexed[episode, policy][name] for name in NUMERIC_COLUMNS]
        statistics[policy], counts = _bootstrap_statistics(matrix[draws])
    output = []
    for comparison in _paired_comparisons(indexed, members):
        policy, reference = comparison["policy"], comparison["reference"]
        for metric in UNCERTAINTY_METRICS:
            difference = statistics[policy][metric] - statistics[reference][metric]
            valid = difference[np.isfinite(difference)]
            interval = np.quantile(valid, [.025, .975]) if len(valid) else [None, None]
            output.append(dict(policy=policy, reference=reference, metric=metric,
                estimate=comparison[metric], lower_95=interval[0], upper_95=interval[1],
                observed_completed=len(members), observed_cvar95_tail=int(np.ceil(.05 * len(members))),
                valid_replicates=len(valid), requested_replicates=len(draws),
                minimum_resampled_completed=int(counts.min()), maximum_resampled_completed=int(counts.max())))
    return output


def reliability_report(report_dir: Path, frozen_dir: Path, *, seed: int = 20261004,
                       repetitions: int = 5000, block_length: int = 3) -> dict:
    """Report a separately saved correction using preregistered episode populations.

    Blocks are drawn on the entire chronological schedule before filtering each
    population. Every policy and population uses the same episode-index draws.
    """
    directory = Path(report_dir)
    rows, _ = _load_rows(directory)
    old_rows, _ = _load_rows(Path(frozen_dir))
    episodes = sorted({row["episode"] for row in rows})
    if episodes != sorted({row["episode"] for row in old_rows}):
        raise InputError("solver correction must retain the frozen episode schedule")
    indexed = {(row["episode"], row["policy"]): row for row in rows}
    old_indexed = {(row["episode"], row["policy"]): row for row in old_rows}
    decisions = json.loads((directory / "decisions.json").read_text(encoding="utf-8"))
    decision_index = {(row["episode"], row["policy"]): row for row in decisions}
    if len(decision_index) != len(decisions) or set(decision_index) != set(indexed):
        raise InputError("decision records must cover every scheduled policy/episode once")
    sufficient = []
    for episode in episodes:
        flags = [decision_index[episode, policy]["sufficient_history_by_data"] for policy in POLICIES]
        if any(type(flag) is not bool for flag in flags) or len(set(flags)) != 1:
            raise InputError("history availability must be a shared data-defined boolean")
        if flags[0]:
            sufficient.append(episode)
    completed = {policy: [episode for episode in episodes
                         if indexed[episode, policy]["status"] == "COMPLETED"] for policy in POLICIES}
    common = [episode for episode in episodes if all(episode in completed[policy] for policy in POLICIES)]
    frozen_common = [episode for episode in episodes
                     if all(old_indexed[episode, policy]["status"] == "COMPLETED" for policy in POLICIES)]
    matched_old = [episode for episode in frozen_common if episode in common]
    populations = dict(common_completed=(indexed, common),
        common_sufficient_history=(indexed, [episode for episode in common if episode in sufficient]),
        frozen_common_original=(old_indexed, frozen_common),
        frozen_common_matched_original=(old_indexed, matched_old),
        frozen_common_matched_corrected=(indexed, matched_old))
    metrics = {"own_completed": _population_statistics(indexed, completed, len(episodes))}
    draws = _moving_block_indices(len(episodes), repetitions, block_length, seed)
    uncertainty = []
    for population, (source, members) in populations.items():
        metrics[population] = _population_statistics(source, {policy: members for policy in POLICIES}, len(episodes))
        # The full frozen set is also the matched set when no old completion is lost.
        if population == "frozen_common_original":
            continue
        uncertainty.extend(dict(population=population, **item)
                           for item in _bootstrap_comparisons(source, episodes, members, draws))
    statuses = {policy: dict(Counter(row["status"] for row in rows if row["policy"] == policy)) for policy in POLICIES}
    decision_summary = {}
    for policy in POLICIES:
        records = [decision_index[episode, policy] for episode in episodes]
        decision_summary[policy] = dict(
            scheduled_decisions=sum(row["scheduled_decisions"] for row in records),
            decisions_made=sum(row["decisions_made"] for row in records),
            decisions_skipped=sum(row["decisions_skipped"] for row in records),
            skip_affected_episode_ids=[row["episode"] for row in records if row["decisions_skipped"]],
            reasons=dict(Counter(skip["reason"] for row in records for skip in row["skips"])))
    result = dict(total_scheduled=len(episodes), scheduled_episode_ids=episodes, status_counts=statuses,
        completion_rates={policy: len(completed[policy]) / len(episodes) for policy in POLICIES},
        own_completed_episode_ids=completed, common_completed_episode_ids=common,
        sufficient_history_episode_ids=sufficient,
        common_sufficient_history_episode_ids=populations["common_sufficient_history"][1],
        frozen_common_episode_ids=frozen_common, frozen_common_matched_episode_ids=matched_old,
        frozen_common_unavailable_in_correction=[episode for episode in frozen_common if episode not in common],
        metrics=metrics, decision_summary=decision_summary, uncertainty=uncertainty,
        bootstrap=dict(seed=seed, generator="PCG64", repetitions=repetitions, block_length=block_length,
            block_rule="noncircular uniform block starts on full schedule; concatenate and truncate to schedule length",
            filtering="shared episode draws first, then each fixed population; missing slots retained",
            interval="2.5/97.5 percentiles of finite paired statistic differences",
            interpretation="exploratory, conditional on the named completed population; zero inclusion is not equivalence"),
        definitions=dict(error="negative liability-plus-hedge P&L before trading and financing costs",
            sd="sample SD (ddof=1)", rmse="sqrt(mean(error squared)), without demeaning",
            cvar95="mean largest ceil(0.05 * completed count) net P&L losses",
            failure="no terminal value assigned to failures; completion denominator is all scheduled episodes",
            sufficient_history="data available at every scheduled pre-expiry decision, irrespective of realized outcome"))
    (directory / "reliability_populations.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (directory / "uncertainty.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(uncertainty[0]))
        writer.writeheader()
        writer.writerows(uncertainty)
    flat_metrics = [dict(population=population, **{key: value for key, value in item.items() if key != "episode_ids"})
                    for population, items in metrics.items() for item in items]
    with (directory / "population_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_metrics[0]))
        writer.writeheader()
        writer.writerows(flat_metrics)
    return result


def _write_markdown(directory: Path, summary: dict, paired: dict) -> None:
    def number(value):
        return "unavailable" if value is None else f"{value:,.2f}"

    lines = ["# Historical basket hedge benchmark", "",
        "Real historical constituent prices and option surfaces; model-valued non-traded basket option.", "",
        f"Financial comparisons use **{paired['matched_completed']} of {paired['total_episodes']} episodes** "
        "completed by all five policies. Failures remain in the table and raw episode records. "
        "Completion rates use all scheduled episodes. Risk statistics are conditional on successful completion; "
        "they do not measure losses in failed runs.", "",
        "| Policy | Error SD | Error RMSE | P&L SD | CVaR95 | Mean turnover | Mean trading costs | Mean fills | Margin breaches | Optimizer failures | Completion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["metrics"]:
        columns = [LABELS[row["policy"]]] + [number(row[key]) for key in (
            "hedge_error_sd", "hedge_error_rmse", "net_pnl_sd", "cvar95", "mean_turnover",
            "mean_transaction_costs", "mean_fills")]
        columns += [str(row["margin_breaches"]), str(paired["optimizer_failures"][row["policy"]]),
                    f"{100 * row['completion_rate']:.2f}%"]
        lines.append("| " + " | ".join(columns) + " |")
    lines += ["", "Financial columns are USD; fills are counts. Error is negative liability-plus-hedge P&L, "
              "including hedge dividends, before trading and funding costs. CVaR95 is the mean of the largest "
              "ceil(5% × matched episode count) net P&L losses. These are empirical results, not confidence bounds.", "",
              "## Paired differences", "", "Every value below is policy minus reference on the identical completed episode set. "
              "Negative error SD/RMSE or cost differences mean less risk or cost; positive P&L differences mean higher net P&L. "
              "The CSV/JSON also includes signed error, absolute/squared error, P&L SD, CVaR95, turnover and fill differences.", "",
              "| Policy | Reference | Error SD difference | Error RMSE difference | Mean P&L difference | Mean cost difference |",
              "|---|---|---:|---:|---:|---:|"]
    for row in paired["comparisons"]:
        columns = [LABELS[row["policy"]], LABELS[row["reference"]]] + [number(row[key]) for key in (
            "hedge_error_sd_difference", "hedge_error_rmse_difference", "mean_net_pnl_difference", "mean_trading_cost_difference")]
        lines.append("| " + " | ".join(columns) + " |")
    lines += ["", "## Failures", ""]
    for policy, counts in paired["status_counts"].items():
        lines.append(f"- {LABELS[policy]}: " + ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    if any(paired["optimizer_failures"].values()):
        lines += ["", "`FAILED_OPTIMIZER` records a numerical convergence failure in the benchmark driver, "
            "separately from an engine breach or data gap. Its account, pending orders and last valid checkpoint "
            "are retained in `failed_runs.json`; its terminal P&L and hedge error remain unavailable. "
            "The common comparison set excludes that episode for every policy. "
            "The optimizer settings were not relaxed to turn these failures into successful results."]
    lines += ["", "## VolGAN calibration and interpretation", "",
        "The existing nominal 90% return-band coverage is **SPY 49.12%, QQQ 60.79%, IWM 65.86%**. "
        "The weak calibration remains unchanged for this financial benchmark. The model is a multi-asset VolGAN adaptation, "
        "not a reproduction of the paper's original SPX experiment.", "",
        "Paired risk and cost differences show the economic results of the existing model. They cannot by themselves "
        "attribute performance to calibration: VolGAN also generates IV changes, whereas Gaussian and parametric "
        "baselines carry current IV forward. Delayed stock execution and model-valued basket prices remain shared approximations. "
        "No model, alpha or trading parameter was tuned from these results.", "",
        "## Figures", "", "The four figures are in `figures/` as PNG and SVG. "
        "The representative episode is the first test episode, selected before outcomes. "
        "Distribution, time-series and risk/cost figures use the same matched completed episode set.", ""]
    (directory / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
