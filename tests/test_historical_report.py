"""Independent paired arithmetic, retained failures and stale-result protection."""

import csv
import json
import os

import numpy as np
import pytest

from basket_hedging.contracts import InputError
from basket_hedging.historical import POLICIES, summarize
from basket_hedging.historical_report import (
    _bootstrap_statistics, _moving_block_indices, paired_report, reliability_report)


def saved_results(directory, *, failed=True):
    rows = []
    for index, baseline_error in enumerate((-10., 10., 1000.)):
        for policy in POLICIES:
            error = baseline_error / 2 if policy == "delta" else baseline_error
            cost = 3. if policy == "delta" else 2.
            stopped = failed and index == 2 and policy == "volgan"
            rows.append(dict(episode=f"2020-0{index + 1}-02", policy=policy,
                status="STOP_WITH_BREACH" if stopped else "COMPLETED",
                hedge_error=None if stopped else error, net_pnl=None if stopped else -error - cost,
                turnover=120. if policy == "delta" else 100.,
                trading_costs=cost, fills=4 if policy == "delta" else 3))
    with (directory / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "summary.json").write_text(json.dumps(dict(metrics=summarize(rows))), encoding="utf-8")
    return rows


def test_paired_arithmetic_uses_all_five_completed_intersection(tmp_path):
    saved_results(tmp_path)
    result = paired_report(tmp_path)
    # The extreme third episode completed for four policies but must not enter any pair.
    assert result["matched_completed"] == 2
    assert result["total_episodes"] == 3
    assert result["status_counts"]["volgan"] == {"COMPLETED": 2, "STOP_WITH_BREACH": 1}
    assert result["status_counts"]["unhedged"] == {"COMPLETED": 3}
    pair = next(row for row in result["comparisons"] if row["policy"] == "delta" and row["reference"] == "unhedged")
    # Errors [-5, 5] versus [-10, 10]: sample SDs sqrt(50), sqrt(200).
    assert pair["hedge_error_sd_difference"] == pytest.approx(-(50 ** .5))
    assert pair["hedge_error_rmse_difference"] == -5
    assert pair["mean_hedge_error_difference"] == 0
    assert pair["mean_absolute_error_difference"] == -5
    assert pair["mean_squared_error_difference"] == -75
    assert pair["mean_net_pnl_difference"] == -1
    assert pair["cvar95_difference"] == -4
    assert pair["mean_turnover_difference"] == 20
    assert pair["mean_trading_cost_difference"] == 1
    assert pair["mean_fills_difference"] == 1
    assert len(result["comparisons"]) == 12
    assert {row["reference"] for row in result["comparisons"]} == {"unhedged", "delta", "gaussian"}
    assert json.loads((tmp_path / "paired_differences.json").read_text()) == result
    markdown = (tmp_path / "benchmark.md").read_text()
    assert "49.12%" in markdown and "60.79%" in markdown and "65.86%" in markdown
    assert "first test episode" in markdown
    assert "STOP_WITH_BREACH=1" in markdown


@pytest.mark.parametrize("corruption", ["duplicate", "missing_policy", "nonfinite"])
def test_malformed_saved_financial_results_are_rejected(tmp_path, corruption):
    rows = saved_results(tmp_path)
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "missing_policy":
        rows.pop()
    else:
        rows[0]["hedge_error"] = float("nan")
    with (tmp_path / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(InputError):
        paired_report(tmp_path)


def test_latest_blocked_run_cannot_reuse_old_results(tmp_path):
    saved_results(tmp_path)
    blocked = tmp_path / "blocked.json"
    blocked.write_text("{}")
    timestamp = (tmp_path / "summary.json").stat().st_mtime_ns + 1000000
    os.utime(blocked, ns=(timestamp, timestamp))
    with pytest.raises(InputError, match="blocked"):
        paired_report(tmp_path)


def test_no_matched_completion_keeps_failures_without_invented_statistics(tmp_path):
    rows = saved_results(tmp_path)
    for row in rows:
        if row["policy"] == "volgan":
            row.update(status="STOP_DATA_GAP", hedge_error=None, net_pnl=None)
    with (tmp_path / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "summary.json").write_text(json.dumps(dict(metrics=summarize(rows))))
    result = paired_report(tmp_path)
    assert result["matched_completed"] == 0
    assert all(row["hedge_error_rmse_difference"] is None for row in result["comparisons"])
    assert result["status_counts"]["volgan"]["STOP_DATA_GAP"] == 3


def test_solver_failure_remains_separate_from_engine_breach_and_terminal_metrics(tmp_path):
    rows = saved_results(tmp_path)
    rows[-1]["status"] = "FAILED_OPTIMIZER"
    with (tmp_path / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "summary.json").write_text(json.dumps(dict(metrics=summarize(rows))))
    result = paired_report(tmp_path)
    assert result["matched_completed"] == 2
    assert result["optimizer_failures"]["volgan"] == 1
    assert result["status_counts"]["volgan"] == {"COMPLETED": 2, "FAILED_OPTIMIZER": 1}
    assert result["excluded_episode_ids"] == ["2020-03-02"]
    markdown = (tmp_path / "benchmark.md").read_text()
    assert "Optimizer failures" in markdown
    assert "FAILED_OPTIMIZER=1" in markdown
    assert "failed_runs.json" in markdown
    assert "conditional on successful completion" in markdown


def test_bootstrap_statistics_keep_missing_slots_and_hand_computed_tail():
    sample = np.array([
        [[-10, 8, 100, 2, 3], [10, -12, 100, 2, 3], [np.nan] * 5],
        [[5, -8, 120, 3, 4], [np.nan] * 5, [np.nan] * 5],
        [[np.nan] * 5] * 3])
    values, counts = _bootstrap_statistics(sample)
    assert counts.tolist() == [2, 1, 0]
    assert values["mean_absolute_error_difference"][:2].tolist() == [10, 5]
    assert values["hedge_error_rmse_difference"][:2].tolist() == [10, 5]
    assert values["hedge_error_sd_difference"][0] == pytest.approx(200 ** .5)
    assert np.isnan(values["hedge_error_sd_difference"][1:]).all()
    assert values["cvar95_difference"][:2].tolist() == [12, 8]
    assert values["mean_trading_cost_difference"][:2].tolist() == [2, 3]
    assert values["mean_turnover_difference"][:2].tolist() == [100, 120]
    assert all(np.isnan(value[2]) for value in values.values())


def test_bootstrap_cvar_uses_four_tail_observations_for_62_completed_episodes():
    sample = np.zeros((1, 71, 5))
    sample[0, :62, 1] = -np.arange(1, 63)
    sample[0, 62:] = np.nan
    values, counts = _bootstrap_statistics(sample)
    assert counts.tolist() == [62]
    assert values["cvar95_difference"].tolist() == [(59 + 60 + 61 + 62) / 4]


def test_moving_blocks_are_deterministic_contiguous_and_do_not_wrap():
    draws = _moving_block_indices(7, 100, 3, 20261004)
    assert np.array_equal(draws, _moving_block_indices(7, 100, 3, 20261004))
    assert draws.shape == (100, 7)
    assert draws.min() >= 0 and draws.max() <= 6
    assert (draws[:, 0] <= 4).all() and (draws[:, 3] <= 4).all() and (draws[:, 6] <= 4).all()
    assert np.all(np.diff(draws[:, :3], axis=1) == 1)
    assert np.all(np.diff(draws[:, 3:6], axis=1) == 1)
    assert _moving_block_indices(3, 2, 3, 17).tolist() == [[0, 1, 2], [0, 1, 2]]
    with pytest.raises(InputError, match="bootstrap"):
        _moving_block_indices(3, 100, 4, 17)


def correction_results(tmp_path):
    frozen, current = tmp_path / "frozen", tmp_path / "current"
    frozen.mkdir()
    current.mkdir()
    saved_results(frozen)
    rows = saved_results(current, failed=False)
    decisions = [dict(episode=row["episode"], policy=row["policy"], scheduled_decisions=2,
        decisions_made=1 if row["episode"] == "2020-02-02" else 2,
        decisions_skipped=int(row["episode"] == "2020-02-02"),
        skips=[dict(time="2020-02-02", reason="missing consecutive history")]
              if row["episode"] == "2020-02-02" else [],
        sufficient_history_by_data=row["episode"] != "2020-02-02") for row in rows]
    (current / "decisions.json").write_text(json.dumps(decisions))
    return current, frozen


def test_correction_populations_and_paired_uncertainty_are_explicit(tmp_path):
    current, frozen = correction_results(tmp_path)
    frozen_before = {path.name: path.read_bytes() for path in frozen.iterdir()}
    result = reliability_report(current, frozen, repetitions=200, block_length=1)
    assert result["total_scheduled"] == 3
    assert result["common_completed_episode_ids"] == ["2020-01-02", "2020-02-02", "2020-03-02"]
    assert result["common_sufficient_history_episode_ids"] == ["2020-01-02", "2020-03-02"]
    assert result["frozen_common_matched_episode_ids"] == ["2020-01-02", "2020-02-02"]
    assert result["completion_rates"] == {policy: 1. for policy in POLICIES}
    own = result["metrics"]["own_completed"][0]
    assert own["completed_episodes"] == 3 and own["scheduled_episodes"] == 3
    assert own["hedge_error_sd"] == pytest.approx(((1000200 - 1000000 / 3) / 2) ** .5)
    assert own["hedge_error_rmse"] == pytest.approx((1000200 / 3) ** .5)
    assert own["cvar95"] == 1002 and own["cvar95_tail_episodes"] == 1
    assert result["metrics"]["frozen_common_matched_original"] == result["metrics"]["frozen_common_matched_corrected"]
    interval = next(row for row in result["uncertainty"]
                    if row["population"] == "frozen_common_matched_corrected"
                    and row["policy"] == "delta" and row["reference"] == "unhedged"
                    and row["metric"] == "mean_absolute_error_difference")
    # Both paired absolute differences are -5, regardless of chronological resampling.
    assert interval["estimate"] == -5
    assert interval["lower_95"] == -5 and interval["upper_95"] == -5
    assert interval["valid_replicates"] <= 200
    assert interval["observed_cvar95_tail"] == 1
    assert result["decision_summary"]["delta"]["decisions_skipped"] == 1
    assert result["decision_summary"]["delta"]["scheduled_decisions"] == 6
    assert {path.name: path.read_bytes() for path in frozen.iterdir()} == frozen_before
    assert json.loads((current / "reliability_populations.json").read_text()) == result
    assert reliability_report(current, frozen, repetitions=200, block_length=1) == result


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "policy_dependent"])
def test_history_sensitivity_rejects_incomplete_or_policy_dependent_membership(tmp_path, corruption):
    current, frozen = correction_results(tmp_path)
    path = current / "decisions.json"
    records = json.loads(path.read_text())
    if corruption == "duplicate":
        records.append(records[0])
    elif corruption == "missing":
        records.pop()
    else:
        records[0]["sufficient_history_by_data"] = False
    path.write_text(json.dumps(records))
    with pytest.raises(InputError):
        reliability_report(current, frozen, repetitions=10, block_length=1)
