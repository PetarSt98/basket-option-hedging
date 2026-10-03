"""Independent ensemble arithmetic and a small reproducible end-to-end run."""

import json
from math import sqrt

import pytest

from basket_hedging.contracts import InputError
from basket_hedging.stage2 import POLICIES, aggregate_results, run_stage2


def outcome(path, policy, pnl, error, status="COMPLETED"):
    return dict(path=path, policy=policy, terminal_pnl=pnl, terminal_hedge_error=error,
                status=status, turnover=100, transaction_costs=2, fills=3,
                breaches=int(status == "STOP_WITH_BREACH"), skipped_decisions=0,
                rejected_orders=0, optimizer_decisions=1)


def test_ensemble_metrics_use_independent_sample_moments_and_loss_tail():
    rows = [outcome(path, policy, pnl, error)
            for policy in POLICIES for path, (pnl, error) in enumerate(((10., -2.), (-20., 4.)))]
    result = aggregate_results(rows)
    for record in result["outcomes"]:
        assert record["mean_terminal_pnl"] == -5
        assert record["terminal_pnl_sd"] == pytest.approx(sqrt(450))
        assert record["hedge_error_mean"] == 1
        assert record["hedge_error_sd"] == pytest.approx(sqrt(18))
        assert record["hedge_error_rmse"] == pytest.approx(sqrt(10))
        assert record["cvar95_loss"] == 20
        assert record["completion_frequency"] == 1
        assert record["breach_frequency"] == 0
    # Identical risk across policies means a zero reduction denominator.
    assert result["variance_reduction_retained"] is None


def test_failed_paths_remain_in_frequencies_and_activity_but_not_matched_terminal_risk():
    rows = [outcome(path, policy, path * 100., path * 20.) for path in range(2) for policy in POLICIES]
    rows[-1] = outcome(1, "learned_generator", None, None, "STOP_WITH_BREACH")
    result = aggregate_results(rows)
    assert result["total_paths"] == 2 and result["matched_completed_paths"] == 1
    for record in result["outcomes"]:
        assert record["terminal_pnl_sd"] is None
        assert record["mean_terminal_pnl"] == 0
        assert record["mean_fills"] == 3
    assert result["outcomes"][-1]["breach_frequency"] == .5
    assert result["outcomes"][-1]["completion_frequency"] == .5
    assert result["outcomes"][0]["completion_frequency"] == 1


def test_retained_variance_reduction_is_not_clamped():
    scale = dict(unhedged=4., delta=2., true_simulator=2., learned_generator=1.)
    rows = [outcome(path, policy, 0., direction * scale[policy])
            for path, direction in enumerate((-1, 1)) for policy in POLICIES]
    assert aggregate_results(rows)["variance_reduction_retained"] == pytest.approx(15 / 12)
    with pytest.raises(InputError):
        aggregate_results(rows[:-1])
    rows[-1]["policy"] = "delta"
    with pytest.raises(InputError):
        aggregate_results(rows)


def test_small_complete_pipeline_repeats_with_identical_artifacts(tmp_path):
    pytest.importorskip("torch")
    options = dict(test_paths=2, workers=1, train_paths=8, validation_paths=4, epochs=2,
                   days=2, scenarios=8, pricing_samples=32, accounting_samples=32,
                   diagnostic_samples=16, progress=lambda message: None)
    first = run_stage2(tmp_path / "first", **options)
    second = run_stage2(tmp_path / "second", **options)
    assert first == second
    assert first["matched_completed_paths"] == 2
    for record in first["outcomes"]:
        assert record["completion_frequency"] == 1
        assert record["total_skipped_decisions"] == 0
    assert first["outcomes"][-1]["total_optimizer_decisions"] == 4
    for name in ("model.json", "training_history.csv", "generator_validation.json", "path_results.csv",
                 "hedging_summary.json", "test_manifest.json", "experiment.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    experiment = json.loads((tmp_path / "first" / "experiment.json").read_text())
    train = set(experiment["train"]["seeds"])
    validation = set(experiment["validation"]["seeds"])
    test = set(experiment["test"]["seeds"])
    assert not (train & validation or train & test or validation & test)
    assert experiment["test"]["used_for_training_or_selection"] is False


@pytest.mark.parametrize("changes", [{"test_paths": 0}, {"workers": True}, {"days": -1},
                                     {"scenarios": 1}, {"pricing_samples": 1},
                                     {"accounting_samples": 1}, {"diagnostic_samples": 1}])
def test_invalid_experiment_counts_fail_before_work(tmp_path, changes):
    with pytest.raises(InputError):
        run_stage2(tmp_path, **changes)
    assert list(tmp_path.iterdir()) == []
