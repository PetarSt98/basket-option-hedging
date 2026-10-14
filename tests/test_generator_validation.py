"""Independent statistical arithmetic and conditional-validation reproducibility."""

from datetime import datetime, timezone
import json

import numpy as np
import pytest

from basket_hedging.contracts import InputError
from basket_hedging.generator_validation import compare_return_samples, energy_score, validate_generator


@pytest.mark.parametrize("observation,samples,expected", [
    ([0.], [[0.], [2.]], 0.5),
    ([3.], [[0.], [2.]], 1.5),
    ([0., 0.], [[0., 0.], [3., 4.]], 1.25),
    ([1., 2.], [[1., 2.], [1., 2.]], 0.),
])
def test_energy_score_hand_calculated_distances(observation, samples, expected):
    # The two off-diagonal distances count in N**2, along with two zeros.
    assert energy_score(observation, samples) == pytest.approx(expected)


def test_energy_score_preserves_joint_geometry_raw_units_and_inputs():
    observation = np.array([0.002, -0.01])
    samples = np.array([[0.01, 0.02], [-0.03, 0.01], [0.004, -0.01]])
    original_observation, original_samples = observation.copy(), samples.copy()
    score = energy_score(observation, samples)
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    assert energy_score(observation, samples[[2, 0, 1]]) == pytest.approx(score)
    assert energy_score(observation[::-1], samples[:, ::-1]) == pytest.approx(score)
    assert energy_score(observation @ rotation, samples @ rotation) == pytest.approx(score)
    assert energy_score(observation * 100, samples * 100) == pytest.approx(score * 100)
    np.testing.assert_array_equal(observation, original_observation)
    np.testing.assert_array_equal(samples, original_samples)


def test_energy_score_distinguishes_dependence_with_identical_marginals():
    positive = np.array([[-1., -1.], [1., 1.]])
    negative = np.array([[-1., 1.], [1., -1.]])
    np.testing.assert_array_equal(np.sort(positive, axis=0), np.sort(negative, axis=0))
    matching = np.mean([energy_score(outcome, positive) for outcome in positive])
    wrong_dependence = np.mean([energy_score(outcome, negative) for outcome in positive])
    assert matching == pytest.approx(np.sqrt(2) / 2)
    assert wrong_dependence == pytest.approx(2 - np.sqrt(2) / 2)
    assert matching < wrong_dependence


@pytest.mark.parametrize("observation,samples", [
    (0., [[0.], [1.]]), ([[0.]], [[0.], [1.]]),
    ([0.], [0., 1.]), ([0.], [[0.]]),
    ([], [[], []]), ([0., 1.], [[0.], [1.]]),
    ([np.nan], [[0.], [1.]]), ([np.inf], [[0.], [1.]]),
    ([0.], [[0.], [np.nan]]), ([0.], [[0.], [-np.inf]]),
    ([1e308], [[-1e308], [0.]]),
])
def test_energy_score_rejects_invalid_dimensions_or_nonfinite_arithmetic(observation, samples):
    with pytest.raises(InputError):
        energy_score(observation, samples)


def test_independent_joint_and_basket_moments_quantiles_and_errors():
    # Means (0,0), sample variances (1,4), covariance 2 and correlation 1.
    reference = np.array([[-1., -2.], [0., 0.], [1., 2.]])
    generated = np.array([[0., -4.], [1., 0.], [2., 4.]])
    report = compare_return_samples(reference, generated, basket_weights=(0.25, 0.75))
    assert report["assets"]["mean"]["reference"] == [0, 0]
    assert report["assets"]["mean"]["generated"] == [1, 0]
    assert report["assets"]["mean"]["relative_absolute_error"] == [None, None]
    assert report["assets"]["variance"]["reference"] == [1, 4]
    assert report["assets"]["variance"]["generated"] == [1, 16]
    assert report["assets"]["variance"]["relative_absolute_error"] == [0, 3]
    assert report["assets"]["covariance"]["reference"] == [[1, 2], [2, 4]]
    assert report["assets"]["covariance"]["generated"] == [[1, 4], [4, 16]]
    assert report["assets"]["correlation"]["reference"] == [[1, 1], [1, 1]]
    assert report["assets"]["quantiles"]["0.05"]["reference"] == pytest.approx([-0.9, -1.8])
    assert report["assets"]["wasserstein_1"] == pytest.approx([1, 4 / 3])
    # Fixed-value weights give basket samples [-1.75,0,1.75], [-3,.25,3.5].
    assert report["basket"]["mean"]["generated"] == 0.25
    assert report["basket"]["variance"]["reference"] == 3.0625
    assert report["basket"]["variance"]["generated"] == 10.5625
    assert report["basket"]["quantiles"]["0.95"]["generated"] == pytest.approx(3.175)
    assert report["basket"]["wasserstein_1"] == pytest.approx(3.25 / 3)


def test_one_asset_and_zero_variance_metrics_have_strict_json_values():
    report = compare_return_samples(np.zeros((3, 1)), np.ones((3, 1)), basket_weights=(1,))
    assert report["assets"]["covariance"]["reference"] == [[0]]
    assert report["assets"]["correlation"]["reference"] == [[None]]
    assert report["assets"]["correlation"]["absolute_error"] == [[None]]
    assert report["basket"]["wasserstein_1"] == 1
    json.dumps(report, allow_nan=False)


class FixedSampler:
    stock_ids = ("A", "B")
    lags = 3
    available_at = datetime(2025, 12, 1, tzinfo=timezone.utc)

    def __init__(self):
        self.calls = []

    def sample_log_returns(self, features, horizon_days, normals):
        self.calls.append((features.copy(), horizon_days, normals.copy()))
        # An intentionally imperfect learned law keeps the validation independent.
        return np.sqrt(horizon_days) * normals @ np.array([[0.01, 0.005], [0, 0.02]])


def validation(model, **overrides):
    settings = dict(initial_spots={"A": 100., "B": 50.}, volatilities=(0.2, 0.3),
                    correlation=((1., 0.4), (0.4, 1.)), coefficients={"A": 1., "B": 2.},
                    samples=12, seed=94)
    settings.update(overrides)
    return validate_generator(model, **settings)


def test_validation_repeats_with_independent_streams_causal_states_and_fixed_share_units():
    first, second = FixedSampler(), FixedSampler()
    report = validation(first)
    assert report == validation(second)
    assert report["basket_value_weights"] == [0.5, 0.5]  # 100*1 equals 50*2.
    assert len(report["cases"]) == 8
    assert {case["horizon_days"] for case in report["cases"]} == {1, 3}
    assert {case["state"] for case in report["cases"]} == {
        "normal", "recent_positive", "recent_negative", "cold_start"}
    seeds = [case[key] for case in report["cases"] for key in ("reference_seed", "learned_seed")]
    assert len(seeds) == len(set(seeds))
    assert all(case["decision_time"] > first.available_at.isoformat() for case in report["cases"])
    for case in report["cases"]:
        decision = datetime.fromisoformat(case["decision_time"])
        horizon = datetime.fromisoformat(case["horizon"])
        assert horizon.weekday() < 5
        assert (decision.weekday() == 4) == (case["horizon_days"] == 3)
        assert (horizon - decision).total_seconds() == case["horizon_days"] * 86400
    assert len(report["cases"][0]["input_ids"]) > len(report["cases"][-1]["input_ids"])
    assert report["cases"][0]["history_times"] == [
        "2026-01-02T09:00:00+00:00", "2026-01-02T16:00:00+00:00",
        "2026-01-05T09:00:00+00:00", "2026-01-05T16:00:00+00:00"]
    assert not np.array_equal(first.calls[0][0], first.calls[2][0])
    assert not np.array_equal(first.calls[2][0], first.calls[4][0])
    assert not np.array_equal(first.calls[0][2], first.calls[1][2])
    for left, right in zip(first.calls, second.calls):
        np.testing.assert_array_equal(left[2], right[2])
    json.dumps(report, allow_nan=False)


def test_zero_volatility_reference_is_exact_without_stochastic_expected_values():
    report = validation(FixedSampler(), volatilities=(0., 0.))
    for case in report["cases"]:
        assert case["assets"]["mean"]["reference"] == [0, 0]
        assert case["assets"]["variance"]["reference"] == [0, 0]
        assert case["basket"]["variance"]["reference"] == 0


@pytest.mark.parametrize("output", [np.zeros((12, 1)), np.full((12, 2), np.nan),
                                    np.full((12, 2), 1000.)])
def test_invalid_generated_samples_are_rejected_instead_of_reported_as_comparable(output):
    model = FixedSampler()
    model.sample_log_returns = lambda features, horizon_days, normals: output
    with pytest.raises(InputError):
        validation(model)


@pytest.mark.parametrize("changes", [dict(samples=1), dict(seed=-1),
                                    dict(initial_spots=(100, 0)), dict(coefficients=(0, 0)),
                                    dict(coefficients={"A": 1}),
                                    dict(correlation=((1, 2), (2, 1)))])
def test_validation_rejects_invalid_inputs(changes):
    with pytest.raises(InputError):
        validation(FixedSampler(), **changes)


@pytest.mark.parametrize("reference,generated,weights", [
    ([[0], [1]], [[0]], (1,)), ([[0], [float("nan")]], [[0], [1]], (1,)),
    ([[0], [1]], [[0], [1]], (0.5,)),
])
def test_return_metric_validation(reference, generated, weights):
    with pytest.raises(InputError):
        compare_return_samples(reference, generated, basket_weights=weights)
