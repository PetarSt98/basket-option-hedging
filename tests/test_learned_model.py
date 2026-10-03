"""Training isolation and independently calculated Gaussian inference checks."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import subprocess
import sys

import numpy as np
import pytest

from basket_hedging.contracts import InputError
from basket_hedging.learned_model import ConditionalGaussianModel, fit_generator
from basket_hedging.learning_data import TrainingData


AVAILABLE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def dataset(split="train", seed=11, rows=32):
    random = np.random.default_rng(seed)
    days = np.resize([1., 3.], rows)
    features = np.column_stack((np.resize([-1., 1.], rows), random.normal(size=rows), np.log(days)))
    normals = random.normal(size=(rows, 2))
    returns = .01 * np.sqrt(days[:, None]) * (normals @ np.array([[1., .4], [0., 1.2]]))
    start = datetime(2020 if split == "train" else 2022, 1, 1, tzinfo=timezone.utc)
    decisions = tuple(start + timedelta(days=4*index) for index in range(rows))
    targets = tuple(time + timedelta(days=float(horizon)) for time, horizon in zip(decisions, days))
    return TrainingData(("S1", "S2"), 0, features, returns, days, decisions, targets,
                        (f"{split}-path",) * rows, split, (seed,))


def fit(train=None, validation=None, **settings):
    pytest.importorskip("torch")
    options = dict(seed=23, epochs=8, hidden=4, batch_size=16, available_at=AVAILABLE)
    options.update(settings)
    return fit_generator(train or dataset(), validation or dataset("validation", 12), **options)


def fixed_model():
    # mu=(.4,-.2), L=[[2,0],[.5,3]]. Softplus inverse gives exact chosen diagonal.
    output = np.array([.4, -.2, np.log(np.expm1(2-1e-4)), .5, np.log(np.expm1(3-1e-4))])
    return ConditionalGaussianModel(("S1", "S2"), 0, np.zeros(3), np.ones(3),
        (np.zeros((2, 3)), np.zeros((2, 2)), np.zeros((5, 2))),
        (np.zeros(2), np.zeros(2), output), AVAILABLE, AVAILABLE - timedelta(days=1))


def test_fixed_density_has_independent_mean_covariance_draws_and_calendar_scaling():
    model = fixed_model()
    expected_mean = np.array([.004, -.002])
    expected_covariance = 1e-4 * np.array([[4., 1.], [1., 9.25]])
    one_mean, one_covariance = model.moments(np.zeros(3), 1.)
    three_mean, three_covariance = model.moments(np.zeros(3), 3.)
    np.testing.assert_allclose(one_mean, expected_mean, rtol=0, atol=1e-16)
    np.testing.assert_allclose(one_covariance, expected_covariance, rtol=0, atol=1e-16)
    np.testing.assert_allclose(three_mean, 3*expected_mean, rtol=0, atol=1e-16)
    np.testing.assert_allclose(three_covariance, 3*expected_covariance, rtol=0, atol=1e-16)
    draws = model.sample_log_returns(np.zeros(3), 1., np.array([[1., 0.], [0., 1.], [-1., -1.]]))
    np.testing.assert_allclose(draws, expected_mean + .01*np.array([[2., .5], [0., 3.], [-2., -3.5]]),
                               rtol=0, atol=1e-16)
    assert one_covariance[0, 1] > 0  # Joint sampling retains the learned off-diagonal dependence.


def test_training_is_reproducible_and_preserves_global_rngs_and_thread_configuration():
    torch = pytest.importorskip("torch")
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    first, second = fit(), fit()
    assert first.training_history == second.training_history
    for left, right in zip(first.weights + first.biases, second.weights + second.biases):
        np.testing.assert_array_equal(left, right)
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads


def test_preprocessing_is_train_only_and_validation_selects_saved_checkpoint():
    train, validation = dataset(), dataset("validation", 12)
    shifted = validation.features.copy()
    shifted[:, :2] += 100
    model = fit(train, replace(validation, features=shifted))
    assert model.feature_mean[0] == 0. and model.feature_scale[0] == 1.
    np.testing.assert_array_equal(model.feature_mean, np.mean(train.features, axis=0))
    np.testing.assert_array_equal(model.feature_scale, np.std(train.features, axis=0))
    assert model.trained_through == max(validation.target_times)
    best = min(model.training_history, key=lambda row: row["validation_nll"])
    assert model.metadata["best_epoch"] == best["epoch"]
    losses = []
    for features, days, realized in zip(shifted, validation.horizon_days, validation.log_returns):
        mean, covariance = model.moments(features, float(days))
        # Independent full-covariance Gaussian density, then remove the fixed
        # Jacobian for the normalized-return loss used during training.
        residual = realized - mean
        raw_nll = .5*(2*np.log(2*np.pi) + np.linalg.slogdet(covariance)[1]
                      + residual @ np.linalg.solve(covariance, residual))
        losses.append(raw_nll - 2*np.log(.01*np.sqrt(days)))
    np.testing.assert_allclose(np.mean(losses), best["validation_nll"], rtol=0, atol=1e-9)


def test_constant_train_feature_uses_unit_scale():
    train = dataset()
    features = train.features.copy()
    features[:, 1] = 7.
    model = fit(replace(train, features=features), epochs=1)
    assert model.feature_mean[1] == 7. and model.feature_scale[1] == 1.


@pytest.mark.parametrize("conflict", ["path", "seed", "chronology", "availability", "schema"])
def test_split_and_information_cutoff_checks_reject_leakage(conflict):
    train, validation = dataset(), dataset("validation", 12)
    settings = {}
    if conflict == "path":
        validation = replace(validation, path_ids=train.path_ids)
    elif conflict == "seed":
        validation = replace(validation, seeds=train.seeds)
    elif conflict == "chronology":
        validation = replace(validation, decision_times=train.decision_times, target_times=train.target_times)
    elif conflict == "availability":
        settings["available_at"] = min(validation.decision_times)
    else:
        validation = replace(validation, stock_ids=("S2", "S1"))
    with pytest.raises(InputError):
        fit(train, validation, epochs=1, **settings)


def test_saved_numpy_model_matches_training_without_mutable_parameter_aliases(tmp_path):
    model = fit(epochs=2)
    path = model.save(tmp_path / "model.json")
    loaded = ConditionalGaussianModel.load(path)
    features, normals = np.array([.5, -.2, np.log(3)]), np.array([[1., -1.], [0., 2.]])
    np.testing.assert_array_equal(model.sample_log_returns(features, 3., normals),
                                  loaded.sample_log_returns(features, 3., normals))
    assert model.metadata == loaded.metadata and model.training_history == loaded.training_history
    for values in loaded.weights + loaded.biases + (loaded.feature_mean, loaded.feature_scale):
        with pytest.raises(ValueError):
            values.flat[0] = 999
        with pytest.raises(ValueError):
            values.setflags(write=True)
    with pytest.raises(TypeError):
        loaded.metadata["epochs"] = 999


def test_saved_model_inference_does_not_import_optional_torch(tmp_path):
    path = fixed_model().save(tmp_path / "model.json")
    script = """
import builtins, json, sys
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name.split('.')[0] == 'torch':
        raise ImportError('torch deliberately unavailable for inference')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
from basket_hedging.learned_model import ConditionalGaussianModel
model = ConditionalGaussianModel.load(sys.argv[1])
print(json.dumps(model.moments([0., 0., 0.], 1.)[0].tolist()))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(path)],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [.004, -.002]


@pytest.mark.parametrize("fault", ["version", "missing", "shape", "nonfinite", "cutoff"])
def test_malformed_saved_models_are_rejected(tmp_path, fault):
    path = fixed_model().save(tmp_path / "model.json")
    raw = json.loads(path.read_text())
    if fault == "version":
        raw["model_version"] = "another-model"
    elif fault == "missing":
        del raw["feature_mean"]
    elif fault == "shape":
        raw["weights"][0] = [[0.]]
    elif fault == "nonfinite":
        raw["biases"][0][0] = float("nan")
    else:
        raw["trained_through"] = (AVAILABLE + timedelta(days=1)).isoformat()
    path.write_text(json.dumps(raw))
    with pytest.raises(InputError):
        ConditionalGaussianModel.load(path)


@pytest.mark.parametrize("features,days,normals", [
    ([0., 0.], 1., [[0., 0.]]), ([0., 0., float("nan")], 1., [[0., 0.]]),
    ([0., 0., 0.], 0., [[0., 0.]]), ([0., 0., 0.], float("inf"), [[0., 0.]]),
    ([0., 0., 0.], 1., [[0.]]), ([0., 0., 0.], 1., [[float("inf"), 0.]]),
])
def test_invalid_inference_inputs_are_rejected(features, days, normals):
    with pytest.raises(InputError):
        fixed_model().sample_log_returns(features, days, normals)
