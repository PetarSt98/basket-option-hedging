"""Independent feature/calendar expectations and synthetic split reproducibility."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import log, sqrt
import os
import subprocess
import sys

import numpy as np
import pytest

from basket_hedging.contracts import InputError
from basket_hedging import learning_data
from basket_hedging.learning_data import (
    TrainingData, conditioning_features, experiment_seed, generate_learning_data,
)
from basket_hedging.market import Observation


START = datetime(2024, 1, 8, 16, tzinfo=timezone.utc)
STOCKS = ("A", "B", "C")
CORRELATION = ((1, .35, .35), (.35, 1, .35), (.35, .35, 1))


def frame(time, values, *, available=None, tag="obs"):
    return [Observation(f"{tag}/{time.isoformat()}/{stock}", stock, value, time, available or time)
            for stock, value in values.items()]


def generated(**options):
    kwargs = dict(split="train", paths=2, start=START, days=6, lags=3)
    kwargs.update(options)
    return generate_learning_data(STOCKS, (100, 80, 120), (.2, .225, .25), CORRELATION, **kwargs)


def test_cold_start_features_have_prices_horizon_and_explicit_missing_masks():
    features, ids = conditioning_features(STOCKS, [], {"A": 100, "B": 80, "C": 120},
                                          START, START+timedelta(days=3))
    assert features.shape == (19,)
    assert features[:3] == pytest.approx([log(100), log(80), log(120)])
    assert features[3:18].tolist() == [0]*15
    assert features[-1] == pytest.approx(log(3))
    assert ids == ()
    with pytest.raises(ValueError):
        features.setflags(write=True)


def test_joint_return_lags_are_newest_first_and_use_actual_elapsed_days():
    friday = datetime(2024, 1, 5, 16, tzinfo=timezone.utc)
    monday_open = datetime(2024, 1, 8, 9, tzinfo=timezone.utc)
    monday_close = START
    history = (frame(friday, {"A": 100, "B": 50})
               + frame(monday_open, {"A": 110, "B": 55})
               + frame(monday_close, {"A": 121, "B": 49.5}))
    features, ids = conditioning_features(("A", "B"), history, {"A": 121, "B": 49.5},
                                          monday_close, monday_close+timedelta(days=1), lags=2)
    expected = [log(121), log(49.5), log(1.1)/sqrt(7/24), log(.9)/sqrt(7/24),
                log(1.1)/sqrt(65/24), log(1.1)/sqrt(65/24), 1, 1, log(7/24), log(65/24), 0]
    assert features == pytest.approx(expected, abs=1e-14)
    assert ids == tuple(observation.id for observation in history)


def test_partial_joint_frame_is_not_filled_using_future_or_other_times():
    t0, t1, t2 = START, START+timedelta(days=1), START+timedelta(days=2)
    history = frame(t0, {"A": 100, "B": 50}) + frame(t1, {"A": 900}) + frame(t2, {"A": 110, "B": 55})
    features, ids = conditioning_features(("A", "B"), history, {"A": 110, "B": 55},
                                          t2, t2+timedelta(days=1), lags=2)
    assert features[2:6] == pytest.approx([log(1.1)/sqrt(2), log(1.1)/sqrt(2), 0, 0])
    assert features[6:8].tolist() == [1, 0]
    assert features[8:10] == pytest.approx([log(2), 0])
    assert history[2].id not in ids


def test_future_and_delayed_history_mutations_leave_current_features_unchanged():
    t0, now, future = START, START+timedelta(days=1), START+timedelta(days=2)
    past = frame(t0, {"A": 100, "B": 50}) + frame(now, {"A": 110, "B": 55})
    baseline = conditioning_features(("A", "B"), past, {"A": 110, "B": 55}, now, future)
    delayed = frame(t0+timedelta(hours=2), {"A": 1, "B": 10000}, available=future, tag="delayed")
    for values in ({"A": 1, "B": 10000}, {"A": 1000000, "B": .01}):
        candidate = conditioning_features(("A", "B"), past+delayed+frame(future, values),
                                            {"A": 110, "B": 55}, now, future)
        assert np.array_equal(candidate[0], baseline[0])
        assert candidate[1] == baseline[1]


def test_conflicting_released_values_fail_but_identical_duplicates_are_stable():
    history = frame(START, {"A": 100})
    duplicate = Observation("same-price", "A", 100, START, START)
    one = conditioning_features(("A",), history+[duplicate], {"A": 100}, START, START+timedelta(days=1))
    two = conditioning_features(("A",), [duplicate]+history, {"A": 100}, START, START+timedelta(days=1))
    assert np.array_equal(one[0], two[0]) and one[1] == two[1]
    with pytest.raises(InputError, match="conflicting stock values"):
        conditioning_features(("A",), history+[replace(duplicate, value=101)], {"A": 100},
                              START+timedelta(hours=1), START+timedelta(days=1))
    with pytest.raises(InputError, match="current spot conflicts"):
        conditioning_features(("A",), history, {"A": 101}, START, START+timedelta(days=1))


def test_zero_lags_use_only_current_prices_and_target_duration():
    vector, _ = conditioning_features(("A",), frame(START, {"A": 100}), {"A": 100},
                                      START, START+timedelta(days=3), lags=0)
    assert vector == pytest.approx([log(100), log(3)])


def test_generated_rows_match_calendar_dimensions_and_cold_starts():
    data = generated()
    assert data.features.shape == (12, 19)
    assert data.log_returns.shape == (12, 3)
    assert data.horizon_days.tolist() == [1, 1, 1, 1, 3, 1]*2
    assert data.decision_times[:6] == tuple(datetime(2024, 1, day, 16, tzinfo=timezone.utc)
                                           for day in (8, 9, 10, 11, 12, 15))
    assert data.target_times[:6] == tuple(datetime(2024, 1, day, 16, tzinfo=timezone.utc)
                                         for day in (9, 10, 11, 12, 15, 16))
    assert data.path_ids == ("train/00000",)*6 + ("train/00001",)*6
    assert data.features[[0, 6], 3:18].tolist() == [[0]*15, [0]*15]
    assert data.features[1, 12:15].tolist() == [1, 1, 0]  # Overnight and open-to-close.
    assert data.features[2, 12:15].tolist() == [1, 1, 1]
    assert len(data.seeds) == 2
    assert not hasattr(data, "volatilities") and not hasattr(data, "correlation")


def test_zero_volatility_has_independent_zero_labels_and_known_features():
    data = generate_learning_data(("S",), (100,), (0,), ((1,),), split="train",
                                  paths=1, start=START, days=2, lags=1)
    assert data.log_returns.tolist() == [[0], [0]]
    assert data.features[0] == pytest.approx([log(100), 0, 0, 0, 0])
    assert data.features[1] == pytest.approx([log(100), 0, 1, log(7/24), 0])


def test_labels_use_next_close_and_mutating_later_prices_cannot_change_features(monkeypatch):
    values = [100, 110, 121, 130, 144]

    def hand_path(stock_ids, initial_spots, volatilities, correlation, times, seed):
        return iter((time, {"S": value}) for time, value in zip(times, values))

    monkeypatch.setattr(learning_data, "market_path", hand_path)
    kwargs = dict(split="train", paths=1, start=START, days=2, lags=1)
    before = generate_learning_data(("S",), (100,), (.2,), ((1,),), **kwargs)
    assert before.log_returns[:, 0] == pytest.approx([log(121/100), log(144/121)])
    assert before.features[1] == pytest.approx([log(121), log(1.1)/sqrt(7/24), 1, log(7/24), 0])
    values[-2:] = [.1, 1000000]
    after = generate_learning_data(("S",), (100,), (.2,), ((1,),), **kwargs)
    assert np.array_equal(before.features, after.features)
    assert before.log_returns[0, 0] == after.log_returns[0, 0]
    assert after.log_returns[1, 0] == pytest.approx(log(1000000/121))


def test_synthetic_generation_repeats_and_splits_have_disjoint_seed_roots():
    train = generated()
    repeat = generated()
    validation = generated(split="validation", start=datetime(2025, 1, 6, tzinfo=timezone.utc))
    assert np.array_equal(train.features, repeat.features)
    assert np.array_equal(train.log_returns, repeat.log_returns)
    assert train.manifest() == repeat.manifest()
    assert set(train.seeds).isdisjoint(validation.seeds)
    assert set(train.path_ids).isdisjoint(validation.path_ids)
    assert max(train.target_times) < min(validation.decision_times)
    assert not np.array_equal(train.log_returns, validation.log_returns)
    assert set(train.seeds).isdisjoint(experiment_seed("test", i) for i in range(2))
    manifest = train.manifest()
    assert manifest["rows"] == 12 and manifest["paths"] == 2 and manifest["feature_count"] == 19
    assert len(manifest["dataset_sha256"]) == 64
    changed = replace(train, log_returns=train.log_returns+.01)
    assert changed.manifest()["dataset_sha256"] != manifest["dataset_sha256"]


def test_data_arrays_are_detached_and_cannot_regain_write_permission():
    original = generated(paths=1, days=2)
    features = original.features.copy()
    data = replace(original, features=features)
    features[0, 0] = -100
    assert data.features[0, 0] == pytest.approx(log(100))
    for array in (data.features, data.log_returns, data.horizon_days):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_stable_seed_is_independent_of_python_hash_randomization():
    command = "from basket_hedging.learning_data import experiment_seed; print(experiment_seed('train-path',0))"
    results = [subprocess.run([sys.executable, "-B", "-c", command], check=True, capture_output=True, text=True,
                              env={**os.environ, "PYTHONHASHSEED": value}).stdout.strip() for value in ("1", "987")]
    assert results[0] == results[1]
    assert int(results[0]) == experiment_seed("train-path", 0)
    assert int(results[0]) == 137779683704438885854556308124853216522
    assert experiment_seed("train-path", 0) != experiment_seed("train-path", 1)
    assert experiment_seed("train-path", 0) != experiment_seed("validation-path", 0)


@pytest.mark.parametrize("changes", [
    {"split": "test"}, {"split": "TRAIN"}, {"paths": 0}, {"paths": True},
    {"days": 0}, {"lags": -1}, {"master_seed": -1},
    {"start": datetime(2024, 1, 8)}, {"start": datetime(2024, 1, 6, tzinfo=timezone.utc)},
])
def test_invalid_generation_configuration(changes):
    with pytest.raises(InputError):
        generated(**changes)


@pytest.mark.parametrize("changes", [
    {"split": "test"}, {"lags": -1}, {"stock_ids": ("A", "A", "C")},
    {"features": np.zeros((2, 18))}, {"log_returns": np.zeros((2, 2))},
    {"horizon_days": np.array([1, float("nan")])}, {"horizon_days": np.array([1, 3])},
    {"seeds": ()}, {"path_ids": ("a", "b")},
    {"decision_times": (START, START)},
])
def test_invalid_direct_training_data_is_rejected(changes):
    with pytest.raises(InputError):
        replace(generated(paths=1, days=2), **changes)


@pytest.mark.parametrize("spots,time,horizon,lags", [
    ({"A": 0}, START, START+timedelta(days=1), 3),
    ({"A": float("nan")}, START, START+timedelta(days=1), 3),
    ({}, START, START+timedelta(days=1), 3),
    ({"A": 100}, START, START, 3),
    ({"A": 100}, START, START+timedelta(days=1), -1),
])
def test_invalid_conditioning_inputs(spots, time, horizon, lags):
    with pytest.raises(InputError):
        conditioning_features(("A",), [], spots, time, horizon, lags=lags)
