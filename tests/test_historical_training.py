"""Small manufactured prices validate chronology; no historical performance claims."""

from datetime import date, datetime, timezone

import numpy as np
import pytest

from basket_hedging.contracts import InputError
from basket_hedging.historical_training import (
    close_time, gaussian_history_data, historical_training_indices,
)
from basket_hedging.learned_model import ConditionalGaussianModel, fit_generator


def arrays():
    days = np.array(["2017-12-22", "2017-12-26", "2017-12-27", "2017-12-28", "2017-12-29",
                     "2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05",
                     "2020-01-02", "2020-01-03"])
    return dict(dates=days, symbols=np.array(["SPY", "QQQ", "IWM"]),
                close=np.array([[100 + row, 50 + 2 * row, 200 - row] for row in range(len(days))], float),
                consecutive=np.array([False] + [True] * 8 + [False, True]),
                valid_surface=np.ones((len(days), 3), bool))


def test_joint_rows_chronological_targets_and_independent_expected_returns():
    data = arrays()
    train = gaussian_history_data(data, split="train", lags=2, warmup=2)
    validation = gaussian_history_data(data, split="validation", lags=2, warmup=2)
    assert len(train.features) == 2 and len(validation.features) == 3
    assert train.stock_ids == ("SPY", "QQQ", "IWM")
    np.testing.assert_allclose(train.log_returns[0], np.log([103 / 102, 56 / 54, 197 / 198]))
    np.testing.assert_allclose(train.features[0, :3], np.log([102, 54, 198]))
    # Latest lag features use yesterday->today, not today's forward label.
    np.testing.assert_allclose(train.features[0, 3:6], np.log([102 / 101, 54 / 52, 198 / 199]))
    assert max(train.target_times) < min(validation.decision_times)
    assert max(validation.target_times).date() == date(2018, 1, 5)
    assert train.seeds == validation.seeds == ()
    assert "historical" in train.manifest()["data_kind"]
    assert train.manifest()["simulator_seeds"] is None


def test_mutating_final_test_prices_does_not_change_training_or_validation():
    original, future = arrays(), arrays()
    future["close"][-2:] *= 1000
    for split in ("train", "validation"):
        before = gaussian_history_data(original, split=split, lags=2, warmup=2)
        after = gaussian_history_data(future, split=split, lags=2, warmup=2)
        np.testing.assert_array_equal(before.features, after.features)
        np.testing.assert_array_equal(before.log_returns, after.log_returns)
        assert before.manifest()["dataset_sha256"] == after.manifest()["dataset_sha256"]


@pytest.mark.parametrize("missing", ["session", "surface", "price"])
def test_missing_joint_information_is_not_compressed_into_a_next_day_example(missing):
    data = arrays()
    if missing == "session":
        data["consecutive"][3] = False
    elif missing == "surface":
        data["valid_surface"][3, 1] = False
    else:
        data["close"][3, 2] = np.nan
    with pytest.raises(InputError, match="no valid historical train"):
        historical_training_indices(data, split="train", warmup=2)


def test_regular_close_clock_tracks_dst_without_rescaling_a_daily_gan_transition():
    assert close_time("2026-01-09").hour == 21
    assert close_time("2026-07-09").hour == 20
    assert (close_time("2026-03-09") - close_time("2026-03-06")).total_seconds() / 3600 == 71


def test_original_gaussian_fitter_accepts_real_history_adapter_and_roundtrips(tmp_path):
    pytest.importorskip("torch")
    train = gaussian_history_data(arrays(), split="train", lags=2, warmup=2)
    validation = gaussian_history_data(arrays(), split="validation", lags=2, warmup=2)
    model = fit_generator(train, validation, epochs=2, hidden=4, batch_size=2, seed=17,
                          available_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
    assert model.metadata["train"]["simulator_seeds"] is None
    assert model.metadata["validation"]["rows"] == 3
    assert len(model.training_history) == 2
    np.testing.assert_allclose(model.feature_mean, train.features.mean(axis=0))
    destination = model.save(tmp_path / "historical_gaussian.json")
    loaded = ConditionalGaussianModel.load(destination)
    draws = np.eye(3)
    np.testing.assert_array_equal(model.sample_log_returns(train.features[0], 1, draws),
                                  loaded.sample_log_returns(train.features[0], 1, draws))
