"""Small artificial arrays verify checkpoint selection plumbing and chronology."""

from datetime import datetime, timezone
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from basket_hedging.contracts import InputError
from basket_hedging.volgan import (
    VolGANConfig, VolGANDiscriminator, VolGANGenerator, VolGANModel,
    fit_volgan, validate_prepared_volgan,
)
from basket_hedging.surfaces import SurfaceGrid


GRID = SurfaceGrid((.8, 1., 1.2), (.05, .1))
TIME = datetime(2017, 12, 29, 21, tzinfo=timezone.utc)
ASSETS = ("SPY", "QQQ", "IWM")


def training_arguments():
    random = np.random.default_rng(82)
    features = random.normal(0, .01, (8, 3, GRID.size+3))
    features[:, :, 2] = .2
    features[:, :, 3:] = np.log(.2)
    targets = random.normal(0, .01, (8, 3*(GRID.size+1)))
    return dict(conditions=features.reshape(8, -1), targets=targets, asset_ids=ASSETS,
                grid=GRID, available_at=TIME, trained_through=TIME,
                config=VolGANConfig(calibration_epochs=1, epochs=3, batch_size=4, seed=712))


def test_scheduled_snapshots_cannot_change_training_weights_or_rng():
    before_rng = torch.random.get_rng_state().clone()
    original = fit_volgan(**training_arguments())
    snapshots, weights, epochs = [], [], []

    def observe(model):
        snapshots.append(model)
        epochs.append(model.metadata["selected_epoch"])
        weights.append(tuple(p.detach().clone() for p in model.generator.parameters()))
        assert model.metadata["training_rows"] == 8
        assert model.metadata["smoothness_m"] > 0
        assert model.metadata["smoothness_t"] > 0
        # An observer's RNG, mode or weight mutation must remain isolated.
        torch.manual_seed(991)
        torch.randn(37)
        model.generator.train()
        model.discriminator.train()
        with torch.no_grad():
            for parameter in model.generator.parameters():
                parameter.zero_()

    observed = fit_volgan(**training_arguments(), checkpoint_epochs=(1, 3), checkpoint=observe)
    assert epochs == [1, 3]
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert snapshots[0].generator is not snapshots[1].generator
    assert snapshots[1].generator is not observed.generator
    assert any(not torch.equal(a, b) for a, b in zip(*weights))
    for name in ("generator", "discriminator"):
        left, right = getattr(original, name), getattr(observed, name)
        for key, expected in left.state_dict().items():
            assert torch.equal(expected, right.state_dict()[key])
    assert original.metadata["final_generator_sha256"] == observed.metadata["final_generator_sha256"]
    assert original.training_history == observed.training_history


@pytest.mark.parametrize("schedule,callback", [
    ((1, 1), True), ((3, 1), True), ((0,), True), ((4,), True),
    ((1.5,), True), ((True,), True), ((1,), False),
])
def test_invalid_checkpoint_schedules_rejected_before_training(schedule, callback):
    with pytest.raises(InputError, match="checkpoint"):
        fit_volgan(**training_arguments(), checkpoint_epochs=schedule,
                   checkpoint=(lambda model: None) if callback else None)


def evaluation_fixture():
    # Separate date segments deliberately mark their gaps as nonconsecutive.
    segments = [np.arange("2017-12-01", "2018-02-10", dtype="datetime64[D]"),
                np.arange("2018-12-01", "2019-02-10", dtype="datetime64[D]"),
                np.arange("2020-01-01", "2020-01-04", dtype="datetime64[D]")]
    days = np.concatenate(segments).astype(str)
    row = np.arange(len(days))[:, None]
    prices = np.array([100., 150., 70.])*np.exp(.002*np.sin(.3*row+np.arange(3)))
    iv = (.2 + .01*np.sin(.2*row[:, :, None, None]+np.arange(3)[None, :, None, None])
          + np.array([.02, .01, 0.])[None, None, :, None]
          + np.array([0., .01])[None, None, None, :])
    consecutive = np.ones(len(days), dtype=bool)
    consecutive[np.r_[0, np.cumsum([len(v) for v in segments])[:-1]]] = False
    data = dict(dates=days, symbols=np.array(ASSETS), close=prices, iv=iv,
                consecutive=consecutive, valid_surface=np.ones((len(days), 3), bool))
    config = VolGANConfig(calibration_epochs=1, epochs=1, seed=315)
    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        generator = VolGANGenerator(3*(GRID.size+3), 3*(GRID.size+1), config)
        discriminator = VolGANDiscriminator(3*(GRID.size+3), 3*(GRID.size+1), config)
    model = VolGANModel(ASSETS, GRID, config, generator, discriminator, TIME, TIME)
    return model, data


def test_development_evaluation_ignores_even_nonfinite_future_prices_and_surfaces():
    model, data = evaluation_fixture()
    expected = validate_prepared_volgan(model, data, year=2018, samples=8, seed=316)
    changed = {key: value.copy() for key, value in data.items()}
    future = changed["dates"] >= "2019-01-01"
    changed["close"][future] = np.nan
    changed["iv"][future] = np.inf
    actual = validate_prepared_volgan(model, changed, year=2018, samples=8, seed=316)
    assert actual == expected
    assert expected["start"].startswith("2018-")
    assert expected["end"].startswith("2018-")


@pytest.mark.parametrize("year", [2018, 2019])
def test_evaluation_calls_sampler_only_for_selected_year(year):
    model, data = evaluation_fixture()
    sampled_indices = []

    def sample(feature, target, normals):
        sampled_indices.append(target)
        return model.sample(feature, data["close"][target-1], normals)

    result = validate_prepared_volgan(model, data, year=year, samples=8, seed=317,
                                     sampler=sample)
    assert len(sampled_indices) == result["rows"] > 0
    assert all(str(data["dates"][index]).startswith(str(year)) for index in sampled_indices)
    assert result["start"] == data["dates"][sampled_indices[0]]
    assert result["end"] == data["dates"][sampled_indices[-1]]


def acceptance_function():
    path = Path(__file__).resolve().parents[1]/"examples/calibrate_volgan.py"
    spec = importlib.util.spec_from_file_location("calibration_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.acceptance


@pytest.mark.parametrize("deterioration,failed_gate", [
    (None, None), ("spread", "spread_interval_score"),
    ("wide_intervals", "return_interval_score_90"),
    ("surface", "log_iv_mean_forecast_rmse"),
])
def test_joint_and_surface_safeguards_are_required_despite_improved_energy(deterioration, failed_gate):
    original = dict(energy_score=.01, return_5_95_percentile_coverage=[.5, .6, .65],
                    return_interval_score_90=[.02, .025, .024],
                    return_mean_forecast_rmse=[.01, .012, .011],
                    log_iv_mean_forecast_rmse=[.1, .12, .11],
                    spread_qqq_minus_iwm={"interval_score_90": .01})
    candidate = deepcopy(original)
    candidate.update(energy_score=.009, return_5_95_percentile_coverage=[.7, .75, .8],
                     spread_qqq_minus_iwm={"interval_score_90": .008})
    if deterioration == "spread":
        candidate["spread_qqq_minus_iwm"]["interval_score_90"] = .012
    elif deterioration == "wide_intervals":
        candidate["return_5_95_percentile_coverage"] = [.9]*3
        candidate["return_interval_score_90"][1] = .04
    elif deterioration == "surface":
        candidate["log_iv_mean_forecast_rmse"][2] = .2
    gates = dict(energy_score_ratio=.98, coverage_error_improvement=.05,
                 max_asset_coverage_deterioration=.03, max_diagnostic_ratio=1.1,
                 spread_interval_score_ratio=.95)
    result = acceptance_function()(candidate, original, gates)
    assert result["energy_score"] and result["mean_coverage_error"]
    if failed_gate is None:
        assert all(result.values())
    else:
        assert not result[failed_gate]
        assert not all(result.values())
