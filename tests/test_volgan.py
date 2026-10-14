"""Tiny artificial arrays test model mechanics, never historical performance."""

from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from basket_hedging.contracts import InputError
from basket_hedging.surfaces import SurfaceGrid, flatten_surface, unflatten_surface
from basket_hedging.volgan import (
    RETURN_SCALE, VolGANConfig, VolGANDiscriminator, VolGANGenerator, VolGANModel,
    build_training_pairs, conditioning_features, fit_volgan, prepared_training_pairs, reconstruct_state,
    smoothness_penalties, transition_target,
)


GRID = SurfaceGrid((.8, 1., 1.2), (.05, .1))
TIME = datetime(2017, 12, 29, 21, tzinfo=timezone.utc)


def joint_fixture():
    times = tuple(TIME - timedelta(days=30-i) for i in range(30))
    steps = np.arange(30)[:, None]
    # Different, jointly aligned returns and surfaces: no independent asset shuffle.
    spots = np.array([100., 200., 80.])*np.exp(.003*np.sin(steps*.3+np.arange(3)))
    iv = (.2 + .01*np.sin(steps[:, :, None, None]+np.arange(3)[None, :, None, None])
          + np.array([.02, .01, .0])[None, None, :, None] + np.array([0., .01])[None, None, None, :])
    return spots, iv, times


@pytest.fixture(scope="module")
def trained_model():
    spots, iv, times = joint_fixture()
    x, y, _, ends = build_training_pairs(spots, iv, times, GRID, cutoff=TIME)
    return fit_volgan(x, y, asset_ids=("SPY", "QQQ", "IWM"), grid=GRID,
                      available_at=TIME, trained_through=ends[-1],
                      config=VolGANConfig(calibration_epochs=1, epochs=2, batch_size=3, seed=789),
                      metadata={"dataset": "tiny artificial unit fixture"})


def test_surface_order_and_transform_round_trip():
    surfaces = np.array([[[.11, .21], [.12, .22], [.13, .23]],
                         [[.31, .41], [.32, .42], [.33, .43]]])
    assert flatten_surface(surfaces, GRID)[0].tolist() == [.11, .12, .13, .21, .22, .23]
    np.testing.assert_array_equal(unflatten_surface(flatten_surface(surfaces, GRID), GRID), surfaces)
    history = np.zeros((21, 2))
    history[-2:] = [[-.02, .01], [.01, -.03]]
    condition = conditioning_features(history, surfaces, GRID)
    assert condition[0] == pytest.approx(np.sqrt(252)*.01)
    assert condition[1] == pytest.approx(-np.sqrt(252)*.02)
    assert condition[2] == pytest.approx(np.sqrt(12*(.01**2+.02**2)))
    next_spots = np.array([101., 195.])
    next_iv = surfaces*np.array([1.1, .9])[:, None, None]
    target = transition_target(np.array([100., 200.]), next_spots, surfaces, next_iv, GRID)
    spots_back, iv_back = reconstruct_state(condition, np.array([100., 200.]), target[None, :], GRID)
    np.testing.assert_allclose(spots_back[0], next_spots, rtol=1e-14)
    np.testing.assert_allclose(iv_back[0], next_iv, rtol=1e-14)


def test_training_cutoff_and_joint_rows_use_no_future_prices():
    spots, iv, times = joint_fixture()
    cutoff = times[25]
    x, y, decisions, targets = build_training_pairs(spots, iv, times, GRID, cutoff=cutoff)
    assert len(x) == 4
    assert decisions[0] == times[21] and targets[-1] == cutoff
    np.testing.assert_allclose(y[0].reshape(3, 7)[:, 0], np.sqrt(252)*np.log(spots[22]/spots[21]))
    spots[26:], iv[26:] = np.nan, np.nan
    changed = build_training_pairs(spots, iv, times, GRID, cutoff=cutoff)
    np.testing.assert_array_equal(x, changed[0])
    np.testing.assert_array_equal(y, changed[1])
    with pytest.raises(InputError, match="chronological"):
        build_training_pairs(spots, iv, times[::-1], GRID, cutoff=cutoff)


def test_paper_network_architecture_is_one_joint_adversarial_pair():
    config = VolGANConfig()
    generator = VolGANGenerator(306, 300, config)
    discriminator = VolGANDiscriminator(306, 300, config)
    assert [(layer.in_features, layer.out_features) for layer in generator.layers if isinstance(layer, torch.nn.Linear)] == [(338, 16), (16, 32), (32, 300)]
    assert [(layer.in_features, layer.out_features) for layer in discriminator.layers if isinstance(layer, torch.nn.Linear)] == [(606, 16), (16, 1)]
    assert isinstance(generator.layers[1], torch.nn.Softplus)
    assert isinstance(generator.layers[3], torch.nn.Softplus)
    assert isinstance(discriminator.layers[-1], torch.nn.Sigmoid)
    assert config.calibration_epochs == 25 and config.epochs == 10_000


def test_smoothness_uses_correct_axes_and_log_surface_with_independent_values():
    # log sigma = 2*m + 3*t: every m slope is 2 and every t slope is 3.
    grid = SurfaceGrid((.8, 1., 1.4), (.1, .3))
    g = 2*np.array(grid.moneyness)[:, None] + 3*np.array(grid.maturities)[None, :]
    values = torch.tensor(np.tile(g, (2, 3, 1, 1)), dtype=torch.float64, requires_grad=True)
    m, t = smoothness_penalties(values, grid)
    assert m.item() == pytest.approx(3*2*2*4)
    assert t.item() == pytest.approx(3*3*1*9)
    (m+t).backward()
    assert torch.isfinite(values.grad).all()


def test_both_networks_train_and_save_load_preserves_joint_samples(trained_model, tmp_path):
    model = trained_model
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model.config.seed)
        initial_g = VolGANGenerator(27, 21, model.config)
        initial_d = VolGANDiscriminator(27, 21, model.config)
    assert any(not torch.equal(a, b) for a, b in zip(initial_g.parameters(), model.generator.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(initial_d.parameters(), model.discriminator.parameters()))
    assert model.metadata["initial_generator_sha256"] != model.metadata["final_generator_sha256"]
    assert model.metadata["initial_discriminator_sha256"] != model.metadata["final_discriminator_sha256"]
    assert model.metadata["smoothness_m"] > 0 and model.metadata["smoothness_t"] > 0
    assert [row["phase"] for row in model.training_history] == ["calibration", "training", "training"]
    spots, iv, _ = joint_fixture()
    condition = conditioning_features(np.diff(np.log(spots), axis=0), iv[-1], GRID)
    noise = np.random.default_rng(871).standard_normal((5, 32))
    expected = model.sample(condition, spots[-1], noise)
    assert expected[0].shape == (5, 3) and expected[1].shape == (5, 3, 3, 2)
    assert np.all(expected[0] > 0) and np.all(expected[1] > 0)
    restored = VolGANModel.load(model.save(tmp_path/"volgan.json"))
    actual = restored.sample(condition, spots[-1], noise)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert restored.asset_ids == ("SPY", "QQQ", "IWM")


def test_training_is_reproducible_and_preserves_global_random_state(trained_model):
    spots, iv, times = joint_fixture()
    x, y, _, ends = build_training_pairs(spots, iv, times, GRID, cutoff=TIME)
    before = torch.random.get_rng_state().clone()
    model = fit_volgan(x, y, asset_ids=trained_model.asset_ids, grid=GRID,
                       available_at=TIME, trained_through=ends[-1], config=trained_model.config)
    assert torch.equal(before, torch.random.get_rng_state())
    for key, value in trained_model.generator.state_dict().items():
        assert torch.equal(value, model.generator.state_dict()[key])
    for key, value in trained_model.discriminator.state_dict().items():
        assert torch.equal(value, model.discriminator.state_dict()[key])


def test_training_restarts_networks_after_calibration(monkeypatch):
    starts = []
    real_optimizer = torch.optim.RMSprop

    def record_initial_parameters(parameters, **kwargs):
        parameters = tuple(parameters)
        starts.append(tuple(parameter.detach().clone() for parameter in parameters))
        return real_optimizer(parameters, **kwargs)

    monkeypatch.setattr(torch.optim, "RMSprop", record_initial_parameters)
    spots, iv, times = joint_fixture()
    x, y, _, ends = build_training_pairs(spots, iv, times, GRID, cutoff=TIME)
    fit_volgan(x, y, asset_ids=("SPY", "QQQ", "IWM"), grid=GRID,
               available_at=TIME, trained_through=ends[-1],
               config=VolGANConfig(calibration_epochs=1, epochs=1, batch_size=3))
    assert len(starts) == 4
    for first, restarted in ((starts[0], starts[2]), (starts[1], starts[3])):
        assert all(torch.equal(a, b) for a, b in zip(first, restarted))


def test_volgan_never_accepts_a_gaussian_checkpoint(tmp_path):
    path = tmp_path/"model.json"
    path.write_text(json.dumps({"model_version": "conditional-neural-gaussian-v1"}), encoding="utf-8")
    with pytest.raises(InputError, match="no Gaussian fallback"):
        VolGANModel.load(path)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_bad_surface_is_rejected(value):
    iv = np.full((3, *GRID.shape), .2)
    iv[0, 0, 0] = value
    with pytest.raises(InputError):
        conditioning_features(np.zeros((21, 3)), iv, GRID)


def test_unscaled_one_session_output_even_across_weekend():
    iv = np.full((1, *GRID.shape), .2)
    condition = conditioning_features(np.zeros((21, 1)), iv, GRID)
    draws = np.zeros((1, GRID.size+1))
    draws[0, 0] = RETURN_SCALE*np.log(1.02)
    result, _ = reconstruct_state(condition, np.array([100.]), draws, GRID)
    assert result.item() == pytest.approx(102.)  # No sqrt(3) multiplication for Friday-Monday.


def test_invalid_training_cutoff_rejected():
    with pytest.raises(InputError, match="postdates"):
        fit_volgan(np.ones((1, 27)), np.ones((1, 21)), asset_ids=("SPY", "QQQ", "IWM"),
                   grid=GRID, available_at=TIME, trained_through=TIME+timedelta(days=1))


def test_prepared_windows_do_not_bridge_missing_session_or_surface():
    spots, iv, times = joint_fixture()
    data = dict(close=spots, iv=iv, dates=np.array([t.date().isoformat() for t in times]),
                symbols=np.array(["SPY", "QQQ", "IWM"]), consecutive=np.r_[False, np.ones(29, dtype=bool)],
                valid_surface=np.ones((30, 3), dtype=bool))
    x, y, indices = prepared_training_pairs(data, GRID, split="train")
    assert indices == tuple(range(22, 30))
    np.testing.assert_allclose(y[0].reshape(3, 7)[:, 0], np.sqrt(252)*np.log(spots[22]/spots[21]))
    data["valid_surface"][1, 2] = False
    _, _, indices = prepared_training_pairs(data, GRID, split="train")
    assert indices == tuple(range(24, 30))
    data["consecutive"][2] = False
    _, _, indices = prepared_training_pairs(data, GRID, split="train")
    assert indices == tuple(range(24, 30))


def test_joint_network_allows_cross_asset_response_to_one_condition():
    generator = VolGANGenerator(27, 21, VolGANConfig())
    with torch.no_grad():
        for parameter in generator.parameters():
            parameter.fill_(.01)
    condition = torch.zeros((1, 27), requires_grad=True)
    result = generator(condition, torch.zeros((1, 32)))
    # The third asset's return can depend on the first asset's observed return.
    result[0, 14].backward()
    assert condition.grad[0, 0] > 0
