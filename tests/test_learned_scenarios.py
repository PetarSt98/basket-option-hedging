"""Learned market draws enter the same signed pricing/optimization boundary."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from basket_hedging.contracts import DataGap, InputError, Instrument, readonly
from basket_hedging.engine import PolicyView
from basket_hedging.learned_model import ConditionalGaussianModel
from basket_hedging.learned_scenarios import LearnedScenarios
from basket_hedging.market import Observation, ValuationPacket
from basket_hedging.scenario_hedge import ScenarioHedge
from basket_hedging.scenarios import ConditionalGBMScenarios
from basket_hedging.valuation import PricingParameters


START = datetime(2026, 1, 5, 16, tzinfo=timezone.utc)
STOCKS = ("S1", "S2")
VOLATILITIES = (.2, .3)
CORRELATION = ((1., .35), (.35, 1.))


def state(time=START, *, review_hours=24, execution_hours=17, expiry_hours=24):
    expiry = time + timedelta(hours=expiry_hours)
    registry = {key: Instrument(key, "stock") for key in STOCKS}
    registry.update(
        B=Instrument("B", "basket_put", multiplier=10, strike=145, expiry=expiry,
                     coefficients=(("S1", 1.), ("S2", .5))),
        P=Instrument("P", "put", multiplier=5, strike=105, expiry=expiry, underlying="S1"),
        C=Instrument("C", "call", multiplier=20, strike=75, expiry=expiry, underlying="S2"))
    marks = {"S1": 100., "S2": 80., "B": 8., "P": 3., "C": 7.}
    current = tuple(Observation(f"current-{key}", key, marks[key], time, time) for key in STOCKS)
    previous = tuple(Observation(f"previous-{key}", key, price, time-timedelta(days=1), time-timedelta(days=1))
                     for key, price in zip(STOCKS, (99., 82.)))
    packet = ValuationPacket(time, marks, {"unchanged": marks}, observed_at={key: time for key in marks})
    return PolicyView("decision", time, 0, readonly(registry), readonly({"B": -2}),
                      10000, 300, 50, 9650, packet, previous+current, (), (*STOCKS, "P", "C"), False,
                      readonly({"earliest_execution": time + timedelta(hours=execution_hours),
                                "next_rebalance_time": time + timedelta(hours=review_hours)}),
                      market=readonly({row.instrument_id: row for row in current}))


def model(*, sensitive_to_history=False):
    # Constant output exactly matches the declared GBM's log-return law; no fit
    # is involved in this pricing/integration fixture.
    sigma = np.array(VOLATILITIES)
    mu = -.5*sigma**2/(365*.01)
    factor = np.diag(sigma) @ np.linalg.cholesky(CORRELATION) / (np.sqrt(365)*.01)
    lower = factor[np.tril_indices(2)].copy()
    lower[[0, 2]] = np.log(np.expm1(lower[[0, 2]]-1e-4))
    first, second, final = np.zeros((2, 7)), np.zeros((2, 2)), np.zeros((5, 2))
    if sensitive_to_history:
        first[0, 2], second[0, 0], final[0, 0] = 40., 1., .2
    return ConditionalGaussianModel(STOCKS, 1, np.zeros(7), np.ones(7),
        (first, second, final), (np.zeros(2), np.zeros(2), np.concatenate((mu, lower))),
        START-timedelta(days=1), START-timedelta(days=2))


def pricing():
    return PricingParameters(STOCKS, VOLATILITIES, CORRELATION, .01,
                             START-timedelta(days=1), 71, samples=32)


def source(*, samples=2, seed=123, learned=None):
    return LearnedScenarios(learned or model(), pricing(), samples=samples, seed=seed)


def fixed_future_prices(monkeypatch):
    # Two joint scenarios, independent of the provider's implementation:
    # basket levels 125/155, put payoffs 20/0, constituent put 15/0, call 0/15.
    returns = np.log(np.array([[90., 70.], [110., 90.]]) / np.array([100., 80.]))

    def sample(self, features, horizon_days, normals):
        assert normals.shape == (2, 2)
        assert horizon_days == 1.
        return returns.copy()

    monkeypatch.setattr(ConditionalGaussianModel, "sample_log_returns", sample)


@pytest.mark.parametrize("quantity,expected_loss", [(-2, [240., -160.]), (2, [-240., 160.])])
def test_expiry_repricing_has_independent_signs_multipliers_and_output_dimensions(monkeypatch, quantity, expected_loss):
    fixed_future_prices(monkeypatch)
    current = state()
    generated = source().generate(current, {"B": quantity}, ("S1", "S2", "P", "C"))
    assert generated.delta_v.shape == (2,) and generated.delta_h.shape == (2, 4)
    np.testing.assert_allclose(generated.delta_v, expected_loss, rtol=0, atol=1e-10)
    np.testing.assert_allclose(generated.delta_h, [[-10., -10., 60., -140.], [10., 10., -15., 160.]],
                               rtol=0, atol=1e-10)
    assert generated.horizon == START+timedelta(days=1)
    assert current.positions == {"B": -2}
    assert (current.cash, current.posted_margin_cash, current.restricted_short_proceeds,
            current.available_cash) == (10000, 300, 50, 9650)
    assert current.pending_orders == () and current.valuation.marks["B"] == 8.
    assert generated.input_ids == ("current-S1", "current-S2", "previous-S1", "previous-S2")


def test_relevant_expiry_shortens_model_time_and_uses_exact_payoff(monkeypatch):
    fixed_future_prices(monkeypatch)
    current = state(review_hours=72, execution_hours=17, expiry_hours=24)
    generated = source().generate(current, {"B": -2}, ("S1", "P", "C"))
    assert generated.horizon == START+timedelta(days=1)
    np.testing.assert_allclose(generated.delta_v, [240., -160.], rtol=0, atol=1e-10)


def test_existing_policy_passes_exact_repriced_arrays_and_actual_holdings_to_optimizer(monkeypatch):
    fixed_future_prices(monkeypatch)
    current = replace(state(), positions=readonly({"B": -2, "S1": -10, "S2": 3, "P": 2, "C": 1}))
    captured = {}

    def inspect(delta_v, delta_h, previous, costs, **settings):
        captured["called"] = True
        # S2 is fixed: its gain/loss adds -30/+30 to the loss response.
        np.testing.assert_allclose(delta_v, [270., -190.], rtol=0, atol=1e-10)
        np.testing.assert_allclose(delta_h, [[-10., 60., -140.], [10., -15., 160.]], rtol=0, atol=1e-10)
        assert previous == [-10, 2, 1] and costs == [.02, .1, .2]
        assert settings["alpha"] == .1 and settings["gross_scale"] == 400
        return SimpleNamespace(holdings=np.array([4., 5., 6.]), intercept=1234., mse=0., penalty=0.,
                               objective=0., iterations=1, kkt_residual=0.,
                               polish_attempted=False, polish_succeeded=False)

    monkeypatch.setattr("basket_hedging.scenario_hedge.solve_hedge", inspect)
    strategy = ScenarioHedge(source(), ("S1", "P", "C"), {"S1": .02, "P": .1, "C": .2}, alpha=.1)
    assert strategy(current) == {"S1": 4., "P": 5., "C": 6.}  # Absolute targets, not trades.
    assert captured == {"called": True}
    assert current.positions == {"B": -2, "S1": -10, "S2": 3, "P": 2, "C": 1}
    assert current.cash == 10000  # The regression intercept never becomes cash.


def test_future_history_mutation_cannot_change_features_scenarios_or_provenance():
    current = state()
    generator = source(samples=4, learned=model(sensitive_to_history=True))
    future = tuple(Observation(f"future-{key}", key, 1e6, START+timedelta(hours=1), START+timedelta(hours=1))
                   for key in STOCKS)
    future += tuple(Observation(f"unreleased-past-{key}", key, 1e6,
                                START-timedelta(hours=2), START+timedelta(hours=1)) for key in STOCKS)
    first = generator.generate(current, {"B": -2}, STOCKS)
    second = generator.generate(replace(current, history=current.history+future), {"B": -2}, STOCKS)
    np.testing.assert_array_equal(first.delta_v, second.delta_v)
    np.testing.assert_array_equal(first.delta_h, second.delta_h)
    assert first.input_ids == second.input_ids
    changed_past = tuple(replace(row, value=90.) if row.id == "previous-S1" else row for row in current.history)
    third = generator.generate(replace(current, history=changed_past), {"B": -2}, STOCKS)
    assert not np.array_equal(first.delta_h, third.delta_h)  # The fixture really uses released lag features.


@pytest.mark.parametrize("time,review_hours,execution_hours", [
    (START, 24, 17), (START+timedelta(days=4), 72, 65),
])
def test_provider_uses_actual_weekday_or_weekend_horizon_and_same_base_normals_as_true_provider(
        monkeypatch, time, review_hours, execution_hours):
    current = state(time, review_hours=review_hours, execution_hours=execution_hours, expiry_hours=720)
    captured = {}
    original = ConditionalGaussianModel.sample_log_returns

    def record(self, features, horizon_days, normals):
        captured.update(days=horizon_days, horizon_feature=features[-1], normals=normals.copy())
        return original(self, features, horizon_days, normals)

    monkeypatch.setattr(ConditionalGaussianModel, "sample_log_returns", record)
    learned = source(samples=8).generate(current, {"B": -2}, STOCKS)
    true = ConditionalGBMScenarios(pricing(), replay_volatilities=VOLATILITIES,
                                  replay_correlation=CORRELATION, samples=8, seed=123).generate(current, {"B": -2}, STOCKS)
    assert captured["days"] == review_hours/24
    assert captured["horizon_feature"] == pytest.approx(np.log(review_hours/24))
    assert learned.horizon == time+timedelta(hours=review_hours)
    # Recover independent normals from the true provider's stock prices using
    # the GBM equation, not its seed helper or the learned model's sampler.
    years = review_hours/(365*24)
    sigma = np.array(VOLATILITIES)
    true_returns = np.log((true.delta_h+np.array([100., 80.]))/np.array([100., 80.]))
    shocks = (true_returns+.5*sigma**2*years)/(sigma*np.sqrt(years))
    normals = np.linalg.solve(np.linalg.cholesky(CORRELATION), shocks.T).T
    np.testing.assert_allclose(captured["normals"], normals, rtol=0, atol=3e-14)
    np.testing.assert_allclose(learned.delta_h, true.delta_h, rtol=0, atol=5e-14)
    np.testing.assert_allclose(learned.delta_v, true.delta_v, rtol=0, atol=5e-12)


def test_repeated_calls_and_larger_samples_preserve_scenario_prefix():
    current = state()
    first = source(samples=4).generate(current, {"B": -2}, STOCKS)
    repeated = source(samples=4).generate(current, {"B": -2}, STOCKS)
    larger = source(samples=7).generate(current, {"B": -2}, STOCKS)
    for name in ("delta_v", "delta_h"):
        np.testing.assert_array_equal(getattr(first, name), getattr(repeated, name))
        np.testing.assert_array_equal(getattr(first, name), getattr(larger, name)[:4])
    changed = source(samples=4, seed=124).generate(current, {"B": -2}, STOCKS)
    assert not np.array_equal(first.delta_h, changed.delta_h)


def test_unavailable_model_and_nonexecutable_horizon_are_rejected():
    unavailable = replace(model(), available_at=START+timedelta(seconds=1))
    with pytest.raises(InputError, match="not available"):
        source(learned=unavailable).generate(state(), {"B": -2}, STOCKS)
    with pytest.raises(DataGap, match="no interval"):
        source().generate(state(review_hours=16, execution_hours=17), {"B": -2}, STOCKS)
    with pytest.raises(InputError, match="ordered constituents"):
        LearnedScenarios(replace(model(), stock_ids=STOCKS[::-1]), pricing())
