"""Tiny artificial unit fixtures only; these are not historical benchmark data."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import log

import numpy as np
import pytest

from basket_hedging.contracts import DataGap, Funding, InputError, Instrument, Opening
from basket_hedging.engine import Engine, PolicyView
from basket_hedging.historical_hedge import (
    PAPER_ALPHAS, PaperAICHedge, SurfaceBasketValuator,
    reprice_basket_scenarios, select_paper_alpha,
)
from basket_hedging.market import Borrow, Event, Fixing, Observation, Quote, Rules
from basket_hedging.policies import ScheduledDeltaPolicy
from basket_hedging.reporting import write_reports
from basket_hedging.risk import UnnettedLiabilityStressMargin
from basket_hedging.scenarios import ScenarioChanges
from basket_hedging.surfaces import IVSurface, SurfaceGrid


START = datetime(2026, 1, 9, 16, tzinfo=timezone.utc)  # Friday.
HORIZON = START + timedelta(days=3)
EXPIRY = START + timedelta(days=30)
GRID = SurfaceGrid((0.8, 1.2), (1 / 365, 1.0))
STOCKS = ("SPY", "QQQ", "IWM")


def surfaces(time=START, sigma=0.2):
    return {key: IVSurface(GRID, np.full((2, 2), sigma), time, time, "tiny unit fixture")
            for key in STOCKS}


def setup(*, expiry=EXPIRY, quantity=-2, multiplier=10):
    instruments = {key: Instrument(key, "stock") for key in STOCKS}
    # Initial equal values are 100 each, fixed shares are 1, 2, 0.5.
    instruments["B"] = Instrument("B", "basket_put", multiplier=multiplier, strike=300,
                                   expiry=expiry, coefficients=tuple(zip(STOCKS, (1, 2, 0.5))))
    spots = dict(zip(STOCKS, (100.0, 50.0, 200.0)))
    pricing = SurfaceBasketValuator(STOCKS, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                            (0.0, 0.0, 1.0)), START, samples=512)
    packet = pricing.packet(START, spots, instruments, surfaces())
    observations = tuple(Observation(key, key, value, START, START) for key, value in spots.items())
    view = PolicyView("decision", START, 0, instruments, {"B": quantity}, 10000,
                      0, 0, 10000, packet, observations, (), STOCKS, False,
                      {"earliest_execution": HORIZON - timedelta(hours=7),
                       "next_rebalance_time": HORIZON}, market={o.instrument_id: o for o in observations})
    return pricing, view


def changes(view, pricing, spots, iv=None, horizon=HORIZON):
    count = len(spots)
    return reprice_basket_scenarios(
        view, view.positions, STOCKS, future_spots=np.asarray(spots),
        future_surfaces=np.full((count, 3, 2, 2), 0.2) if iv is None else iv,
        grid=GRID, horizon=horizon, valuator=pricing, seed=123,
        input_ids=("released-close", "released-surfaces"))


@pytest.mark.parametrize("quantity", [-2, 2])
def test_exact_expiry_signed_book_contract_multiplier_and_fixed_coefficients(quantity):
    pricing, view = setup(expiry=HORIZON, quantity=quantity, multiplier=10)
    packet = replace(view.valuation, marks={"SPY": 100.0, "QQQ": 50.0, "IWM": 200.0, "B": 8.0})
    view = replace(view, valuation=packet)
    result = changes(view, pricing, [[90, 40, 180], [110, 60, 220]])
    # Fixed shares 1,2,.5 give basket levels 260,340 -> exact put payoff 40,0.
    np.testing.assert_array_equal(result.delta_v, [-quantity * 10 * 32, quantity * 10 * 8])
    np.testing.assert_array_equal(result.delta_h, [[-10, -10, -20], [10, 10, 20]])
    assert view.instruments["B"].coefficients == (("SPY", 1), ("QQQ", 2), ("IWM", 0.5))
    assert view.cash == 10000 and view.positions == {"B": quantity}


def test_generated_surface_changes_affect_option_loss_with_identical_spots():
    pricing, view = setup()
    iv = np.full((2, 3, 2, 2), 0.1)
    iv[1] = 0.5
    result = changes(view, pricing, [[100, 50, 200]] * 2, iv)
    assert result.delta_v[1] > result.delta_v[0] + 10
    np.testing.assert_array_equal(result.delta_h, np.zeros((2, 3)))
    assert np.var(result.delta_v) > 0  # Volatility risk enters the optimizer response.
    assert result.input_ids == ("released-close", "released-surfaces")


def test_actual_and_generated_marks_share_pricer_and_frozen_random_draws():
    pricing, view = setup()
    result = changes(view, pricing, [[98, 51, 202]] * 2)
    actual = pricing.packet(HORIZON, dict(zip(STOCKS, (98.0, 51.0, 202.0))),
                             view.instruments, surfaces(HORIZON))
    np.testing.assert_array_equal(result.delta_v,
                                  [20 * (actual.marks["B"] - view.valuation.marks["B"])] * 2)
    assert "zero pricing rates and dividend yields" in actual.metadata["historical_basket_proxy"]
    # Same scenario on repeated calls is identical without sharing any mutable RNG.
    np.testing.assert_array_equal(result.delta_v, changes(view, pricing, [[98, 51, 202]] * 2).delta_v)


def test_remaining_maturity_changes_selected_surface_volatility_without_moving_contract():
    pricing, view = setup()
    sloped = {key: IVSurface(GRID, np.array([[0.1, 0.5], [0.1, 0.5]]), START, START, "unit")
              for key in STOCKS}
    current = pricing._volatilities(START, view.instruments["B"], sloped)
    later = pricing._volatilities(HORIZON, view.instruments["B"], sloped)
    # Linear interpolation at T=30/365 then T=27/365 on grid 1/365,1.
    assert current == pytest.approx((0.1 + 0.4 * 29 / 364,) * 3)
    assert later == pytest.approx((0.1 + 0.4 * 26 / 364,) * 3)
    assert view.instruments["B"].strike == 300 and view.instruments["B"].expiry == EXPIRY


def test_previous_close_surface_is_visible_but_future_surface_rejected():
    pricing, view = setup()
    opening = HORIZON - timedelta(hours=7)
    packet = pricing.packet(opening, {key: view.valuation.marks[key] for key in STOCKS},
                             view.instruments, surfaces())
    assert set(packet.metadata["surface_observed_at"].values()) == {START.isoformat()}
    with pytest.raises(InputError, match="unreleased"):
        pricing.packet(START, {key: view.valuation.marks[key] for key in STOCKS},
                        view.instruments, surfaces(HORIZON))


def test_official_expiry_payoff_needs_no_new_surface_or_monte_carlo():
    pricing, view = setup(expiry=HORIZON)
    spot = dict(zip(STOCKS, (90.0, 40.0, 180.0)))
    result = pricing.packet(HORIZON, spot, view.instruments, {}, fixings={"B": spot})
    assert result.marks["B"] == 40.0
    with pytest.raises(DataGap, match="official fixing"):
        pricing.packet(HORIZON, spot, view.instruments, {})


def test_live_surface_delta_and_settlement_reports_use_serializable_native_scalars(tmp_path):
    pricing, view = setup(expiry=HORIZON)
    assert all(type(delta) is float for delta in view.valuation.deltas["B"].values())
    first_fill, final_fill = START + timedelta(hours=1), HORIZON + timedelta(hours=1)
    start_spots = dict(zip(STOCKS, (100.0, 50.0, 200.0)))
    expiry_spots = dict(zip(STOCKS, (90.0, 40.0, 180.0)))

    def quotes(time, spots):
        return tuple(Quote(f"{time.isoformat()}/{key}", key, time, time, value, value, 10000, 10000)
                     for key, value in spots.items())

    events = (
        Event("start", START, 0, view.valuation, decision=True,
              earliest_execution=first_fill, order_deadline=HORIZON,
              origination_price=view.valuation.marks["B"]),
        Event("fill", first_fill, 0, pricing.packet(first_fill, start_spots, view.instruments, surfaces()),
              quotes=quotes(first_fill, start_spots), execution=True),
        Event("expiry", HORIZON, 0, pricing.packet(HORIZON, expiry_spots, view.instruments, {},
                                                  fixings={"B": expiry_spots}),
              fixings=(Fixing("official", "B", HORIZON, HORIZON, expiry_spots),),
              decision=True, terminal_decision=True, earliest_execution=final_fill, order_deadline=final_fill),
        Event("exit", final_fill, 0, pricing.packet(final_fill, expiry_spots, view.instruments, {}),
              quotes=quotes(final_fill, expiry_spots), execution=True),
    )
    rules = Rules(final_fill, STOCKS, 0, borrow={key: Borrow(True, 10000) for key in STOCKS},
                  funding=Funding(borrow_rates=dict.fromkeys(STOCKS, 0)))
    result = Engine(view.instruments, Opening("ORIGINATE", initial_capital=10000, liability_id="B"),
                    rules, UnnettedLiabilityStressMargin()).run(events, ScheduledDeltaPolicy())
    assert result.status == "COMPLETED" and len(result.fills) == 6
    assert type(result.settlement_outcomes[0]["funded"]) is bool
    write_reports([result], tmp_path)
    assert (tmp_path / "delta" / "summary.json").is_file()


@pytest.mark.parametrize("invalid", ["endpoint_execution", "old_24h", "expiry_before_close", "stale_mark", "basket_hedge"])
def test_historical_timing_and_hedge_boundaries(invalid):
    pricing, view = setup(expiry=START + timedelta(days=1) if invalid == "expiry_before_close" else EXPIRY)
    if invalid == "endpoint_execution":
        view = replace(view, rules={**view.rules, "earliest_execution": HORIZON})
    if invalid == "stale_mark":
        view = replace(view, valuation=replace(view.valuation, observed_at={"B": START - timedelta(days=1),
                                                                          **{key: START for key in STOCKS}}))
    if invalid == "basket_hedge":
        with pytest.raises(InputError, match="constituent stocks"):
            reprice_basket_scenarios(view, {}, ("B",), future_spots=np.ones((2, 3)),
                                     future_surfaces=np.ones((2, 3, 2, 2)), grid=GRID,
                                     horizon=HORIZON, valuator=pricing, seed=1)
        return
    with pytest.raises((InputError, DataGap)):
        changes(view, pricing, [[100, 50, 200]] * 2,
                horizon=START + timedelta(days=1) if invalid == "old_24h" else HORIZON)


def regression(response, design, seed):
    return ScenarioChanges(("S",), np.asarray(response, float), np.asarray(design, float).reshape(-1, 1),
                           HORIZON, seed)


def test_paper_aic_hand_calculation_uses_training_intercept_and_holdings():
    fit = regression([1, 5], [-1, 1], 1)  # Mean 3; slope 2; centered variance 1.
    validation = regression([-1, 7], [-2, 2], 2)
    selected = select_paper_alpha(fit, validation, [0], [1], gross_scale=10, alphas=(0.01, 0.2))
    # Soft threshold: beta=2-alpha*10/2 ->1.95 or1; intercept stays3.
    # Validation residuals ±.1 or±2 ->RSS .02 or8; both count intercept+slope.
    assert selected.alpha == 0.01
    assert selected.scores[0]["rss"] == pytest.approx(0.02)
    assert selected.scores[0]["aic"] == pytest.approx(2 * log(0.01) + 4)
    assert selected.scores[1]["rss"] == pytest.approx(8)
    assert selected.scores[1]["aic"] == pytest.approx(2 * log(4) + 4)
    shifted = select_paper_alpha(fit, regression([9, 17], [-2, 2], 2), [0], [1], gross_scale=10,
                                 alphas=(0.01,))
    assert shifted.scores[0]["rss"] == pytest.approx(200.02)  # Validation intercept not re-fit.
    assert PAPER_ALPHAS == tuple(i / 100 for i in range(1, 21))


def test_paper_aic_zero_trade_still_counts_existing_nonzero_holding():
    fit = regression([1, 5], [-1, 1], 1)
    result = select_paper_alpha(fit, regression([0, 7], [-2, 2], 2), [2], [1], gross_scale=10,
                                alphas=(0.2,))
    assert result.scores[0]["parameters"] == 2  # Intercept + held shares, although trade is zero.
    with pytest.raises(InputError, match="independent"):
        select_paper_alpha(fit, fit, [0], [1], gross_scale=10)


def test_aic_policy_selects_once_and_preserves_existing_strategy_interface():
    pricing, view = setup()

    class Source:
        def __init__(self, seed):
            self.seed, self.calls = seed, 0

        def generate(self, current, fixed, hedges):
            self.calls += 1
            return ScenarioChanges(hedges, np.array([-2.0, 2.0]),
                                   np.array([[-1.0, 0, 0], [1.0, 0, 0]]), HORIZON, self.seed)

    fit, validation = Source(1), Source(2)
    policy = PaperAICHedge(fit, validation, STOCKS, dict.fromkeys(STOCKS, 0.01))
    assert policy(replace(view, decision_event=False)) is None
    targets = policy(view)
    selection = policy.selection
    assert set(targets) == set(STOCKS) and targets["SPY"] > 0
    policy(replace(view, positions={**view.positions, **targets}))
    assert policy.selection is selection and validation.calls == 1
    assert view.cash == 10000
