"""Conditional changes use current state, signed units and independent toy draws."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from basket_hedging.contracts import DataGap, InputError, Instrument
from basket_hedging.engine import PolicyView
from basket_hedging.market import Observation, ValuationPacket
from basket_hedging.scenarios import ConditionalGBMScenarios, ScenarioChanges
from basket_hedging.valuation import PricingParameters


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
EXPIRY = START + timedelta(days=30)


def parameters(**changes):
    values = dict(stock_ids=("S",), volatilities=(0.0,), correlation=((1.0,),),
                  risk_free_rate=0.0, available_at=START, pricing_seed=71, samples=32)
    values.update(changes)
    return PricingParameters(**values)


def view(*, expiry=EXPIRY, basket_kind="basket_put"):
    registry = {
        "S": Instrument("S", "stock"),
        "B": Instrument("B", basket_kind, multiplier=10, strike=110, expiry=expiry,
                        coefficients=(("S", 1),)),
        "P": Instrument("P", "put", multiplier=5, strike=105, expiry=expiry, underlying="S"),
        "C": Instrument("C", "call", multiplier=20, strike=90, expiry=expiry, underlying="S"),
    }
    marks = {"S": 100.0, "B": 8.0, "P": 3.0, "C": 7.0}
    packet = ValuationPacket(START, marks, {"unchanged": marks},
                             observed_at={key: START for key in marks})
    observation = Observation("released-S", "S", 100.0, START, START)
    return PolicyView(event_id="decision", time=START, sequence=0, instruments=registry,
                      positions={"B": -2}, cash=10000, posted_margin_cash=0,
                      restricted_short_proceeds=0, available_cash=10000, valuation=packet,
                      history=(observation,), order_history=(), hedge_universe=("S", "P", "C"),
                      terminal_decision=False,
                      rules={"earliest_execution": START + timedelta(hours=1),
                             "next_rebalance_time": START + timedelta(days=1)},
                      market={"S": observation})


def source(pricing=None, **changes):
    values = dict(replay_volatilities=(0.0,), replay_correlation=((1.0,),), samples=4)
    values.update(changes)
    return ConditionalGBMScenarios(pricing or parameters(), **values)


@pytest.mark.parametrize("quantity,expected", [(-2, 40.0), (2, -40.0)])
def test_zero_volatility_signed_response_and_candidate_contract_multipliers(quantity, expected):
    current = view()
    generated = source().generate(current, {"B": quantity}, ("S", "P", "C"))
    # At unchanged S=100 and r=sigma=0: basket put=10, constituent put=5, call=10.
    # Current declared marks 8/3/7 intentionally test level differences (an intercept).
    # -q*10*(10-8); per-share 100-100; per-contract 5*(5-3), 20*(10-7).
    np.testing.assert_array_equal(generated.delta_v, [expected] * 4)
    np.testing.assert_array_equal(generated.delta_h, [[0.0, 10.0, 60.0]] * 4)
    assert generated.hedge_ids == ("S", "P", "C")
    assert generated.horizon == START + timedelta(days=1)
    assert generated.input_ids == ("released-S",)
    assert current.positions == {"B": -2} and current.cash == 10000


def test_fixed_book_can_include_stock_and_constituent_option_without_double_counting_candidates():
    generated = source().generate(view(), {"B": -2, "P": 3, "S": 5}, ("C",))
    # Basket contributes +40 loss response; long three puts contribute -3*5*(5-3).
    np.testing.assert_array_equal(generated.delta_v, [10.0] * 4)
    np.testing.assert_array_equal(generated.delta_h, [[60.0]] * 4)
    empty = source().generate(view(), {}, ())
    np.testing.assert_array_equal(empty.delta_v, np.zeros(4))
    assert empty.delta_h.shape == (4, 0)


def test_expiry_truncation_uses_hypothetical_joint_spots_for_exact_payoff():
    expiry = START + timedelta(hours=6)
    current = view(expiry=expiry)
    generated = source(replay_volatilities=(0.4,), samples=16).generate(current, {"B": -2}, ("S", "P"))
    future_spot = generated.delta_h[:, 0] + 100.0
    # This independent formula checks joint scenarios and exact expiry payoffs.
    np.testing.assert_allclose(generated.delta_v, 20 * (np.maximum(110 - future_spot, 0) - 8),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(generated.delta_h[:, 1], 5 * (np.maximum(105 - future_spot, 0) - 3),
                               rtol=0, atol=1e-12)
    assert generated.horizon == expiry
    assert not hasattr(current, "fixings")  # No realized future fixing is supplied.


@pytest.mark.parametrize("quantity", [-2, 2])
def test_basket_call_response_uses_signed_obligation_change(quantity):
    current = view(expiry=START + timedelta(hours=6), basket_kind="basket_call")
    generated = source(replay_volatilities=(0.4,), samples=8).generate(current, {"B": quantity}, ("S",))
    future_spot = generated.delta_h[:, 0] + 100.0
    np.testing.assert_allclose(generated.delta_v, -quantity * 10 * (np.maximum(future_spot - 110, 0) - 8),
                               rtol=0, atol=1e-12)


def test_zero_strike_call_has_exact_stock_hedge_in_every_scenario():
    current = view(basket_kind="basket_call")
    registry = dict(current.instruments)
    registry["B"] = replace(registry["B"], strike=0, coefficients=(("S", 2),))
    marks = {"S": 100.0, "B": 200.0}
    packet = ValuationPacket(START, marks, {"unchanged": marks}, observed_at={key: START for key in marks})
    current = replace(current, instruments=registry, valuation=packet)
    generated = source(replay_volatilities=(0.2,), samples=8).generate(current, {"B": -2}, ("S",))
    # Short two contracts * multiplier ten * two shares per basket unit => +40-share hedge.
    np.testing.assert_allclose(generated.delta_v, 40 * generated.delta_h[:, 0], rtol=0, atol=1e-11)
    assert np.ptp(generated.delta_h[:, 0]) > 0


def test_earlier_candidate_expiry_truncates_step_but_unrelated_expiry_does_not():
    current = view()
    registry = dict(current.instruments)
    registry["P"] = replace(registry["P"], expiry=START + timedelta(hours=3))
    current = replace(current, instruments=registry)
    assert source().generate(current, {"B": -2}, ("S",)).horizon == START + timedelta(days=1)
    assert source().generate(current, {"B": -2}, ("P",)).horizon == START + timedelta(hours=3)


def test_repetition_sample_prefix_and_holdings_do_not_change_conditional_stock_draws():
    current = view()
    generator = source(replay_volatilities=(0.2,), samples=8, seed=123)
    first = generator.generate(current, {"B": -2}, ("S", "P"))
    second = generator.generate(current, {"B": -2}, ("S", "P"))
    doubled = generator.generate(current, {"B": -4}, ("S", "P"))
    more = source(replay_volatilities=(0.2,), samples=12, seed=123).generate(current, {"B": -2}, ("S", "P"))
    reordered = generator.generate(current, {"B": -2}, ("P", "S"))
    np.testing.assert_array_equal(first.delta_v, second.delta_v)
    np.testing.assert_array_equal(first.delta_h, second.delta_h)
    np.testing.assert_array_equal(first.delta_h, doubled.delta_h)
    np.testing.assert_array_equal(first.delta_v * 2, doubled.delta_v)
    np.testing.assert_array_equal(first.delta_h, more.delta_h[:8])
    np.testing.assert_array_equal(first.delta_h[:, ::-1], reordered.delta_h)
    changed = source(replay_volatilities=(0.2,), samples=8, seed=124).generate(current, {"B": -2}, ("S",))
    assert not np.array_equal(first.delta_h[:, 0], changed.delta_h[:, 0])


def test_pricing_seed_and_sample_count_do_not_change_physical_scenario_draws():
    current = view()
    physical = dict(replay_volatilities=(0.2,), samples=8, seed=123)
    first = source(parameters(volatilities=(0.2,), samples=32), **physical).generate(current, {"B": -2}, ("S",))
    second = source(parameters(volatilities=(0.2,), samples=128, pricing_seed=72), **physical).generate(
        current, {"B": -2}, ("S",))
    np.testing.assert_array_equal(first.delta_h, second.delta_h)
    assert not np.array_equal(first.delta_v, second.delta_v)  # Inner MC remains approximate.


def test_scenarios_use_current_state_not_future_history_or_unrelated_mark_fields():
    current = view()
    future = START + timedelta(days=10)
    future_observation = Observation("unreleased-future", "S", 999, future, future)
    first = source(replay_volatilities=(0.2,)).generate(current, {"B": -2}, ("S",))
    changed = replace(current, history=current.history + (future_observation,),
                      cash=-999, positions={"B": -999})
    second = source(replay_volatilities=(0.2,)).generate(changed, {"B": -2}, ("S",))
    np.testing.assert_array_equal(first.delta_v, second.delta_v)
    np.testing.assert_array_equal(first.delta_h, second.delta_h)
    # Unrelated options need neither marks nor fresh observations.
    marks = {"S": 100.0, "B": 8.0}
    packet = ValuationPacket(START, marks, {"unchanged": marks}, observed_at={key: START for key in marks})
    third = source(replay_volatilities=(0.2,)).generate(replace(current, valuation=packet), {"B": -2}, ("S",))
    np.testing.assert_array_equal(first.delta_v, third.delta_v)


def test_scenario_arrays_are_detached_and_cannot_regain_write_permission():
    response, hedges = np.array([1.0, 2.0]), np.array([[3.0], [4.0]])
    result = ScenarioChanges(("S",), response, hedges, EXPIRY, 1)
    response[:] = 9
    hedges[:] = 9
    np.testing.assert_array_equal(result.delta_v, [1.0, 2.0])
    np.testing.assert_array_equal(result.delta_h, [[3.0], [4.0]])
    for values in (result.delta_v, result.delta_h):
        with pytest.raises(ValueError):
            values.flat[0] = 0
        with pytest.raises(ValueError):
            values.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.seed = 2


@pytest.mark.parametrize("response,hedges", [
    ([1], [[1]]), ([[1], [2]], [[1], [2]]), ([1, 2], [1, 2]),
    ([1, 2], [[1, 2], [3, 4]]), ([float("nan"), 2], [[1], [2]]),
    ([1, 2], [[1], [float("inf")]]),
])
def test_scenario_change_dimensions_and_finiteness_are_validated(response, hedges):
    with pytest.raises(InputError):
        ScenarioChanges(("S",), response, hedges, EXPIRY, 1)


@pytest.mark.parametrize("changes", [
    {"samples": 1}, {"samples": True}, {"seed": -1}, {"max_observation_age_seconds": -1},
    {"replay_volatilities": (-0.1,)}, {"replay_volatilities": ()},
    {"replay_correlation": ((0.9,),)}, {"replay_correlation": (1.0,)},
])
def test_invalid_generator_configuration_is_rejected(changes):
    with pytest.raises(InputError):
        source(**changes)


@pytest.mark.parametrize("hedges,fixed", [
    (("B",), {}), (("unknown",), {}), (("S", "S"), {}),
    (("S",), {"S": 1}), (("S",), {"unknown": 1}),
    (("S",), {"B": float("nan")}),
])
def test_invalid_candidates_or_fixed_positions_are_rejected(hedges, fixed):
    with pytest.raises(InputError):
        source().generate(view(), fixed, hedges)


def test_unpermitted_expired_unreleased_and_stale_inputs_are_rejected():
    current = view()
    with pytest.raises(InputError, match="permitted"):
        source().generate(replace(current, hedge_universe=("S",)), {"B": -2}, ("P",))
    with pytest.raises(InputError, match="expiry"):
        source().generate(view(expiry=START), {"B": -2}, ("S",))
    with pytest.raises(InputError, match="parameters were not available"):
        source(parameters(available_at=START + timedelta(seconds=1))).generate(current, {"B": -2}, ("S",))
    delayed = Observation("future-release", "S", 100, START, START + timedelta(seconds=1))
    with pytest.raises(InputError, match="released"):
        source().generate(replace(current, market={"S": delayed}), {"B": -2}, ("S",))
    old = Observation("old", "S", 100, START - timedelta(seconds=1), START)
    with pytest.raises(DataGap, match="stale conditional spot"):
        source().generate(replace(current, market={"S": old}), {"B": -2}, ("S",))
    with pytest.raises(DataGap, match="missing conditional spot"):
        source().generate(replace(current, market={}), {"B": -2}, ("S",))
    packet = replace(current.valuation, observed_at={"B": START - timedelta(seconds=1), "S": START})
    with pytest.raises(DataGap, match="stale current scenario mark"):
        source().generate(replace(current, valuation=packet), {"B": -2}, ("S",))


def test_missing_candidate_mark_is_a_gap_and_zero_option_mark_is_valid():
    current = view()
    marks = {"S": 100.0, "B": 8.0}
    packet = ValuationPacket(START, marks, {"unchanged": marks}, observed_at={key: START for key in marks})
    with pytest.raises(DataGap, match="missing current scenario mark/timestamp for P"):
        source().generate(replace(current, valuation=packet), {"B": -2}, ("P",))
    marks["P"] = 0.0
    packet = ValuationPacket(START, marks, {"unchanged": marks}, observed_at={key: START for key in marks})
    result = source().generate(replace(current, valuation=packet), {"B": -2}, ("P",))
    np.testing.assert_array_equal(result.delta_h, [[25.0]] * 4)  # Five-share contract * (5 - 0).


def test_scenario_seed_is_stable_across_python_hash_seeds_and_utc_representations():
    current = view()
    shifted = replace(current, time=START.astimezone(timezone(timedelta(hours=2))))
    generator = source(replay_volatilities=(0.2,))
    np.testing.assert_array_equal(generator.generate(current, {"B": -2}, ("S",)).delta_h,
                                  generator.generate(shifted, {"B": -2}, ("S",)).delta_h)
    script = """
from datetime import datetime, timezone
from basket_hedging.scenarios import ConditionalGBMScenarios
from basket_hedging.valuation import PricingParameters
t = datetime(2026, 1, 1, tzinfo=timezone.utc)
p = PricingParameters(('S',), (.2,), ((1.,),), 0, t, 71, samples=4)
g = ConditionalGBMScenarios(p, replay_volatilities=(.2,), replay_correlation=((1.,),), samples=4)
print(g._seed(t, 0))
"""
    outputs = []
    for hash_seed in ("1", "999"):
        result = subprocess.run([sys.executable, "-B", "-c", script], check=True,
                                capture_output=True, text=True,
                                env=dict(os.environ, PYTHONHASHSEED=hash_seed))
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]
