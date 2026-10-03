"""Pricing checks against analytic numbers and reproducibility boundaries."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from basket_hedging.contracts import DataGap, InputError, Instrument
from basket_hedging.market import Fixing, Observation, PricingView, Quote, ValuationPacket
from basket_hedging.valuation import BasketValuator, PricingParameters, StressScenario, black_scholes


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
EXPIRY = START + timedelta(days=365)
SCENARIOS = (StressScenario("unchanged"), StressScenario("down", 0.8, 1.2), StressScenario("up", 1.2, 1.2))


def parameters(**kwargs):
    values = dict(stock_ids=("S",), volatilities=(0.2,), correlation=((1.0,),),
                  risk_free_rate=0.05, available_at=START, pricing_seed=2718,
                  samples=65536, bump_relative=1e-4)
    values.update(kwargs)
    return PricingParameters(**values)


def instruments(coefficient=1, multiplier=100, basket_kind="basket_put"):
    return {
        "S": Instrument("S", "stock"),
        "B": Instrument("B", basket_kind, strike=100, multiplier=multiplier,
                        expiry=EXPIRY, coefficients=(("S", coefficient),)),
        "P": Instrument("P", "put", strike=100, multiplier=multiplier, expiry=EXPIRY, underlying="S"),
        "C": Instrument("C", "call", strike=100, multiplier=multiplier, expiry=EXPIRY, underlying="S"),
    }


def test_black_scholes_against_independent_analytic_reference_numbers():
    # S=K=100, r=.05, sigma=.20, T=1: closed-form analytic reference values.
    call, call_delta = black_scholes(100, 100, 1, 0.05, 0.2, "call")
    put, put_delta = black_scholes(100, 100, 1, 0.05, 0.2, "put")
    assert call == pytest.approx(10.450583572185565, abs=1e-12)
    assert put == pytest.approx(5.573526022256971, abs=1e-12)
    assert call_delta == pytest.approx(0.6368306511756191, abs=1e-12)
    assert put_delta == pytest.approx(-0.3631693488243809, abs=1e-12)


@pytest.mark.parametrize("coefficient,spot,expected_delta", [(1, 100, -0.3631693488243809),
                                                            (2, 50, -0.7263386976487618)])
def test_one_stock_arithmetic_basket_price_and_delta_match_vanilla(coefficient, spot, expected_delta):
    value = BasketValuator(parameters(), SCENARIOS).packet(START, {"S": spot}, instruments(coefficient))
    # 65,536 antithetic draws: these deliberately loose numerical tolerances are
    # ~several MC standard errors (price standard error <.04, delta <.004 here).
    # They test pricing mechanics, not confidence about trading performance.
    assert value.marks["B"] == pytest.approx(5.573526022256971, abs=0.12)
    assert value.deltas["B"]["S"] == pytest.approx(expected_delta, abs=0.008)
    assert value.stress_marks["unchanged"] == value.marks


@pytest.mark.parametrize("coefficient,spot,expected_delta", [(1, 100, .6368306511756191),
                                                          (2, 50, 1.2736613023512382)])
def test_one_stock_basket_call_matches_independent_vanilla_values(coefficient, spot, expected_delta):
    value = BasketValuator(parameters(), SCENARIOS).packet(
        START, {"S": spot}, instruments(coefficient, basket_kind="basket_call"))
    # Analytic S_eff=K=100, r=.05, sigma=.2, T=1; delta scales by basket shares.
    # Monte Carlo is approximate: use the same justified tolerances as the put.
    assert value.marks["B"] == pytest.approx(10.450583572185565, abs=.12)
    assert value.deltas["B"]["S"] == pytest.approx(expected_delta, abs=.008)


def test_basket_call_exact_official_expiry_and_zero_volatility_price():
    valuator = BasketValuator(parameters(volatilities=(0,), risk_free_rate=0, samples=16), SCENARIOS)
    registry = instruments(basket_kind="basket_call")
    before = valuator.packet(START, {"S": 120}, registry)
    assert before.marks["B"] == 20
    assert before.deltas["B"]["S"] == pytest.approx(1, abs=1e-11)
    # The current unofficial spot is irrelevant to contractual settlement.
    expired = valuator.packet(EXPIRY, {"S": 80}, registry,
                  fixings={"B": {"S": 130}, "P": {"S": 130}, "C": {"S": 130}})
    assert expired.marks["B"] == 30
    assert expired.deltas == {}
    assert all(scenario["B"] == 30 for scenario in expired.stress_marks.values())


def test_unit_prices_and_deltas_do_not_multiply_by_contract_size():
    valuator = BasketValuator(parameters(samples=256), SCENARIOS)
    small = valuator.packet(START, {"S": 100}, instruments(multiplier=1))
    large = valuator.packet(START, {"S": 100}, instruments(multiplier=100))
    assert small.marks == large.marks
    assert small.deltas == large.deltas


def test_zero_volatility_arithmetic_basket_and_central_difference_are_deterministic():
    valuator = BasketValuator(parameters(volatilities=(0,), risk_free_rate=0, samples=16), SCENARIOS)
    packet = valuator.packet(START, {"S": 80}, instruments())
    assert packet.marks["B"] == 20
    assert packet.deltas["B"]["S"] == pytest.approx(-1, abs=1e-11)
    assert packet.marks["P"] == 20
    assert packet.marks["C"] == 0


@pytest.mark.parametrize("kwargs", [
    {"stock_ids": ("S", "S")}, {"volatilities": (-0.2,)}, {"volatilities": ()},
    {"risk_free_rate": float("nan")}, {"samples": 1}, {"samples": 1.5},
    {"samples": True}, {"pricing_seed": -1}, {"bump_relative": 0},
    {"bump_relative": 1}, {"correlation": ((0.9,),)}, {"correlation": ((float("nan"),),)},
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(InputError):
        parameters(**kwargs)


@pytest.mark.parametrize("correlation", [
    ((1, .2), (.3, 1)),  # Asymmetric.
    ((1, 1.1), (1.1, 1)),  # Not a correlation matrix.
    ((1, .9, .9), (.9, 1, -.9), (.9, -.9, 1)),  # Symmetric, bounded, indefinite.
])
def test_invalid_correlation_is_not_repaired(correlation):
    n = len(correlation)
    with pytest.raises(InputError):
        parameters(stock_ids=tuple(f"S{i}" for i in range(n)), volatilities=(0.2,) * n,
                   correlation=correlation)


def test_singular_valid_correlation_is_supported_and_roundoff_disclosed():
    config = parameters(stock_ids=("A", "Z"), volatilities=(.2, .2), correlation=((1, 1), (1, 1)), samples=128)
    book = {"A": Instrument("A", "stock"), "Z": Instrument("Z", "stock"),
            "B": Instrument("B", "basket_put", expiry=EXPIRY, strike=100,
                            coefficients=(("A", .5), ("Z", .5)))}
    value = BasketValuator(config, SCENARIOS).packet(START, {"A": 100, "Z": 100}, book)
    assert value.marks["B"] >= 0
    assert value.metadata["correlation_eigenvalue_tolerance"] == 1e-12
    assert value.metadata["correlation_minimum_eigenvalue"] == 0


@pytest.mark.parametrize("scenarios", [(), (StressScenario("down", .9),),
                                          (StressScenario("unchanged", .9),),
                                          (StressScenario("unchanged"), StressScenario("unchanged"))])
def test_stress_names_and_unchanged_scenario_are_validated(scenarios):
    with pytest.raises(InputError):
        BasketValuator(parameters(samples=16), scenarios)


def test_unreleased_parameters_future_observations_and_invalid_spots_are_rejected():
    valuator = BasketValuator(parameters(samples=16), SCENARIOS)
    with pytest.raises(InputError, match="parameters were not available"):
        valuator.packet(START - timedelta(seconds=1), {"S": 100}, instruments())
    with pytest.raises(InputError, match="future observations"):
        valuator.packet(START, {"S": 100}, instruments(), observed_at={"S": START + timedelta(seconds=1)})
    for spot in (0, -1, float("nan"), float("inf")):
        with pytest.raises(InputError):
            valuator.packet(START, {"S": spot}, instruments())
    with pytest.raises(DataGap, match="missing pricing spot"):
        valuator.packet(START, {}, instruments())


def test_expiry_requires_official_fixing_and_ignores_unofficial_replay_spot():
    valuator = BasketValuator(parameters(samples=16), SCENARIOS)
    with pytest.raises(DataGap, match="official fixing"):
        valuator.packet(EXPIRY, {"S": 140}, instruments())
    fixing = {"B": {"S": 90}, "P": {"S": 90}, "C": {"S": 90}}
    value = valuator.packet(EXPIRY, {"S": 140}, instruments(), fixings=fixing)
    assert value.marks == {"S": 140, "B": 10, "P": 10, "C": 0}
    assert value.deltas == {}
    assert all(stress["B"] == 10 for stress in value.stress_marks.values())
    after = valuator.packet(EXPIRY + timedelta(seconds=1), {"S": 140}, instruments())
    assert after.marks == {"S": 140}
    assert after.deltas == {}


def test_packet_records_oldest_underlying_observation_and_provenance():
    config = parameters(stock_ids=("A", "Z"), volatilities=(.2, .3), correlation=((1, .2), (.2, 1)), samples=32)
    book = {"A": Instrument("A", "stock"), "Z": Instrument("Z", "stock"),
            "B": Instrument("B", "basket_put", expiry=EXPIRY, strike=100,
                            coefficients=(("A", .5), ("Z", .5)))}
    older = START - timedelta(seconds=30)
    value = BasketValuator(config, SCENARIOS).packet(START, {"A": 100, "Z": 100}, book,
                 observed_at={"A": START, "Z": older}, input_ids=("obs-A", "obs-Z"))
    assert value.observed_at["B"] == older
    assert value.input_ids == ("obs-A", "obs-Z")
    assert value.metadata["parameters"]["pricing_seed"] == 2718
    assert value.metadata["parameters"]["samples"] == 32
    assert value.metadata["scenarios"][1]["name"] == "down"


def test_pricing_is_reproducible_and_does_not_consume_market_randomness():
    config = parameters(samples=128)
    valuator = BasketValuator(config, SCENARIOS)
    market = np.random.Generator(np.random.PCG64(42))
    reference_market = np.random.Generator(np.random.PCG64(42))
    before = market.standard_normal(3)
    first = valuator.packet(START, {"S": 100}, instruments())
    second = BasketValuator(config, SCENARIOS).packet(START, {"S": 100}, instruments())
    other_count = BasketValuator(replace(config, samples=256), SCENARIOS).packet(START, {"S": 100}, instruments())
    after = market.standard_normal(3)
    assert np.array_equal(np.concatenate((before, after)), reference_market.standard_normal(6))
    assert first == second
    assert first.metadata["pricing_seed_derived"] == other_count.metadata["pricing_seed_derived"]
    # The same UTC instant in a different timezone has the same pricing seed/draws.
    shifted = valuator.packet(START.astimezone(timezone(timedelta(hours=2))), {"S": 100}, instruments())
    assert shifted.marks == first.marks
    assert shifted.deltas == first.deltas


def test_pricing_parameters_detach_mutable_input_sequences():
    stocks, volatilities, correlation = ["S"], [.2], [[1.0]]
    config = parameters(stock_ids=stocks, volatilities=volatilities, correlation=correlation, samples=16)
    stocks[0] = "other"
    volatilities[0] = .9
    correlation[0][0] = .5
    assert config.stock_ids == ("S",)
    assert config.volatilities == (.2,)
    assert config.correlation == ((1.0,),)


def test_pricing_outputs_are_stable_across_processes_and_python_hash_seeds():
    script = """
from datetime import datetime, timedelta, timezone
import json
from basket_hedging.contracts import Instrument
from basket_hedging.replay import jsonable
from basket_hedging.valuation import BasketValuator, PricingParameters, StressScenario
t = datetime(2026, 1, 1, tzinfo=timezone.utc)
p = PricingParameters(('S',), (.2,), ((1.,),), .05, t, 2718, samples=128)
i = {'S': Instrument('S', 'stock'), 'B': Instrument('B', 'basket_put', strike=100,
     expiry=t+timedelta(days=365), coefficients=(('S', 1),))}
v = BasketValuator(p, (StressScenario('unchanged'),)).packet(t, {'S': 100}, i)
print(json.dumps(jsonable({'marks': v.marks, 'deltas': v.deltas, 'seed': v.metadata['pricing_seed_derived']}), sort_keys=True))
"""
    outputs = []
    for hash_seed in ("1", "98765"):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
        completed = subprocess.run([sys.executable, "-c", script], check=True, text=True,
                                   capture_output=True, env=environment)
        outputs.append(json.loads(completed.stdout))
    assert outputs[0] == outputs[1]


def pricing_view(time=START, *, observed_at=None, fixings=None, supplied_packet=None):
    observation = Observation("released-S", "S", 100, observed_at or time, time)
    return PricingView(time, instruments(), {"S": observation}, {}, fixings or {},
                       (observation,), supplied_packet)


def test_model_adapter_preserves_original_observation_age_and_calibration_cutoff():
    time = START + timedelta(days=1)
    view = pricing_view(time, observed_at=START)
    value = BasketValuator(parameters(samples=32), SCENARIOS).value(view)
    assert value.as_of == time
    assert value.observed_at == {"S": START, "B": START, "P": START, "C": START}
    assert value.input_ids == ("released-S",)
    assert value.calibration_cutoff == START
    assert value.metadata["parameters"]["available_at"] == START.isoformat()
    assert not hasattr(view, "positions") and not hasattr(view, "portfolio")


def test_model_adapter_uses_observations_and_ignores_supplied_valuation_marks():
    valuator = BasketValuator(parameters(samples=32), SCENARIOS)
    view = pricing_view()
    first = valuator.value(view)
    arbitrary = ValuationPacket(START, {"S": 999, "B": 999}, {"unchanged": {"S": 999, "B": 999}})
    other = valuator.value(replace(view, supplied_packet=arbitrary))
    assert first == other
    assert first.marks["S"] == 100
    # Repeated valuations of identical inputs do not depend on caller/account.
    assert valuator.value(view) == first


@pytest.mark.parametrize("field", ["market", "history", "quotes", "fixings"])
def test_model_adapter_rejects_unreleased_view_records(field):
    future = START + timedelta(seconds=1)
    observation = Observation("future-S", "S", 100, START, future)
    values = {
        "market": {"S": observation}, "history": (observation,),
        "quotes": {"S": Quote("future-quote", "S", START, future, 99, 101, 100, 100)},
        "fixings": {"B": Fixing("future-fixing", "B", START, future, {"S": 100})},
    }
    with pytest.raises(InputError, match="unreleased"):
        BasketValuator(parameters(samples=16), SCENARIOS).value(replace(pricing_view(), **{field: values[field]}))


def test_model_adapter_requires_released_calibration_and_stock_inputs():
    valuator = BasketValuator(parameters(samples=16), SCENARIOS)
    with pytest.raises(InputError, match="parameters were not available"):
        valuator.value(pricing_view(START - timedelta(seconds=1)))
    with pytest.raises(DataGap, match="missing pricing spot"):
        valuator.value(replace(pricing_view(), market={}))


def test_model_adapter_uses_official_expiry_only_for_contracts_with_released_fixings():
    valuator = BasketValuator(parameters(samples=16), SCENARIOS)
    fixing = Fixing("official-B", "B", EXPIRY, EXPIRY, {"S": 90})
    view = pricing_view(EXPIRY, fixings={"B": fixing})
    value = valuator.value(view)
    # S observation is 100; official basket fixing 90 means exact put payoff 10.
    assert value.marks == {"S": 100, "B": 10}
    assert value.deltas == {}
    assert value.input_ids == ("released-S", "official-B")
    assert all(scenario["B"] == 10 for scenario in value.stress_marks.values())
    # Candidate calls/puts without fixing are omitted, not invented or priced by MC.
    assert valuator.value(pricing_view(EXPIRY)).marks == {"S": 100}
    assert valuator.value(pricing_view(EXPIRY + timedelta(seconds=1))).marks == {"S": 100}


def test_model_adapter_rejects_fixing_with_wrong_official_time():
    fixing = Fixing("old-B", "B", EXPIRY - timedelta(seconds=1), EXPIRY, {"S": 90})
    with pytest.raises(InputError, match="match the contract and expiry"):
        BasketValuator(parameters(samples=16), SCENARIOS).value(pricing_view(EXPIRY, fixings={"B": fixing}))
