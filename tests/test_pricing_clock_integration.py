"""The dynamic clock, valuation boundary and risk checks share causal inputs."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from basket_hedging.contracts import Funding, InputError, Instrument, Opening
from basket_hedging.engine import Engine
from basket_hedging.market import Event, Observation, Quote, Rules, ValuationPacket
from basket_hedging.risk import UnnettedLiabilityStressMargin
from basket_hedging.valuation import BasketValuator, PricingParameters, StressScenario


START = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
ACTIVATION = START + timedelta(hours=1)
DEADLINE = START + timedelta(hours=2)
HORIZON = START + timedelta(hours=3)
EXPIRY = START + timedelta(days=30)


def instruments(expiry=EXPIRY):
    return {"S": Instrument("S", "stock"),
            "B": Instrument("B", "basket_put", multiplier=100, strike=105,
                            expiry=expiry, coefficients=(("S", 1),))}


def pricer(*, volatility=0, available_at=START):
    parameters = PricingParameters(("S",), (volatility,), ((1.0,),), 0,
                                   available_at, 2718, samples=32)
    return BasketValuator(parameters, (StressScenario("unchanged"), StressScenario("down", 0.9)))


def opening():
    # With zero volatility and zero pricing rate, this put is exactly 105-100=5.
    return Opening("IMPORT", cash=2000, positions={"B": -1},
                   marks={"B": 5}, opening_equity=1500)


def supplied(time, *, include_down=True):
    marks = {"S": 100, "B": 5}
    stresses = {"unchanged": marks}
    if include_down:
        stresses["down"] = {"S": 90, "B": 15}
    return ValuationPacket(time, marks, stresses, {"B": {"S": -1}},
                           {key: time for key in marks}, time)


def start_event(*, valuation=None, include_quote=False):
    quotes = (Quote("initial-quote", "S", START, START, 99, 101, 100, 100),) if include_quote else ()
    return Event("start", START, 0, valuation,
                 observations=(Observation("initial-S", "S", 100, START, START),), quotes=quotes)


def rules(**changes):
    return replace(Rules(HORIZON, ("S",), 0, start_time=START,
                         execution_delay_seconds=3600, order_lifetime_seconds=7200,
                         max_mark_age_seconds=10800,
                         expected_stress_scenarios=("unchanged", "down")), **changes)


def initial_stock_order(view):
    return {"S": 1} if view.event_id == "start" else None


@pytest.mark.parametrize("missing_boundary", [ACTIVATION, DEADLINE])
def test_missing_supplied_packet_at_generated_order_boundary_records_gap_and_keeps_obligations(missing_boundary):
    events = [start_event(valuation=supplied(START))]
    if missing_boundary == DEADLINE:
        events.append(Event("activation-record", ACTIVATION, 0, supplied(ACTIVATION)))
    events.append(Event("later-record", HORIZON, 0, supplied(HORIZON)))
    result = Engine(instruments(), opening(), rules(), UnnettedLiabilityStressMargin()).run(
        events, initial_stock_order)
    assert result.status == "STOP_DATA_GAP"
    assert result.data_gaps[-1]["time"] == missing_boundary
    assert "valuation packet" in result.reason
    assert result.ledger.positions == {"B": -1}
    assert result.ledger.cash == 2000
    assert result.ledger.settlements == {}
    assert result.fills == () and result.breaches == []
    assert result.last_valid_snapshot.time == (START if missing_boundary == ACTIVATION else ACTIVATION)
    assert result.snapshots[-1].risk.equity is None
    assert all(record["as_of"] < missing_boundary for record in result.valuations)


def test_model_prices_generated_timers_from_old_observations_without_creating_quotes_or_fills():
    views = []

    def observe_and_submit(view):
        views.append(view)
        return initial_stock_order(view)

    result = Engine(instruments(), opening(), rules(), UnnettedLiabilityStressMargin(), pricer()).run(
        [start_event(include_quote=True), Event("end", HORIZON, 0)], observe_and_submit)
    assert result.status == "INCOMPLETE_EXIT"
    assert [view.time for view in views] == [START, ACTIVATION, DEADLINE, HORIZON]
    for view in views:
        assert view.valuation.as_of == view.time
        assert view.valuation.observed_at["B"] == START
        assert view.valuation.observed_at["S"] == START
        assert view.valuation.marks["B"] == pytest.approx(5, abs=1e-12)
        assert view.valuation.input_ids == ("initial-S",)
        assert view.quotes["S"].time == START
    assert result.fills == ()  # Neither a model mark nor a timer creates a new quote.
    assert result.ledger.cash == 2000
    assert result.ledger.equity == pytest.approx(1500)
    assert any(order.time == DEADLINE and order.outcome == "expired" for order in result.order_outcomes)


def test_old_model_inputs_stop_at_first_timer_exceeding_permitted_age():
    result = Engine(instruments(), opening(), rules(max_mark_age_seconds=3600),
                    UnnettedLiabilityStressMargin(), pricer()).run(
        [start_event(), Event("later-record", HORIZON, 0)], initial_stock_order)
    assert result.status == "STOP_DATA_GAP"
    assert result.data_gaps[-1]["time"] == DEADLINE
    assert "stale valuation input" in result.reason
    assert result.last_valid_snapshot.time == ACTIVATION
    assert result.last_valid_snapshot.risk.equity == pytest.approx(1500)
    assert result.snapshots[-1].risk.equity is None
    assert result.ledger.positions == {"B": -1}
    assert result.ledger.cash == 2000


def test_missing_official_fixing_stops_at_contractual_expiry_between_source_records():
    result = Engine(instruments(expiry=ACTIVATION), opening(), rules(),
                    UnnettedLiabilityStressMargin(), pricer()).run(
        [start_event(), Event("much-later", HORIZON, 0,
                             observations=(Observation("later-S", "S", 90, HORIZON, HORIZON),))],
        lambda view: None)
    assert result.status == "STOP_DATA_GAP"
    assert result.data_gaps[-1]["time"] == ACTIVATION
    assert "fixing" in result.reason
    assert result.data_gaps[-1]["mandatory_phase"] == "unposted"
    assert result.data_gaps[-1]["due_option_ids"] == ["B"]
    assert result.ledger.cash == 2000
    assert result.ledger.positions == {"B": -1}
    assert result.ledger.settlements == {}
    assert result.last_valid_snapshot.time == START
    assert all(record["as_of"] < HORIZON for record in result.valuations)


def test_released_funding_change_accrues_previous_interval_at_previous_rate_with_model_pricing():
    day_one, day_two = START + timedelta(days=1), START + timedelta(days=2)
    stock_only = {"S": Instrument("S", "stock")}
    events = [start_event(), Event("rate-change", day_one, 0, funding=Funding(cash_rate=0.730)),
              Event("end", day_two, 0)]
    settings = Rules(day_two, ("S",), 0, start_time=START, funding=Funding(cash_rate=0.365),
                     max_mark_age_seconds=172800, expected_stress_scenarios=("unchanged", "down"))
    result = Engine(stock_only, Opening("IMPORT", cash=1000, opening_equity=1000), settings,
                    UnnettedLiabilityStressMargin(), pricer()).run(events, lambda view: None)
    # First day: 1,000 * .365 / 365 = 1. Next: 1,001 * .730 / 365 = 2.002.
    interest = [post.cash_change for post in result.ledger.postings if post.kind == "interest"]
    assert interest == pytest.approx([1, 2.002], abs=1e-12)
    assert result.ledger.cash == pytest.approx(1003.002, abs=1e-10)
    assert result.ledger.components()["net_pnl"] == pytest.approx(3.002, abs=1e-10)
    assert result.status == "COMPLETED"


def test_engine_rejects_model_parameters_not_available_at_simulated_inception():
    engine = Engine(instruments(), opening(), rules(), UnnettedLiabilityStressMargin(),
                    pricer(available_at=ACTIVATION))
    with pytest.raises(InputError, match="parameters were not available"):
        engine.run([start_event(), Event("end", HORIZON, 0)], lambda view: None)
    assert engine.ledger.cash == 2000
    assert engine.ledger.positions == {"B": -1}
    assert not engine.ledger.settlements


def test_identical_released_inputs_produce_identical_policy_packets_and_repricing():
    records = [start_event(), Event("same-inputs", START, 1),
               Event("later", HORIZON, 0,
                     observations=(Observation("later-S", "S", 101, HORIZON, HORIZON),),
                     quotes=(Quote("later-quote", "S", HORIZON, HORIZON, 100, 102, 100, 100),))]
    policy_packets = []
    pricing_packets = []
    results = []
    for should_trade in (False, True):
        model = pricer(volatility=0.2)
        packets = []
        observed = []

        class RepricingProbe:
            def value(self, view):
                first = model.value(view)
                second = model.value(view)
                assert first == second
                packets.append(first)
                return first

        def strategy(view):
            observed.append(view.valuation)
            return {"S": 1} if should_trade else None

        settings = rules(execution_delay_seconds=0, order_lifetime_seconds=None)
        result = Engine(instruments(), Opening("IMPORT", cash=10000, opening_equity=10000),
                        settings, UnnettedLiabilityStressMargin(), RepricingProbe()).run(records, strategy)
        results.append(result)
        pricing_packets.append(packets)
        policy_packets.append(observed)
    assert pricing_packets[0] == pricing_packets[1]
    assert policy_packets[0] == policy_packets[1]
    assert policy_packets[0][0] == policy_packets[0][1]  # Same time, same released inputs.
    assert results[0].ledger.cash == 10000
    assert results[1].ledger.cash == 9898  # One share bought at the later ask of 102.
    assert results[1].ledger.equity == 9999


@pytest.mark.parametrize("explicit_declaration", [False, True])
def test_removing_entire_declared_stress_scenario_cannot_make_account_appear_feasible(explicit_declaration):
    settings = rules(expected_stress_scenarios=("unchanged", "down") if explicit_declaration else None)
    imported = Opening("IMPORT", cash=1300, positions={"B": -1}, marks={"B": 2}, opening_equity=1100)
    initial_marks = {"S": 100, "B": 2}
    initial_packet = ValuationPacket(START, initial_marks,
        {"unchanged": initial_marks, "down": {"S": 90, "B": 4}},
        observed_at={"S": START, "B": START})
    final_marks = {"S": 90, "B": 6}
    complete = ValuationPacket(HORIZON, final_marks,
        {"unchanged": final_marks, "down": {"S": 80, "B": 8}},
        observed_at={"S": HORIZON, "B": HORIZON})
    missing = replace(complete, stress_marks={"unchanged": final_marks})
    result = Engine(instruments(), imported, settings, UnnettedLiabilityStressMargin()).run(
        [start_event(valuation=initial_packet), Event("missing-down", HORIZON, 0, missing)],
        lambda view: None)
    assert result.status == "STOP_DATA_GAP"
    assert "down" in result.reason
    assert result.last_valid_snapshot.time == START
    assert result.last_valid_snapshot.risk.required_margin == 400  # 200 liability + 200 stress loss.
    assert result.ledger.cash == 1300
    assert result.ledger.positions == {"B": -1}
    assert result.snapshots[-1].risk.required_margin is None
    full_result = Engine(instruments(), imported, settings, UnnettedLiabilityStressMargin()).run(
        [start_event(valuation=initial_packet), Event("complete-down", HORIZON, 0, complete)],
        lambda view: None)
    assert full_result.status == "STOP_WITH_BREACH"
    # Full final proxy is 600 liability + 200 stress loss = 800, against equity 700.
    assert full_result.breaches[0].risk.required_margin == 800
    assert full_result.breaches[0].risk.equity == 700
    assert full_result.breaches[0].risk.margin_equity_shortfall == 100


def test_imported_position_limit_failure_is_an_actual_breach_before_policy_or_fill():
    stock_only = {"S": Instrument("S", "stock")}
    imported = Opening("IMPORT", cash=1000, positions={"S": 3}, marks={"S": 100}, opening_equity=1300)
    settings = Rules(HORIZON, ("S",), 0, maximum_position_quantities={"S": 2})
    called = []

    def reducing_policy(view):
        called.append(view.time)
        return {"S": 0}

    result = Engine(stock_only, imported, settings, UnnettedLiabilityStressMargin(), pricer()).run(
        [start_event(include_quote=True), Event("end", HORIZON, 0)], reducing_policy)
    assert result.status == "STOP_WITH_BREACH"
    assert result.breaches[0].time == START
    assert result.breaches[0].phase == "pretrade"
    assert result.breaches[0].risk.position_failures == ("S",)
    assert result.breaches[0].risk.maximum_shortfall == 0  # Position risk, not a cash shortage.
    assert result.ledger.cash == 1000 and result.ledger.positions == {"S": 3}
    assert result.fills == () and called == []


def test_above_limit_target_is_visible_rejection_without_an_actual_breach_or_cash_change():
    stock_only = {"S": Instrument("S", "stock")}
    settings = Rules(HORIZON, ("S",), 0, maximum_position_quantities={"S": 2})
    result = Engine(stock_only, Opening("IMPORT", cash=1000, opening_equity=1000), settings,
                    UnnettedLiabilityStressMargin(), pricer()).run(
        [start_event(), Event("end", HORIZON, 0)],
        lambda view: {"S": 3} if view.time == START else None)
    assert result.status == "COMPLETED"
    assert result.breaches == [] and result.fills == ()
    assert result.ledger.cash == result.ledger.equity == 1000
    assert result.ledger.positions == {}
    rejected = [order for order in result.order_outcomes if order.outcome == "rejected"]
    assert len(rejected) == 1
    assert rejected[0].quantity == 3
    assert "position limit" in rejected[0].reason
