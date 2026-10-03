"""Mechanical target quantities, verified against signed hand calculations."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from basket_hedging.contracts import DataGap, Funding, Instrument, Opening, ShortLot
from basket_hedging.engine import Engine, PolicyView
from basket_hedging.market import Borrow, Event, Quote, Rules, ValuationPacket, readonly
from basket_hedging.policies import ScheduledDeltaPolicy, UnhedgedPolicy
from basket_hedging.risk import UnnettedLiabilityStressMargin


START = datetime(2026, 1, 1, 16, tzinfo=timezone.utc)
NEXT_OPEN = START + timedelta(hours=17)
NEXT_CLOSE = START + timedelta(days=1)
SECOND_OPEN = NEXT_OPEN + timedelta(days=1)
EXPIRY = START + timedelta(days=30)


@pytest.fixture
def instruments():
    return {
        "S": Instrument("S", "stock"),
        "T": Instrument("T", "stock"),
        "B": Instrument("B", "basket_put", multiplier=100, strike=100,
                        expiry=EXPIRY, coefficients=(("S", 1.0),)),
        "P": Instrument("P", "put", multiplier=50, strike=100,
                        expiry=EXPIRY, underlying="S"),
        "C": Instrument("C", "call", multiplier=20, strike=100,
                        expiry=EXPIRY, underlying="S"),
    }


def valuation(time, *, deltas=None):
    marks = {"S": 100, "T": 100, "B": 2, "P": 3, "C": 4}
    return ValuationPacket(time, marks, {"unchanged": marks.copy()},
                           {"B": {"S": -0.4}} if deltas is None else deltas,
                           {key: time for key in marks}, time)


def view(instruments, positions, *, deltas=None, universe=("S", "T"), terminal=False):
    return PolicyView("decision", START, 0, readonly(instruments), readonly(positions),
                      1200, 200, 0, 1000, valuation(START, deltas=deltas).policy_copy(),
                      (), (), universe, terminal, readonly({}))


@pytest.mark.parametrize("existing_stock", [0, -10, 20])
def test_short_put_delta_target_sign_multiplier_and_no_stock_double_count(instruments, existing_stock):
    policy = ScheduledDeltaPolicy()
    targets = policy.targets(view(instruments, {"B": -1, "S": existing_stock}))
    # Nonspot delta = (-1 contract)*(100 units/contract)*(-0.40) = +40 shares.
    assert targets == {"S": -40, "T": 0}


@pytest.mark.parametrize("kind,quantity,unit_delta,expected_target", [
    ("basket_call", -5, 0.5, 250),
    ("basket_put", -5, -0.5, -250),
    ("basket_call", 5, 0.5, -250),
    ("basket_put", 5, -0.5, 250),
])
def test_call_and_put_targets_offset_signed_exposure(instruments, kind, quantity, unit_delta, expected_target):
    instruments["B"] = replace(instruments["B"], kind=kind)
    state = view(instruments, {"B": quantity, "S": -200}, deltas={"B": {"S": unit_delta}})
    # Five contracts * 100 units * .5 delta gives 250 shares; existing stock
    # affects the subsequent order, never this absolute hedge target.
    assert ScheduledDeltaPolicy().targets(state) == {"S": expected_target, "T": 0}


def test_additional_put_and_call_positions_both_contribute_to_nonspot_delta(instruments):
    state = view(instruments, {"B": -1, "P": 2, "C": -3, "S": -10},
                 deltas={"B": {"S": -0.4}, "P": {"S": -0.2}, "C": {"S": 0.5}})
    # Basket +40; long puts 2*50*(-.2)=-20; short calls -3*20*.5=-30.
    # Total nonspot delta -10 means an absolute target of +10 shares.
    assert ScheduledDeltaPolicy().targets(state) == {"S": 10, "T": 0}


@pytest.mark.parametrize("deltas", [{}, {"B": {}}])
def test_missing_required_delta_requests_skipped_decision(instruments, deltas):
    with pytest.raises(DataGap, match="B.*S"):
        ScheduledDeltaPolicy().targets(view(instruments, {"B": -1}, deltas=deltas))


def test_hedge_universe_limits_targets_without_option_selection(instruments):
    state = view(instruments, {"B": -1, "P": 2, "S": -10, "T": 9},
                 deltas={"B": {"S": -0.4}, "P": {"S": -0.2}}, universe=("S", "P"))
    # The stock baseline explicitly closes option hedges, but counts their delta
    # until a real closing fill occurs. Omitted custom-policy targets still persist.
    assert ScheduledDeltaPolicy().targets(state) == {"S": -20, "P": 0}
    assert UnhedgedPolicy().targets(state) == {"S": 0, "P": 0}


def test_omitted_unhedgeable_constituent_does_not_require_irrelevant_delta(instruments):
    instruments["B"] = replace(instruments["B"], coefficients=(("S", 0.5), ("T", 0.5)))
    state = view(instruments, {"B": -1}, deltas={"B": {"S": -0.2}}, universe=("S",))
    assert ScheduledDeltaPolicy().targets(state) == {"S": -20}


def test_terminal_flag_cannot_override_unsettled_nonspot_risk(instruments):
    state = view(instruments, {"B": -1, "S": -10}, terminal=True)
    assert ScheduledDeltaPolicy().targets(state) == {"S": -40, "T": 0}
    settled = replace(state, positions=readonly({"S": -40}),
                      valuation=valuation(START, deltas={}).policy_copy())
    assert ScheduledDeltaPolicy().targets(settled) == {"S": 0, "T": 0}


def test_unhedged_needs_no_delta_and_never_targets_the_liability(instruments):
    state = view(instruments, {"B": -1, "P": 2, "S": -10}, deltas={})
    assert UnhedgedPolicy().targets(state) == {"S": 0, "T": 0}


def test_engine_subtracts_existing_stock_once_and_repeated_target_makes_no_order(instruments):
    opening = Opening("IMPORT", cash=6000, positions={"B": -1, "S": -10},
                      marks={"B": 2, "S": 100}, opening_equity=4800,
                      restricted_short_proceeds=1000,
                      short_lots=(ShortLot("opening-stock-short", "S", 10, 100, START),))
    rules = Rules(SECOND_OPEN, ("S",), 0, funding=Funding(borrow_rates={"S": 0}),
                  borrow={"S": Borrow(True, 100)})
    events = [
        Event("first-close", START, 0, valuation(START), decision=True,
              earliest_execution=NEXT_OPEN, order_deadline=NEXT_CLOSE),
        Event("first-open", NEXT_OPEN, 0, valuation(NEXT_OPEN), execution=True,
              quotes=(Quote("execution", "S", NEXT_OPEN, NEXT_OPEN, 100, 100, 100, 100),)),
        Event("second-close", NEXT_CLOSE, 0, valuation(NEXT_CLOSE), decision=True,
              earliest_execution=SECOND_OPEN, order_deadline=SECOND_OPEN + timedelta(hours=1)),
        Event("horizon", SECOND_OPEN, 0, valuation(SECOND_OPEN), execution=True,
              quotes=(Quote("last-execution", "S", SECOND_OPEN, SECOND_OPEN, 100, 100, 100, 100),)),
    ]
    result = Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()).run(
        events, ScheduledDeltaPolicy())
    assert [fill.quantity for fill in result.fills] == [-30]
    assert result.ledger.positions == {"B": -1, "S": -40}
    assert result.ledger.cash == 9000
    assert result.ledger.restricted_short_proceeds == 4000
    assert result.ledger.equity == 4800
    assert sum(outcome.outcome == "submitted" for outcome in result.order_outcomes) == 1
    assert result.snapshots[-1].residual_delta == {"S": 0, "T": 0}


def test_missing_delta_skips_decision_without_stopping_valid_account(instruments):
    opening = Opening("IMPORT", cash=1200, positions={"B": -1}, marks={"B": 2}, opening_equity=1000)
    rules = Rules(NEXT_OPEN, ("S",), 0)
    events = [Event("close", START, 0, valuation(START, deltas={}), decision=True,
                    earliest_execution=NEXT_OPEN, order_deadline=NEXT_CLOSE),
              Event("horizon", NEXT_OPEN, 0, valuation(NEXT_OPEN, deltas={}))]
    result = Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()).run(
        events, ScheduledDeltaPolicy())
    assert result.status == "INCOMPLETE_EXIT"
    assert result.decisions[0]["status"] == "skipped"
    assert result.data_gaps == []
    assert result.breaches == []
    assert result.fills == ()
    assert result.ledger.cash == 1200
