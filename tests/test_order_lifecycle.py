"""Persistent target orders and point-in-time policies, with hand-derived balances."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from basket_hedging.contracts import DataGap, Funding, Instrument, Opening, ShortLot
from basket_hedging.engine import Engine
from basket_hedging.market import Borrow, Event, Fixing, Observation, Quote, Rules, ValuationPacket
from basket_hedging.policies import ScheduledDeltaPolicy
from basket_hedging.risk import UnnettedLiabilityStressMargin


START = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
REVIEW = START + timedelta(hours=1)
EXECUTE = START + timedelta(hours=2)
HORIZON = START + timedelta(hours=3)
OPTION_EXPIRY = START + timedelta(days=30)


def packet(time, marks, *, deltas=None):
    return ValuationPacket(time, marks, {"unchanged": marks}, deltas or {},
                           {key: time for key in marks}, time)


def quote(time, key="S", *, bid=100, ask=100):
    return Quote(f"quote/{key}/{time.isoformat()}", key, time, time,
                 bid, ask, 1000, 1000)


def market_event(identifier, time, *, marks=None, decision=False, quotes=(), deltas=None):
    marks = {"S": 100} if marks is None else marks
    return Event(identifier, time, 0, packet(time, marks, deltas=deltas),
                 observations=tuple(Observation(f"{identifier}/{key}", key, value, time, time)
                                    for key, value in marks.items()),
                 quotes=quotes, execution=bool(quotes), decision=decision,
                 earliest_execution=EXECUTE if decision else None,
                 order_deadline=HORIZON if decision else None)


def stock_book():
    # Opening equity: 10,000 cash + 20 shares at 100 = 12,000.
    return Opening("IMPORT", cash=10000, positions={"S": 20},
                   marks={"S": 100}, opening_equity=12000)


def stock_events():
    return [market_event("start", START, decision=True),
            market_event("review", REVIEW, decision=True),
            market_event("fill", EXECUTE, quotes=(quote(EXECUTE),)),
            market_event("end", HORIZON)]


def run_stock(policy, events=None, *, opening=None, rules=None):
    return Engine({"S": Instrument("S", "stock")}, opening or stock_book(),
                  rules or Rules(HORIZON, ("S",), 0),
                  UnnettedLiabilityStressMargin()).run(events or stock_events(), policy)


def submitted(result):
    return [outcome for outcome in result.order_outcomes if outcome.outcome == "submitted"]


def test_repeated_target_retains_pending_eighty_and_does_not_duplicate_it():
    views = []

    def target_hundred(view):
        views.append(view)
        return {"S": 100}

    result = run_stock(target_hundred)
    review = next(view for view in views if view.event_id == "review")
    assert review.positions["S"] == 20
    assert [order.quantity for order in review.pending_orders] == [80]
    assert [(order.event_id, order.quantity) for order in submitted(result)] == [("start", 80)]
    assert [fill.quantity for fill in result.fills] == [80]
    assert result.ledger.positions == {"S": 100}
    assert result.ledger.cash == 2000  # 10,000 - 80*100.
    assert result.ledger.equity == 12000
    assert not [outcome for outcome in result.order_outcomes if outcome.outcome == "cancelled"]


def test_smaller_target_cancels_eighty_and_replaces_with_thirty():
    def changing_target(view):
        return {"S": 100 if view.time == START else 50}

    result = run_stock(changing_target)
    assert [(order.event_id, order.quantity) for order in submitted(result)] == [
        ("start", 80), ("review", 30)]
    cancellations = [outcome for outcome in result.order_outcomes if outcome.outcome == "cancelled"]
    assert [(outcome.event_id, outcome.quantity) for outcome in cancellations] == [("review", 80)]
    assert [fill.quantity for fill in result.fills] == [30]
    assert result.ledger.positions == {"S": 50}
    assert result.ledger.cash == 7000  # 10,000 - 30*100.
    assert result.ledger.equity == 12000


def test_larger_target_keeps_compatible_eighty_and_submits_only_fifty_more():
    opening = Opening("IMPORT", cash=20000, positions={"S": 20},
                      marks={"S": 100}, opening_equity=22000)

    def increasing_target(view):
        return {"S": 100 if view.time == START else 150}

    result = run_stock(increasing_target, opening=opening)
    assert [(order.event_id, order.quantity) for order in submitted(result)] == [
        ("start", 80), ("review", 50)]
    assert [fill.quantity for fill in result.fills] == [80, 50]
    assert not [order for order in result.order_outcomes if order.outcome == "cancelled"]
    assert result.ledger.positions == {"S": 150}
    assert result.ledger.cash == 7000  # 20,000 - (80+50)*100.
    assert result.ledger.equity == 22000


def test_two_eligible_pending_orders_share_one_quotes_capacity():
    events = stock_events()
    events[2] = replace(events[2], quotes=(
        Quote("seven-shares", "S", EXECUTE, EXECUTE, 100, 100, 7, 7),))

    def increasing_target(view):
        return {"S": 5 if view.event_id == "start" else 10} if view.decision_event else None

    result = run_stock(increasing_target, events,
                       opening=Opening("IMPORT", cash=2000, opening_equity=2000))
    # Both five-share buys exist and are eligible before the same quote arrives.
    assert [(order.event_id, order.quantity) for order in submitted(result)] == [
        ("start", 5), ("review", 5)]
    assert [(fill.time, fill.quantity) for fill in result.fills] == [(EXECUTE, 5)]
    rejected = [order for order in result.order_outcomes if order.outcome == "rejected"]
    assert len(rejected) == 1
    assert rejected[0].time == EXECUTE and rejected[0].quantity == 5
    assert "size" in rejected[0].reason
    assert result.ledger.positions == {"S": 5}
    assert result.ledger.cash == 1500  # 2,000 - 5 * 100; only two quoted shares remain.
    assert result.ledger.equity == 2000
    assert result.breaches == []


@pytest.mark.parametrize("skip", ["none", "missing_inputs", "omitted_target"])
def test_skipping_or_omitting_target_keeps_an_existing_valid_order(skip):
    def decide(view):
        if view.event_id == "start":
            return {"S": 100}
        if skip == "missing_inputs":
            raise DataGap("test policy sensitivity unavailable")
        return {} if skip == "omitted_target" else None

    result = run_stock(decide)
    assert len(submitted(result)) == 1
    assert [fill.quantity for fill in result.fills] == [80]
    assert result.ledger.cash == 2000
    assert result.data_gaps == []
    assert result.breaches == []
    assert not [outcome for outcome in result.order_outcomes if outcome.outcome == "cancelled"]
    review = next(record for record in result.decisions if record["event_id"] == "review")
    if skip == "missing_inputs":
        assert review["status"] == "skipped"
        assert "sensitivity" in review["reason"]
    elif skip == "omitted_target":
        assert review["targets"] == {}
    else:
        assert review["status"] != "targets"


def test_missing_delta_on_later_review_preserves_original_stock_hedge():
    instruments = {"S": Instrument("S", "stock"),
                   "B": Instrument("B", "basket_put", multiplier=100, strike=100,
                                   expiry=OPTION_EXPIRY, coefficients=(("S", 1),))}
    opening = Opening("IMPORT", cash=6000, positions={"B": -1, "S": -10},
                      marks={"B": 2, "S": 100}, opening_equity=4800,
                      restricted_short_proceeds=1000,
                      short_lots=(ShortLot("existing-short", "S", 10, 100, START),))
    marks = {"S": 100, "B": 2}
    events = [market_event("start", START, marks=marks, decision=True,
                           deltas={"B": {"S": -0.4}}),
              market_event("review", REVIEW, marks=marks, decision=True),
              market_event("fill", EXECUTE, marks=marks, quotes=(quote(EXECUTE),)),
              market_event("end", HORIZON, marks=marks)]
    rules = Rules(HORIZON, ("S",), 0, funding=Funding(borrow_rates={"S": 0}),
                  borrow={"S": Borrow(True, 100)})
    result = Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()).run(
        events, ScheduledDeltaPolicy())
    # -(-1 * 100 * -.4) = -40 desired shares, less -10 held => sell 30.
    assert [(order.event_id, order.quantity) for order in submitted(result)] == [("start", -30)]
    assert [fill.quantity for fill in result.fills] == [-30]
    assert result.ledger.cash == 9000
    assert result.ledger.restricted_short_proceeds == 4000
    assert result.ledger.equity == 4800
    assert result.data_gaps == []
    assert any(record["event_id"] == "review" and record["status"] == "skipped"
               for record in result.decisions)


def test_zero_delay_still_cannot_fill_from_the_quote_that_triggered_submission():
    first = market_event("start", START, quotes=(quote(START, bid=99, ask=101),))
    repeated = replace(first, id="same-time", sequence=1, observations=())
    later = market_event("later", EXECUTE, marks={"S": 110},
                         quotes=(quote(EXECUTE, bid=109, ask=111),))
    final = market_event("end", HORIZON, marks={"S": 110})
    opening = Opening("IMPORT", cash=1000, opening_equity=1000)
    rules = Rules(HORIZON, ("S",), 0, execution_delay_seconds=0)
    result = run_stock(lambda view: {"S": 1}, [first, repeated, later, final],
                       opening=opening, rules=rules)
    assert [(fill.time, fill.price, fill.quantity) for fill in result.fills] == [(EXECUTE, 111, 1)]
    assert result.ledger.cash == 889
    assert result.ledger.equity == 999


def option_instruments():
    return {"S": Instrument("S", "stock"),
            "B": Instrument("B", "basket_put", multiplier=100, strike=100,
                            expiry=OPTION_EXPIRY, coefficients=(("S", 1),)),
            "P": Instrument("P", "put", multiplier=50, strike=100,
                            expiry=OPTION_EXPIRY, underlying="S"),
            "C": Instrument("C", "call", multiplier=20, strike=100,
                            expiry=OPTION_EXPIRY, underlying="S")}


def test_callable_policy_can_trade_constituent_puts_and_calls_but_not_the_basket():
    marks = {"S": 100, "B": 2, "P": 3, "C": 4}
    events = [market_event("start", START, marks=marks, decision=True),
              market_event("fill", EXECUTE, marks=marks,
                           quotes=(quote(EXECUTE, "P", bid=3, ask=3),
                                   quote(EXECUTE, "C", bid=4, ask=4))),
              market_event("end", HORIZON, marks=marks)]

    def options(view):
        return {"P": 1, "C": -1, "B": 1} if view.event_id == "start" else None

    result = Engine(option_instruments(), Opening("IMPORT", cash=10000, opening_equity=10000),
                    Rules(HORIZON, ("S", "P", "C"), 0, lot_sizes={"P": 1, "C": 1}),
                    UnnettedLiabilityStressMargin()).run(events, options)
    assert [(fill.instrument_id, fill.quantity) for fill in result.fills] == [("C", -1), ("P", 1)]
    assert result.ledger.cash == 9930  # 10,000 + 1*20*4 - 1*50*3.
    assert result.ledger.positions == {"C": -1, "P": 1}
    assert result.ledger.equity == 10000
    assert any(order.instrument_id == "B" and order.outcome == "rejected"
               for order in result.order_outcomes)
    assert "B" not in result.ledger.positions


def test_invalid_option_lot_target_does_not_cancel_a_valid_pending_option_order():
    marks = {"S": 100, "P": 3}
    events = [market_event("start", START, marks=marks, decision=True),
              market_event("review", REVIEW, marks=marks, decision=True),
              market_event("fill", EXECUTE, marks=marks, quotes=(quote(EXECUTE, "P", bid=3, ask=3),)),
              market_event("end", HORIZON, marks=marks)]

    def options(view):
        return {"P": 1 if view.event_id == "start" else 0.5} if view.decision_event else None

    result = Engine(option_instruments(), Opening("IMPORT", cash=1000, opening_equity=1000),
                    Rules(HORIZON, ("P",), 0, lot_sizes={"P": 1}),
                    UnnettedLiabilityStressMargin()).run(events, options)
    assert [fill.quantity for fill in result.fills] == [1]
    assert result.ledger.cash == 850
    assert result.ledger.equity == 1000
    assert any(order.event_id == "review" and order.outcome == "rejected"
               and "lot" in order.reason for order in result.order_outcomes)


def test_policy_views_expose_immutable_pending_orders_and_current_released_market():
    seen = []

    def inspect(view):
        seen.append(view)
        assert all(observation.available_at <= view.time for observation in view.history)
        assert not hasattr(view, "events") and not hasattr(view, "rng")
        with pytest.raises(TypeError):
            view.market["S"] = None
        with pytest.raises(TypeError):
            view.quotes["S"] = None
        with pytest.raises(TypeError):
            view.positions["S"] = 0
        if view.pending_orders:
            assert isinstance(view.pending_orders, tuple)
            with pytest.raises(FrozenInstanceError):
                view.pending_orders[0].quantity = 0
        return {"S": 100}

    result = run_stock(inspect)
    review = next(view for view in seen if view.event_id == "review")
    assert review.market["S"].id == "review/S"
    assert review.market["S"].observed_at == REVIEW
    assert review.positions["S"] == 20
    assert review.pending_orders[0].quantity == 80
    # Later fills do not mutate the earlier detached view.
    assert result.ledger.positions["S"] == 100
    assert review.positions["S"] == 20
    assert review.pending_orders[0].quantity == 80


def test_partial_and_delayed_releases_preserve_each_instruments_observation_time():
    earlier = START - timedelta(minutes=1)
    views = []
    initial = Event("start", START, 0, packet(START, {"S": 100, "T": 80}),
                    observations=(Observation("S-initial", "S", 100, START, START),
                                  Observation("T-initial", "T", 80, START, START)),
                    quotes=(quote(START),))
    partial = Event("partial", REVIEW, 0, packet(REVIEW, {"S": 101, "T": 80}),
                    observations=(Observation("S-new", "S", 101, REVIEW, REVIEW),
                                  Observation("T-delayed-old", "T", 79, earlier, REVIEW)),
                    quotes=(quote(REVIEW, bid=100, ask=102),))
    old_quote = Quote("delayed-old-quote", "S", START, HORIZON, 90, 92, 100, 100)
    final = Event("end", HORIZON, 0, packet(HORIZON, {"S": 101, "T": 80}),
                  quotes=(old_quote,))

    def observe(view):
        views.append(view)
        return None

    result = Engine({key: Instrument(key, "stock") for key in ("S", "T")},
                    Opening("IMPORT", cash=1000, opening_equity=1000),
                    Rules(HORIZON, ("S", "T"), 0), UnnettedLiabilityStressMargin()).run(
        [initial, partial, final], observe)
    assert result.status == "COMPLETED"
    partial_view = next(view for view in views if view.event_id == "partial")
    assert partial_view.time == REVIEW and partial_view.time.tzinfo is not None
    assert partial_view.market["S"].observed_at == REVIEW
    assert partial_view.market["T"].observed_at == START
    assert partial_view.market["T"].available_at == START
    assert partial_view.market["T"].value == 80
    assert [record.id for record in partial_view.history] == [
        "S-initial", "T-initial", "S-new", "T-delayed-old"]
    assert views[-1].quotes["S"].time == REVIEW
    assert views[-1].quotes["S"].bid == 100
    with pytest.raises(FrozenInstanceError):
        partial_view.market["T"].observed_at = REVIEW


def test_future_observation_mutation_preserves_pending_orders_decisions_and_ledger_prefix():
    results = []
    for final_price in (90, 130):
        events = stock_events()
        events[-1] = market_event("end", HORIZON, marks={"S": final_price})
        results.append(run_stock(lambda view: {"S": view.market["S"].value}, events))
    first, second = results
    assert [snapshot for snapshot in first.snapshots if snapshot.time <= EXECUTE] == [
        snapshot for snapshot in second.snapshots if snapshot.time <= EXECUTE]
    assert [post for post in first.ledger.postings if post.time <= EXECUTE] == [
        post for post in second.ledger.postings if post.time <= EXECUTE]
    assert [order for order in first.order_outcomes if order.time <= EXECUTE] == [
        order for order in second.order_outcomes if order.time <= EXECUTE]
    assert [decision for decision in first.decisions if decision["time"] <= EXECUTE] == [
        decision for decision in second.decisions if decision["time"] <= EXECUTE]
    assert first.ledger.cash == second.ledger.cash == 2000
    assert first.ledger.equity == 11000  # 2,000 cash + 100 shares at 90.
    assert second.ledger.equity == 15000  # 2,000 cash + 100 shares at 130.


def test_basket_settlement_cancels_opening_orders_and_unwinds_stock_and_live_option_later():
    instruments = option_instruments()
    instruments["B"] = replace(instruments["B"], expiry=EXECUTE)
    opening = Opening("IMPORT", cash=10000, positions={"B": -1, "S": 2, "P": 1},
                      marks={"B": 2, "S": 100, "P": 3}, opening_equity=10150)
    first = market_event("start", START, marks={"B": 2, "S": 100, "P": 3}, decision=True)
    first = replace(first, earliest_execution=HORIZON, order_deadline=HORIZON + timedelta(hours=1))
    expiry = market_event("expiry", EXECUTE, marks={"S": 90, "P": 10},
                          quotes=(quote(EXECUTE, bid=89, ask=91), quote(EXECUTE, "P", bid=10, ask=10)))
    expiry = replace(expiry, fixings=(Fixing("official-basket", "B", EXECUTE, EXECUTE, {"S": 90}),))
    exit_event = market_event("exit", HORIZON, marks={"S": 90, "P": 10},
                              quotes=(quote(HORIZON, bid=89, ask=91), quote(HORIZON, "P", bid=10, ask=10)))

    def always_open_more(view):
        return {"S": 5, "P": 2}

    result = Engine(instruments, opening, Rules(HORIZON, ("S", "P"), 0),
                    UnnettedLiabilityStressMargin()).run([first, expiry, exit_event], always_open_more)
    assert result.status == "COMPLETED"
    assert [(fill.time, fill.instrument_id, fill.quantity) for fill in result.fills] == [
        (HORIZON, "P", -1), (HORIZON, "S", -2)]
    assert result.ledger.positions == {}
    assert result.ledger.cash == 9678  # 10,000 - 100*(100-90) + 50*10 + 2*89.
    assert result.ledger.equity == 9678
    assert result.ledger.components()["net_pnl"] == -472
    assert result.ledger.posted_margin_cash == result.ledger.restricted_short_proceeds == 0
    assert [record["record"].instrument_id for record in result.settlement_outcomes] == ["B"]
    assert sorted(order.quantity for order in result.order_outcomes
                  if order.event_id == "expiry" and order.outcome == "cancelled") == [1, 3]
