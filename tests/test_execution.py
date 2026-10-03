"""Causal replay and execution checks with hand-authored prices and balances."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from basket_hedging.contracts import Funding, InputError, Instrument, Opening, ShortLot
from basket_hedging.engine import Engine
from basket_hedging.market import Borrow, Event, Fixing, Observation, Quote, Rules, ValuationPacket
from basket_hedging.risk import UnnettedLiabilityStressMargin


START = datetime(2026, 1, 1, 16, tzinfo=timezone.utc)
OPEN = START + timedelta(hours=17)
NEXT_CLOSE = START + timedelta(days=1)


def packet(time, marks, *, deltas=None, metadata=None):
    return ValuationPacket(
        time, marks, {"unchanged": marks.copy()}, deltas or {},
        {key: time for key in marks}, time, (), metadata or {},
    )


def quote(time, instrument_id="S", *, bid=100, ask=100, size=100,
          quote_id=None, available_at=None):
    return Quote(quote_id or f"quote-{instrument_id}-{time.isoformat()}", instrument_id,
                 time, available_at or time, bid, ask, size, size)


def test_policy_valuation_copy_freezes_nested_state_and_detaches_source():
    marks = {"S": 100, "B": 2}
    deltas = {"B": {"S": -0.4}}
    metadata = {"parameters": {"volatility": [0.2]}}
    source = packet(START, marks, deltas=deltas, metadata=metadata)
    view = source.policy_copy()
    with pytest.raises(TypeError):
        view.marks["S"] = 90
    with pytest.raises(TypeError):
        view.stress_marks["unchanged"]["B"] = 0
    with pytest.raises(TypeError):
        view.deltas["B"]["S"] = 99
    with pytest.raises(TypeError):
        view.metadata["parameters"]["volatility"][0] = 99
    marks["S"] = 200
    deltas["B"]["S"] = 0
    metadata["parameters"]["volatility"][0] = 0.5
    assert view.marks["S"] == 100
    assert view.deltas["B"]["S"] == -0.4
    assert view.metadata["parameters"]["volatility"] == (0.2,)


@pytest.mark.parametrize("record_field,record", [
    ("observations", Observation("future-spot", "S", 100, START, OPEN)),
    ("quotes", quote(START, available_at=OPEN)),
    ("fixings", Fixing("future-fixing", "B", START, OPEN, {"S": 100})),
])
def test_event_rejects_records_that_have_not_been_released(record_field, record):
    with pytest.raises(InputError, match="unreleased"):
        Event("event", START, 0, packet(START, {"S": 100}), **{record_field: (record,)})


@pytest.mark.parametrize("bid,ask,size", [(101, 100, 1), (99, 100, -1),
                                          (float("nan"), 100, 1)])
def test_malformed_execution_quotes_are_input_errors(bid, ask, size):
    with pytest.raises(InputError):
        quote(START, bid=bid, ask=ask, size=size)


def test_input_timestamps_and_decision_schedule_cannot_look_forward():
    with pytest.raises(InputError):
        ValuationPacket(START, {"S": 100}, {"unchanged": {"S": 100}}, input_cutoff=OPEN)
    with pytest.raises(InputError):
        ValuationPacket(START, {"S": 100}, {"unchanged": {"S": 100}}, observed_at={"S": OPEN})
    with pytest.raises(InputError):
        Event("event", START, 0, packet(START, {"S": 100}), decision=True,
              earliest_execution=START, order_deadline=NEXT_CLOSE)
    with pytest.raises(InputError):
        Observation("backward-release", "S", 100, OPEN, START)


class ZeroMargin:
    """Test-only requirement to isolate funding/execution; never a demo rule."""

    label = "test-only zero requirement"

    def required_margin(self, ledger, valuation):
        return 0.0


class ScriptedTargets:
    """An explicit test script, not an additional production hedge policy."""

    name = "scripted-accounting-test"

    def __init__(self, targets):
        self.script = targets
        self.views = []

    def targets(self, view):
        self.views.append(view)
        return self.script.get(view.event_id, {})


def stocks(*ids):
    return {key: Instrument(key, "stock") for key in ids}


def run_cash(events, targets, *, cash=1000, instruments=None, rules=None):
    instruments = instruments or stocks("S")
    rules = rules or Rules(events[-1].time, tuple(instruments), 0)
    policy = ScriptedTargets(targets)
    result = Engine(instruments, Opening("IMPORT", cash=cash, opening_equity=cash),
                    rules, ZeroMargin()).run(iter(events), policy)
    return result, policy


def decision(time=START, *, event_id="close", marks=None, deadline=NEXT_CLOSE,
             earliest=OPEN, terminal=False, **kwargs):
    return Event(event_id, time, 0, packet(time, marks or {"S": 100}),
                 decision=True, earliest_execution=earliest, order_deadline=deadline,
                 terminal_decision=terminal, **kwargs)


@pytest.mark.parametrize("constraint", ["quote_size", "maximum_quantity", "lot"])
def test_quantity_constraints_use_share_units_for_material_fractional_fills(constraint):
    price = 1e10
    events = [decision(marks={"S": price}),
              Event("open", OPEN, 0, packet(OPEN, {"S": price}), execution=True,
                    quotes=(quote(OPEN, bid=price, ask=price,
                                  size=0 if constraint == "quote_size" else 1),))]
    rules = Rules(OPEN, ("S",), 0,
                  maximum_order_quantities={"S": 0} if constraint == "maximum_quantity" else {},
                  lot_sizes={"S": 1} if constraint == "lot" else {})
    result, _ = run_cash(events, {"close": {"S": 5e-9}}, rules=rules)
    assert not result.fills
    assert result.ledger.cash == 1000
    assert result.ledger.positions == {}
    assert any(o.outcome == "rejected" for o in result.order_outcomes)


def test_close_decision_fills_only_at_later_open_and_costs_are_once():
    events = [
        decision(execution=True, quotes=(quote(START, bid=99, ask=101),),
                 observations=(Observation("close-spot", "S", 100, START, START),)),
        Event("open", OPEN, 0, packet(OPEN, {"S": 110}), execution=True,
              quotes=(quote(OPEN, bid=109, ask=111),)),
    ]
    rules = Rules(OPEN, ("S",), 0, fee_per_order=2, slippage_per_unit=0.5)
    result, policy = run_cash(events, {"close": {"S": 2}}, rules=rules)
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert (fill.time, fill.quantity, fill.price) == (OPEN, 2, 111.5)
    assert result.ledger.cash == 775  # 1,000 - 2*111.5 - 2.
    assert result.ledger.equity == 995
    assert result.ledger.components()["execution_cost"] == 3
    assert result.ledger.components()["fees"] == 2
    assert result.status == "INCOMPLETE_EXIT"
    assert policy.views[0].history[0].value == 100


@pytest.mark.parametrize("quote_kind", ["missing", "stale", "delayed"])
def test_unusable_quote_stays_pending_until_exclusive_deadline(quote_kind):
    quotes = {
        "missing": (),
        "stale": (quote(START),),
        "delayed": (quote(START, available_at=OPEN),),
    }[quote_kind]
    events = [decision(), Event("open", OPEN, 0, packet(OPEN, {"S": 110}),
                               execution=True, quotes=quotes),
              Event("deadline", NEXT_CLOSE, 0, packet(NEXT_CLOSE, {"S": 120}),
                    execution=True, quotes=(quote(NEXT_CLOSE, bid=120, ask=120),))]
    result, _ = run_cash(events, {"close": {"S": 1}})
    assert result.fills == ()
    assert result.ledger.cash == 1000
    assert [(x.event_id, x.outcome) for x in result.order_outcomes] == [
        ("close", "submitted"), ("open", "pending"), ("deadline", "expired")]


def test_terminal_deadline_includes_horizon_execution():
    opening = Opening("IMPORT", cash=100, positions={"S": 1}, marks={"S": 100}, opening_equity=200)
    events = [decision(deadline=OPEN, terminal=True),
              Event("exit", OPEN, 0, packet(OPEN, {"S": 110}), execution=True,
                    quotes=(quote(OPEN, bid=109, ask=111),))]
    result = Engine(stocks("S"), opening, Rules(OPEN, ("S",), 0), ZeroMargin()).run(
        events, ScriptedTargets({"close": {"S": 0}}))
    assert result.status == "COMPLETED"
    assert result.ledger.cash == 209
    assert result.ledger.positions == {}
    assert result.ledger.posted_margin_cash == result.ledger.restricted_short_proceeds == 0
    assert result.ledger.components()["net_pnl"] == 9


def test_earlier_sequential_fill_survives_later_infeasible_order():
    marks = {"A": 100, "Z": 100}
    events = [decision(marks=marks),
              Event("open", OPEN, 0, packet(OPEN, marks), execution=True,
                    quotes=(quote(OPEN, "Z"), quote(OPEN, "A")))]
    rules = Rules(OPEN, ("A", "Z"), 0, fee_per_order=1)
    result, _ = run_cash(events, {"close": {"Z": 1, "A": 1}}, cash=150,
                         instruments=stocks("A", "Z"), rules=rules)
    assert [(p.instrument_id, p.quantity) for p in result.fills] == [("A", 1)]
    assert result.ledger.cash == 49
    assert result.ledger.equity == 149
    assert result.ledger.positions == {"A": 1}
    assert result.ledger.components()["fees"] == 1
    rejected = [x for x in result.order_outcomes if x.outcome == "rejected"]
    assert len(rejected) == 1 and rejected[0].instrument_id == "Z"
    assert rejected[0].projected_risk.cash == -52
    assert result.breaches == []
    assert not any(p.instrument_id == "Z" for p in result.ledger.postings)


@pytest.mark.parametrize("size,lot,borrow,reason", [
    (1, 0, True, "size"), (100, 3, True, "lot"), (100, 0, False, "infeasible"),
])
def test_whole_order_rejections_do_not_charge_fees_or_change_the_book(size, lot, borrow, reason):
    events = [decision(), Event("open", OPEN, 0, packet(OPEN, {"S": 100}),
                               execution=True, quotes=(quote(OPEN, size=size),))]
    rules = Rules(OPEN, ("S",), 0, funding=Funding(borrow_rates={"S": 0}),
                  borrow={"S": Borrow(borrow, 100)}, fee_per_order=7, lot_sizes={"S": lot})
    result, _ = run_cash(events, {"close": {"S": -2}}, rules=rules)
    assert result.ledger.cash == result.ledger.equity == 1000
    assert result.ledger.positions == {}
    assert result.ledger.short_lots == []
    assert result.ledger.components()["fees"] == 0
    assert result.fills == () and result.breaches == []
    rejected = [outcome for outcome in result.order_outcomes if outcome.outcome == "rejected"]
    assert len(rejected) == 1
    assert reason in rejected[0].reason


def test_duplicate_event_and_quote_do_not_repeat_a_fill_or_capacity():
    released_quote = quote(OPEN, size=2)
    close = decision()
    execution = Event("open", OPEN, 0, packet(OPEN, {"S": 100}), execution=True,
                      quotes=(released_quote, released_quote))
    repeated_release = Event("open-again", OPEN, 1, packet(OPEN, {"S": 100}),
                             execution=True, quotes=(released_quote,))
    horizon = Event("end", NEXT_CLOSE, 0, packet(NEXT_CLOSE, {"S": 100}))
    engine = Engine(stocks("S"), Opening("IMPORT", cash=1000, opening_equity=1000),
                    Rules(NEXT_CLOSE, ("S",), 0), ZeroMargin())
    result = engine.run([close, execution, execution, repeated_release, horizon],
                        ScriptedTargets({"close": {"S": 2}}))
    assert len(result.fills) == 1
    assert result.ledger.cash == 800
    assert engine._quote_used == {(released_quote.id, "ask"): 2}


def test_conflicting_duplicate_released_quote_is_a_malformed_input():
    same_id = "reused"
    events = [decision(quotes=(quote(START, quote_id=same_id),)),
              Event("open", OPEN, 0, packet(OPEN, {"S": 110}), execution=True,
                    quotes=(quote(OPEN, bid=110, ask=110, quote_id=same_id),))]
    with pytest.raises(InputError, match="conflicting duplicate quote"):
        run_cash(events, {"close": {"S": 1}})


def test_future_changes_leave_prefix_financial_records_and_decisions_unchanged():
    close = decision(observations=(Observation("close-spot", "S", 100, START, START),))
    opened = Event("open", OPEN, 0, packet(OPEN, {"S": 110}), execution=True,
                   observations=(Observation("open-spot", "S", 110, OPEN, OPEN),),
                   quotes=(quote(OPEN, bid=109, ask=111),))
    results = []
    for later_price in (90, 150):
        horizon = Event("later", NEXT_CLOSE, 0, packet(NEXT_CLOSE, {"S": later_price}),
                        observations=(Observation("future", "S", later_price, NEXT_CLOSE, NEXT_CLOSE),))
        results.append(run_cash([close, opened, horizon], {"close": {"S": 1}})[0])
    left, right = results
    assert [x for x in left.snapshots if x.time <= OPEN] == [x for x in right.snapshots if x.time <= OPEN]
    assert [x for x in left.ledger.postings if x.time <= OPEN] == [x for x in right.ledger.postings if x.time <= OPEN]
    assert left.order_outcomes == right.order_outcomes
    assert left.decisions == right.decisions
    assert left.ledger.equity == 979
    assert right.ledger.equity == 1039


def option_instruments():
    return {"S": Instrument("S", "stock"),
            "P": Instrument("P", "put", multiplier=50, strike=105, expiry=OPEN, underlying="S"),
            "C": Instrument("C", "call", multiplier=50, strike=95, expiry=OPEN, underlying="S")}


def test_missing_one_fixing_leaves_entire_mandatory_phase_unposted():
    opening = Opening("IMPORT", cash=100, positions={"P": 1, "C": -1},
                      marks={"P": 1, "C": 1}, opening_equity=100)
    events = [Event("inception", START, 0, packet(START, {"P": 1, "C": 1})),
              Event("expiry", OPEN, 0, packet(OPEN, {"S": 100}),
                    fixings=(Fixing("fix-P", "P", OPEN, OPEN, {"S": 100}),))]
    rules = Rules(OPEN, ("S",), 0, funding=Funding(cash_rate=0.365))
    result = Engine(option_instruments(), opening, rules, ZeroMargin()).run(events, ScriptedTargets({}))
    assert result.status == "STOP_DATA_GAP"
    assert result.ledger.cash == 100
    assert result.ledger.positions == {"P": 1, "C": -1}
    assert result.ledger.settlements == {}
    assert result.settlement_outcomes == []
    assert all(post.time == START for post in result.ledger.postings)
    assert result.last_valid_snapshot.time == START
    assert result.snapshots[-1].risk.equity is None


def test_committed_settlement_survives_subsequent_valuation_gap_in_report():
    opening = Opening("IMPORT", cash=100, positions={"P": 1}, marks={"P": 1}, opening_equity=150)
    events = [Event("inception", START, 0, packet(START, {"P": 1})),
              Event("expiry", OPEN, 0, None,
                    fixings=(Fixing("fix-P", "P", OPEN, OPEN, {"S": 100}),))]
    result = Engine(option_instruments(), opening, Rules(OPEN, ("S",), 0), ZeroMargin()).run(
        events, ScriptedTargets({}))
    assert result.status == "STOP_DATA_GAP"
    assert result.ledger.cash == 350
    assert result.ledger.positions == {}
    assert len(result.settlement_outcomes) == 1
    assert result.settlement_outcomes[0]["record"].cashflow == 250
    assert result.settlement_outcomes[0]["funded"]
    assert result.last_valid_snapshot.risk.equity == 150


def test_same_event_contractual_cashflows_net_before_risk_and_settle_once():
    opening = Opening("IMPORT", cash=100, positions={"P": 1, "C": -1},
                      marks={"P": 1, "C": 1}, opening_equity=100)
    fixings = (Fixing("fix-P", "P", OPEN, OPEN, {"S": 100}),
               Fixing("fix-C", "C", OPEN, OPEN, {"S": 100}))
    events = [Event("inception", START, 0, packet(START, {"P": 1, "C": 1})),
              Event("expiry", OPEN, 0, packet(OPEN, {"S": 100}), fixings=fixings),
              Event("duplicate-release", OPEN, 1, packet(OPEN, {"S": 100}), fixings=fixings),
              Event("horizon", NEXT_CLOSE, 0, packet(NEXT_CLOSE, {"S": 100}))]
    result = Engine(option_instruments(), opening, Rules(NEXT_CLOSE, ("S",), 0), ZeroMargin()).run(
        events, ScriptedTargets({}))
    # Alphabetical booking order pays C's 250 first, but declared atomic net cash is 100.
    assert result.status == "COMPLETED"
    assert result.ledger.cash == 100
    assert result.ledger.positions == {}
    assert result.breaches == []
    assert len(result.settlement_outcomes) == 2
    assert all(record["funded"] for record in result.settlement_outcomes)
    assert sorted(post.cash_change for post in result.ledger.postings if post.kind == "settlement") == [-250, 250]


def test_unfunded_settlement_is_booked_then_recorded_as_actual_breach():
    opening = Opening("IMPORT", cash=100, positions={"P": -1}, marks={"P": 1}, opening_equity=50)
    events = [Event("inception", START, 0, packet(START, {"P": 1})),
              Event("expiry", OPEN, 0, packet(OPEN, {"S": 100}),
                    fixings=(Fixing("fix-P", "P", OPEN, OPEN, {"S": 100}),))]
    result = Engine(option_instruments(), opening, Rules(OPEN, ("S",), 0), ZeroMargin()).run(
        events, ScriptedTargets({}))
    assert result.status == "STOP_WITH_BREACH"
    assert result.ledger.cash == -150
    assert result.ledger.positions == {}
    assert result.settlement_outcomes[0]["record"].cashflow == -250
    assert not result.settlement_outcomes[0]["funded"]
    assert result.breaches[0].phase == "pretrade"
    assert result.breaches[0].risk.equity_deficit == 150
    assert result.breaches[0].risk.unencumbered_cash_deficit == 150


@pytest.mark.parametrize("put_fixing", [90, 80])
@pytest.mark.parametrize("reverse_records", [False, True])
def test_shared_underlying_fixings_agree_before_settlement(put_fixing, reverse_records):
    instruments = {
        "S": Instrument("S", "stock"),
        "B": Instrument("B", "basket_put", multiplier=100, strike=100,
                        expiry=OPEN, coefficients=(("S", 1),)),
        "P": Instrument("P", "put", multiplier=100, strike=100,
                        expiry=OPEN, underlying="S"),
    }
    opening = Opening("IMPORT", cash=1000, positions={"B": -1, "P": 1},
                      marks={"B": 2, "P": 2}, opening_equity=1000)
    fixings = (Fixing("basket-fix", "B", OPEN, OPEN, {"S": 90}),
               Fixing("put-fix", "P", OPEN, OPEN, {"S": put_fixing}))
    if reverse_records:
        fixings = tuple(reversed(fixings))
    events = [Event("start", START, 0, packet(START, {"B": 2, "P": 2})),
              Event("expiry", OPEN, 0, packet(OPEN, {"S": 90}), fixings=fixings)]
    engine = Engine(instruments, opening, Rules(OPEN, (), 0), UnnettedLiabilityStressMargin())
    if put_fixing != 90:
        with pytest.raises(InputError, match="conflicting official fixing.*S"):
            engine.run(events, ScriptedTargets({}))
        # Neither settlement, nor any other due posting, may be committed.
        assert engine.ledger.cash == engine.ledger.equity == 1000
        assert engine.ledger.positions == {"B": -1, "P": 1}
        assert engine.ledger.settlements == {}
        assert engine.result.settlement_outcomes == []
        assert all(post.time == START for post in engine.ledger.postings)
    else:
        result = engine.run(events, ScriptedTargets({}))
        # Identical puts exactly offset: -1000 + 1000 settlement cash.
        assert result.status == "COMPLETED"
        assert result.ledger.cash == result.ledger.equity == 1000
        assert result.ledger.components()["net_pnl"] == 0
        assert [record["record"].cashflow for record in result.settlement_outcomes] == [-1000, 1000]


def test_stale_essential_mark_is_unknown_and_preserves_last_valid_equity():
    opening = Opening("IMPORT", cash=100, positions={"S": 1}, marks={"S": 100}, opening_equity=200)
    stale = replace(packet(OPEN, {"S": 100}), observed_at={"S": START})
    events = [Event("inception", START, 0, packet(START, {"S": 100})),
              Event("stale-mark", OPEN, 0, stale)]
    result = Engine(stocks("S"), opening, Rules(OPEN, ("S",), 0), ZeroMargin()).run(
        events, ScriptedTargets({}))
    assert result.status == "STOP_DATA_GAP"
    assert result.snapshots[-1].risk.equity is None
    assert result.snapshots[-1].pnl is None
    assert result.last_valid_snapshot.time == START
    assert result.last_valid_snapshot.risk.equity == 200
    assert result.ledger.positions == {"S": 1}


def test_option_order_is_cancelled_at_fixing_cutoff_before_execution():
    events = [decision(marks={"S": 100, "P": 2}),
              Event("expiry", OPEN, 0, packet(OPEN, {"S": 100, "P": 5}), execution=True,
                    quotes=(quote(OPEN, "P", bid=5, ask=5),)),
              Event("horizon", NEXT_CLOSE, 0, packet(NEXT_CLOSE, {"S": 100}))]
    result, _ = run_cash(events, {"close": {"P": 1}}, instruments=option_instruments(),
                         rules=Rules(NEXT_CLOSE, ("P",), 0))
    assert result.fills == ()
    assert any(x.outcome == "expired" and x.reason == "fixing cutoff" for x in result.order_outcomes)
    assert result.ledger.cash == 1000


def test_known_borrow_breach_precedes_pending_fill_and_has_data_gap_flag():
    opening = Opening("IMPORT", cash=1100, positions={"S": -1}, marks={"S": 100},
                      restricted_short_proceeds=100, opening_equity=1000,
                      short_lots=(ShortLot("opening-short", "S", 1, 100, START),))
    rules = Rules(OPEN, ("S",), 0, funding=Funding(borrow_rates={"S": 0}),
                  borrow={"S": Borrow(True, 10)})
    events = [decision(), Event("recall", OPEN, 0, None, borrow={"S": Borrow(False, 0)},
                               execution=True, quotes=(quote(OPEN),))]
    policy = ScriptedTargets({"close": {"S": 0}})
    result = Engine(stocks("S"), opening, rules, ZeroMargin()).run(events, policy)
    assert result.status == "STOP_WITH_BREACH"
    assert len(result.breaches) == 1 and result.breaches[0].data_gap
    assert result.breaches[0].risk.borrow_failures == ("S",)
    assert result.ledger.positions == {"S": -1}
    assert result.ledger.cash == 1100
    assert result.fills == ()
    assert len(policy.views) == 1


def test_new_cash_rate_applies_to_the_following_interval():
    day1, day2 = START + timedelta(days=1), START + timedelta(days=2)
    events = [Event("initial", START, 0, packet(START, {})),
              Event("rate-change", day1, 0, packet(day1, {}), funding=Funding(cash_rate=0.730)),
              Event("horizon", day2, 0, packet(day2, {}))]
    result, _ = run_cash(events, {}, rules=Rules(day2, ("S",), 0, funding=Funding(cash_rate=0.365)))
    interest = [posting.cash_change for posting in result.ledger.postings if posting.kind == "interest"]
    assert interest == pytest.approx([1, 2.002])
    assert result.ledger.cash == pytest.approx(1003.002)


def test_policy_receives_only_frozen_released_information():
    class ProbePolicy:
        name = "read-only-probe"

        def targets(self, view):
            with pytest.raises(TypeError):
                view.positions["S"] = 999
            with pytest.raises(TypeError):
                view.instruments["S"] = Instrument("S", "stock")
            with pytest.raises(TypeError):
                view.valuation.metadata["nested"]["value"] = 999
            assert not hasattr(view, "events")
            assert not hasattr(view, "rng")
            assert [(o.id, o.value) for o in view.history] == [("released", 100)]
            return {}

    event = replace(decision(), valuation=packet(START, {"S": 100}, metadata={"nested": {"value": 1}}),
                    observations=(Observation("released", "S", 100, START, START),))
    result = Engine(stocks("S"), Opening("IMPORT", cash=1000, opening_equity=1000),
                    Rules(OPEN, ("S",), 0), ZeroMargin()).run(
        [event, Event("end", OPEN, 0, packet(OPEN, {"S": 110},
                                            metadata={"nested": {"value": 2}}))], ProbePolicy())
    assert result.status == "COMPLETED"
    assert result.ledger.cash == 1000


def test_mutating_source_rate_dictionary_cannot_reprice_previous_interval():
    day1 = START + timedelta(days=1)
    source_rates = {"S": 0.365}
    opening = Opening("IMPORT", cash=1100, positions={"S": -1}, marks={"S": 100},
                      restricted_short_proceeds=100, opening_equity=1000,
                      short_lots=(ShortLot("old-short", "S", 1, 100, START),))
    rules = Rules(day1, ("S",), 0, funding=Funding(borrow_rates=source_rates),
                  borrow={"S": Borrow(True, 10)})

    def released_events():
        yield Event("inception", START, 0, packet(START, {"S": 100}))
        source_rates["S"] = 0.730
        yield Event("new-rate", day1, 0, packet(day1, {"S": 100}),
                    funding=Funding(borrow_rates={"S": 0.730}))

    result = Engine(stocks("S"), opening, rules, ZeroMargin()).run(
        released_events(), ScriptedTargets({}))
    # The released first interval is one day at 36.5% on 100 borrowed currency units.
    assert result.ledger.components()["borrow_charges"] == pytest.approx(0.1)
    assert result.ledger.cash == pytest.approx(1099.9)


def test_future_source_metadata_mutation_cannot_change_recorded_valuation_parameters():
    source_metadata = {"parameters": {"sigma": 0.2}}

    def released_events():
        yield Event("inception", START, 0, packet(START, {"S": 100}, metadata=source_metadata))
        source_metadata["parameters"]["sigma"] = 0.8
        yield Event("horizon", OPEN, 0, packet(OPEN, {"S": 100}))

    result = Engine(stocks("S"), Opening("IMPORT", cash=1000, opening_equity=1000),
                    Rules(OPEN, ("S",), 0), ZeroMargin()).run(released_events(), ScriptedTargets({}))
    assert result.valuations[0]["metadata"]["parameters"]["sigma"] == 0.2


@pytest.mark.parametrize("field", ["decision", "execution", "terminal_decision"])
def test_event_control_flags_require_booleans_not_truthy_strings(field):
    arguments = dict(decision=True, execution=False, terminal_decision=False,
                     earliest_execution=OPEN, order_deadline=NEXT_CLOSE)
    arguments[field] = "false"
    with pytest.raises(InputError):
        Event("malformed-control", START, 0, packet(START, {"S": 100}), **arguments)
