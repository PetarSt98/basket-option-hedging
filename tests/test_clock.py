"""The replay clock advances through actual data and independent due times."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from basket_hedging.contracts import Funding, InputError, Instrument, Opening
from basket_hedging.engine import Engine
from basket_hedging.market import Event, Rules, ValuationPacket
from basket_hedging.replay import load_case, replay_clock, write_case
from basket_hedging.risk import UnnettedLiabilityStressMargin


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
HORIZON = START + timedelta(hours=10)
STOCKS = {"S": Instrument("S", "stock")}


@dataclass(frozen=True)
class PendingOrder:
    earliest_execution: datetime
    deadline: datetime


def event(hour, *, sequence=0, name=None, **kwargs):
    time = START + timedelta(hours=hour)
    return Event(name or f"data/{hour}/{sequence}", time, sequence, None, **kwargs)


def test_live_orders_add_clock_boundaries_and_cancellation_removes_old_deadline():
    pending = []
    clock = replay_clock([event(0), event(10)], lambda: pending, STOCKS, Rules(HORIZON, ("S",), 0))
    assert next(clock).id == "data/0/0"
    pending.append(PendingOrder(START + timedelta(hours=2), START + timedelta(hours=4)))
    activation = next(clock)
    assert activation.time == START + timedelta(hours=2)
    assert activation.valuation is None
    assert not activation.quotes and not activation.execution and not activation.decision
    pending[:] = [PendingOrder(START + timedelta(hours=3), START + timedelta(hours=5))]
    # The former four-hour deadline disappears. Past activations are not repeated.
    assert [record.time for record in clock] == [START + timedelta(hours=3),
                                                START + timedelta(hours=5), HORIZON]


def test_source_records_replace_same_time_timers_and_keep_every_horizon_sequence():
    records = [event(0), event(2), event(2, sequence=1), event(10), event(10, sequence=1)]
    pending = [PendingOrder(START + timedelta(hours=2), HORIZON)]
    actual = list(replay_clock(records, lambda: pending, STOCKS, Rules(HORIZON, ("S",), 0)))
    assert actual == records


def test_all_registered_option_expiries_advance_clock_in_order():
    instruments = {**STOCKS,
                   "P": Instrument("P", "put", expiry=START + timedelta(hours=3), strike=95, underlying="S"),
                   "B": Instrument("B", "basket_put", expiry=START + timedelta(hours=6), strike=100,
                                   coefficients=(("S", 1),))}
    actual = list(replay_clock([event(0), event(10)], lambda: (), instruments, Rules(HORIZON, ("S", "P"), 0)))
    assert [record.time for record in actual] == [START, START + timedelta(hours=3),
                                                 START + timedelta(hours=6), HORIZON]
    assert actual[1].id == "clock/2026-01-01T03:00:00+00:00"
    assert actual[1].fixings == ()  # The clock cannot manufacture the official fixing.


def test_released_funding_rule_boundary_is_preserved():
    funding_change = event(4, funding=Funding(0.04))
    records = [event(0), funding_change, event(10)]
    actual = list(replay_clock(records, lambda: (), STOCKS, Rules(HORIZON, ("S",), 0)))
    assert actual == records
    assert actual[1].funding.cash_rate == 0.04


def test_explicit_start_and_missing_horizon_create_empty_clock_records():
    actual = list(replay_clock([event(1)], lambda: (), STOCKS, Rules(HORIZON, ("S",), 0, start_time=START)))
    assert [record.time for record in actual] == [START, START + timedelta(hours=1), HORIZON]
    assert actual[0].valuation is actual[-1].valuation is None
    assert actual[0].observations == actual[-1].observations == ()


@pytest.mark.parametrize("explicit_start,expected", [(START, [START, HORIZON]), (None, [HORIZON])])
def test_empty_replay_keeps_required_missing_input_boundaries(explicit_start, expected):
    actual = list(replay_clock([], lambda: (), STOCKS, Rules(HORIZON, ("S",), 0, start_time=explicit_start)))
    assert [record.time for record in actual] == expected
    assert all(record.valuation is None for record in actual)


def test_records_before_declared_start_are_rejected():
    clock = replay_clock([event(0)], lambda: (), STOCKS,
                         Rules(HORIZON, ("S",), 0, start_time=START + timedelta(hours=1)))
    with pytest.raises(InputError, match="precede the start"):
        next(clock)


def test_clock_rejects_backward_availability_instead_of_reordering_data():
    clock = replay_clock([event(2), event(1)], lambda: (), STOCKS, Rules(HORIZON, ("S",), 0))
    assert next(clock).time == START + timedelta(hours=2)
    with pytest.raises(InputError, match="move backward"):
        next(clock)


def test_next_source_row_is_not_read_before_current_record_has_been_consumed():
    def source():
        yield event(0)
        raise AssertionError("future source row was read")

    clock = replay_clock(source(), lambda: (), STOCKS, Rules(HORIZON, ("S",), 0))
    assert next(clock).id == "data/0/0"
    # A first-row breach can stop here without encountering malformed future data.
    with pytest.raises(AssertionError, match="future source row"):
        next(clock)


def test_future_data_is_never_released_past_horizon():
    actual = list(replay_clock([event(0), event(11)], lambda: (), STOCKS, Rules(HORIZON, ("S",), 0)))
    assert [record.time for record in actual] == [START, HORIZON]
    assert actual[-1].valuation is None


def test_case_roundtrip_preserves_start_time_and_calibration_cutoff(tmp_path):
    packet = ValuationPacket(START, {"S": 100}, {"unchanged": {"S": 100}},
                             observed_at={"S": START}, calibration_cutoff=START - timedelta(days=1))
    rules = Rules(HORIZON, ("S",), 0, start_time=START)
    path = write_case(tmp_path / "clock.json", STOCKS,
                      Opening("IMPORT", cash=1000, opening_equity=1000), rules,
                      [Event("inception", START, 0, packet)], {"source": "hand-authored test"})
    case = load_case(path)
    assert case.rules.start_time == START
    assert next(case.events()).valuation.calibration_cutoff == START - timedelta(days=1)


def test_unfilled_order_timer_changes_declared_event_boundary_interest_compounding():
    horizon = START + timedelta(days=2)

    class UnchangedStockPricer:
        def value(self, view):
            return ValuationPacket(view.time, {"S": 100}, {"unchanged": {"S": 100}},
                                   observed_at={"S": START})

    results = []
    for submit_order in (False, True):
        rules = Rules(horizon, ("S",), 0, funding=Funding(cash_rate=0.365),
                      execution_delay_seconds=86400, max_mark_age_seconds=172800)

        def policy(view):
            return {"S": 1} if submit_order and view.time == START else None

        result = Engine(STOCKS, Opening("IMPORT", cash=1000, opening_equity=1000),
                        rules, UnnettedLiabilityStressMargin(), UnchangedStockPricer()).run(
            [Event("start", START, 0), Event("end", horizon, 0)], policy)
        assert result.status == "COMPLETED"
        assert result.fills == () and result.ledger.positions == {}
        assert not result.breaches
        results.append(result)

    # Same cash-only exposure and source records; the pending order adds a one-day timer.
    # One interval: 1,000 * (1 + .365 * 2 / 365) = 1,002.
    # Two intervals: 1,000 * 1.001 * 1.001 = 1,002.001.
    assert [result.ledger.cash for result in results] == pytest.approx(
        [1002.0, 1002.001], abs=1e-10, rel=0)
    interest = [[post.cash_change for post in result.ledger.postings if post.kind == "interest"]
                for result in results]
    assert interest[0] == pytest.approx([2.0], abs=1e-12, rel=0)
    assert interest[1] == pytest.approx([1.0, 1.001], abs=1e-12, rel=0)
