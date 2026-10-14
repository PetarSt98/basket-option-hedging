"""Independent cash distribution checks, including short payment obligations."""

from datetime import datetime, timedelta, timezone

import pytest

from basket_hedging.contracts import Funding, InputError, Instrument, Opening
from basket_hedging.engine import Engine
from basket_hedging.ledger import Ledger
from basket_hedging.market import Borrow, Event, Quote, Rules, ValuationPacket
from basket_hedging.risk import UnnettedLiabilityStressMargin


TIME = datetime(2020, 1, 2, tzinfo=timezone.utc)


@pytest.mark.parametrize("shares", [4, -4, 0])
def test_distribution_sign_reconciliation_and_idempotency(shares):
    ledger = Ledger({"S": Instrument("S", "stock")},
                    Opening("IMPORT", cash=1000, opening_equity=1000))
    ledger.set_marks({"S": 100})
    if shares:
        ledger.fill("trade", TIME, "S", shares, 100)
    before = ledger.cash
    ledger.dividend("distribution", TIME, "S", 2)
    assert ledger.cash == before + shares * 2
    assert ledger.equity == 1000 + shares * 2
    assert ledger.components()["hedge_pnl"] == shares * 2
    posts = ledger.postings
    ledger.dividend("distribution", TIME, "S", 2)
    assert ledger.postings == posts
    with pytest.raises(InputError, match="conflicting"):
        ledger.dividend("distribution", TIME, "S", 3)


def test_ex_date_distribution_and_price_drop_offset_before_cover():
    end = TIME + timedelta(days=1)
    instruments = {"S": Instrument("S", "stock")}
    rules = Rules(end, ("S",), 0, funding=Funding(borrow_rates={"S": 0}),
                  borrow={"S": Borrow(True, 100)})

    def packet(time, price):
        return ValuationPacket(time, {"S": price}, {"unchanged": {"S": price}},
                               observed_at={"S": time})

    # Initial buy at 100, mandatory dividend +8, ex-dividend mark 98, sale at 98.
    opening = Opening("IMPORT", cash=600, positions={"S": 4}, marks={"S": 100},
                      opening_equity=1000)
    events = [Event("start", TIME, 0, packet(TIME, 100), decision=True,
                    earliest_execution=end, order_deadline=end, terminal_decision=True),
              Event("ex", end, 0, packet(end, 98), dividends={"S": 2},
                    quotes=(Quote("q", "S", end, end, 98, 98, 100, 100),))]
    result = Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()).run(
        events, lambda view: {"S": 0})
    assert result.status == "COMPLETED"
    assert result.ledger.cash == 1000
    assert result.ledger.components()["hedge_pnl"] == 0
    distribution = next(p for p in result.ledger.postings if p.kind == "dividend")
    assert distribution.quantity == 4
    assert distribution.cash_change == 8


def test_unfunded_distribution_is_posted_before_breach():
    from basket_hedging.contracts import ShortLot

    instruments = {"S": Instrument("S", "stock")}
    opening = Opening("IMPORT", cash=101, positions={"S": -1}, marks={"S": 100},
                      opening_equity=1, restricted_short_proceeds=100,
                      short_lots=(ShortLot("lot", "S", 1, 100, TIME),))
    packet = ValuationPacket(TIME, {"S": 98}, {"unchanged": {"S": 98}},
                             observed_at={"S": TIME})
    # Import marks must agree, so the opening event uses 100, then the ex-date 98.
    start_packet = ValuationPacket(TIME, {"S": 100}, {"unchanged": {"S": 100}},
                                   observed_at={"S": TIME})
    end = TIME + timedelta(days=1)
    from dataclasses import replace
    events = [Event("start", TIME, 0, start_packet),
              Event("ex", end, 0, replace(packet, as_of=end, observed_at={"S": end}),
                    dividends={"S": 2})]
    rules = Rules(end, ("S",), 0, funding=Funding(borrow_rates={"S": 0}),
                  borrow={"S": Borrow(True, 100)})
    result = Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()).run(
        events, lambda view: None)
    assert result.status == "STOP_WITH_BREACH"
    assert result.ledger.cash == 99
    assert result.ledger.positions == {"S": -1}
    assert result.ledger.equity == 1
    assert result.ledger.available_cash == -1
    assert result.ledger.components()["hedge_pnl"] == 0


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf")])
def test_invalid_distribution_rejected(amount):
    with pytest.raises(InputError):
        Event("bad", TIME, 0, dividends={"S": amount})
