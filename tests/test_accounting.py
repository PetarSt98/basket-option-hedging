"""Hand-derived ledger checks, independent of execution and valuation models."""

from datetime import datetime, timedelta, timezone

import pytest

from basket_hedging.contracts import (
    DataGap, Funding, InputError, Instrument, Opening, ReconciliationError, ShortLot,
    validate_instruments,
)
from basket_hedging.ledger import Ledger


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
EXPIRY = START + timedelta(days=30)


@pytest.fixture
def instruments():
    return {
        "S": Instrument("S", "stock"),
        "B": Instrument("B", "basket_put", multiplier=100, strike=100,
                        expiry=EXPIRY, coefficients=(("S", 1.0),)),
        "P": Instrument("P", "put", multiplier=50, strike=100,
                        expiry=EXPIRY, underlying="S"),
        "C": Instrument("C", "call", multiplier=50, strike=100,
                        expiry=EXPIRY, underlying="S"),
    }


def cash_book(instruments, cash=1000):
    return Ledger(instruments, Opening("IMPORT", cash=cash, opening_equity=cash))


def originate(instruments, fee=0):
    ledger = Ledger(instruments, Opening("ORIGINATE", initial_capital=1000,
                                         liability_id="B", liability_quantity=-1))
    ledger.originate("inception", START, "B", -1, 2, 2, fee=fee)
    return ledger


def snapshot(ledger):
    return (ledger.cash, ledger.positions.copy(), ledger.marks.copy(),
            ledger.posted_margin_cash, tuple(ledger.short_lots), ledger.postings,
            ledger.settlements.copy())


def test_origination_premium_offsets_liability_and_is_booked_once(instruments):
    ledger = originate(instruments, fee=1)
    assert ledger.cash == 1199
    assert ledger.positions == {"B": -1}
    assert ledger.position_values() == {"B": -200}
    assert ledger.equity == 999
    assert ledger.baseline == 1000
    assert ledger.original_premium == 200
    expected = snapshot(ledger)
    ledger.originate("inception", START, "B", -1, 2, 2, fee=1)
    assert snapshot(ledger) == expected
    with pytest.raises(InputError):
        ledger.originate("another-inception", START, "B", -1, 2, 2)
    parts = ledger.components()
    assert parts["liability_pnl"] == 0
    assert parts["fees"] == 1
    assert parts["net_pnl"] == -1


def test_premium_mark_difference_is_liability_pnl(instruments):
    ledger = Ledger(instruments, Opening("ORIGINATE", initial_capital=1000, liability_id="B"))
    ledger.originate("origin", START, "B", -2, 2.5, 2, fee=3)
    assert ledger.cash == 1497
    assert ledger.equity == 1097
    assert ledger.components()["liability_pnl"] == 100
    assert ledger.components()["net_pnl"] == 97


def test_import_is_no_cash_credit_and_uses_opening_marked_equity(instruments):
    ledger = Ledger(instruments, Opening("IMPORT", cash=700, positions={"B": -1},
                                         marks={"B": 3}, opening_equity=400,
                                         original_premium=200))
    assert ledger.cash == 700
    assert ledger.baseline == 400
    assert ledger.postings == ()
    assert ledger.components()["net_pnl"] == 0
    ledger.set_marks({"B": 4})
    assert ledger.components()["liability_pnl"] == -100
    ledger.settle("settle-B", EXPIRY, "B", 4)
    assert ledger.cash == 300
    assert ledger.components()["net_pnl"] == -100


def test_financially_infeasible_import_is_valid_ledger_input(instruments):
    ledger = Ledger(instruments, Opening("IMPORT", cash=-20, positions={"B": -1},
                                         marks={"B": 2}, opening_equity=-220,
                                         posted_margin_cash=50))
    assert ledger.cash == -20
    assert ledger.equity == -220
    assert ledger.available_cash == -70
    assert ledger.components()["net_pnl"] == 0


@pytest.mark.parametrize("field,value", [
    ("opening_equity", 801), ("restricted_short_proceeds", 199),
    ("short_lots", ()), ("posted_margin_cash", -1),
])
def test_malformed_import_is_rejected(instruments, field, value):
    fields = dict(cash=1000, positions={"S": -2}, marks={"S": 100}, opening_equity=800,
                  restricted_short_proceeds=200,
                  short_lots=(ShortLot("old", "S", 2, 100, START),))
    fields[field] = value
    with pytest.raises(InputError):
        Ledger(instruments, Opening("IMPORT", **fields))


@pytest.mark.parametrize("option_id", ["P", "C"])
@pytest.mark.parametrize("quantity", [-2, 2])
def test_option_trade_units_and_marked_equity_before_costs(instruments, option_id, quantity):
    ledger = cash_book(instruments)
    ledger.set_marks({option_id: 2})
    ledger.fill("option", START, option_id, quantity, 2)
    assert ledger.cash == 1000 - quantity * 100
    assert ledger.position_values()[option_id] == quantity * 100
    assert ledger.equity == 1000
    assert ledger.components()["net_pnl"] == 0
    assert ledger.components()["option_premium_turnover"] == 200


def test_spread_slippage_fees_are_charged_once(instruments):
    ledger = cash_book(instruments)
    ledger.set_marks({"P": 2})
    ledger.fill("buy", START, "P", 2, 2.10, fee=3)
    assert ledger.cash == pytest.approx(787)
    assert ledger.equity == pytest.approx(987)
    components = ledger.components()
    assert components["execution_cost"] == pytest.approx(10)
    assert components["fees"] == 3
    assert components["hedge_pnl"] == 0
    ledger.fill("sell", START + timedelta(hours=1), "P", -2, 1.90, fee=3)
    assert ledger.cash == pytest.approx(974)
    assert ledger.positions == {}
    assert ledger.components()["execution_cost"] == pytest.approx(20)
    assert ledger.components()["fees"] == 6
    assert ledger.components()["net_pnl"] == pytest.approx(-26)
    assert len({p.id for p in ledger.postings}) == len(ledger.postings)


def test_margin_allocation_only_changes_cash_restrictions(instruments):
    ledger = cash_book(instruments)
    ledger.allocate_margin("margin1", START, 300)
    assert (ledger.cash, ledger.equity, ledger.available_cash) == (1000, 1000, 700)
    ledger.allocate_margin("margin2", START + timedelta(hours=1), 120)
    assert (ledger.cash, ledger.equity, ledger.available_cash) == (1000, 1000, 880)
    ledger.allocate_margin("margin3", START + timedelta(hours=2), 0)
    assert ledger.available_cash == 1000
    assert ledger.components()["net_pnl"] == 0


def test_fifo_partial_cover_and_crossing_zero(instruments):
    ledger = cash_book(instruments)
    ledger.set_marks({"S": 100})
    ledger.fill("short1", START, "S", -10, 100)
    assert (ledger.cash, ledger.restricted_short_proceeds, ledger.available_cash) == (2000, 1000, 1000)
    assert ledger.equity == 1000
    ledger.set_marks({"S": 120})
    ledger.fill("short2", START + timedelta(hours=1), "S", -5, 120)
    assert ledger.cash == 2600
    assert ledger.restricted_short_proceeds == 1600
    ledger.set_marks({"S": 110})
    ledger.fill("cover12", START + timedelta(hours=2), "S", 12, 110)
    assert ledger.cash == 1280
    assert ledger.positions == {"S": -3}
    assert ledger.restricted_short_proceeds == 360
    assert [(lot.quantity, lot.sale_price) for lot in ledger.short_lots] == [(3, 120)]
    ledger.set_marks({"S": 115})
    ledger.fill("cross", START + timedelta(hours=3), "S", 5, 115)
    assert ledger.cash == 705
    assert ledger.positions == {"S": 2}
    assert ledger.short_lots == []
    assert ledger.available_cash == 705
    assert ledger.equity == 935
    assert ledger.components()["hedge_pnl"] == -65
    assert ledger.components()["stock_turnover"] == 3495


def test_long_to_short_segregates_only_new_short_shares(instruments):
    ledger = Ledger(instruments, Opening("IMPORT", cash=500, positions={"S": 2},
                                         marks={"S": 100}, opening_equity=700))
    ledger.fill("cross", START, "S", -5, 100)
    assert ledger.cash == 1000
    assert ledger.positions == {"S": -3}
    assert ledger.restricted_short_proceeds == 300
    assert ledger.equity == 700


def test_interest_borrow_and_rate_change_boundaries(instruments):
    ledger = Ledger(instruments, Opening("IMPORT", cash=1000, positions={"S": -2},
                                         marks={"S": 100}, opening_equity=800,
                                         posted_margin_cash=300, restricted_short_proceeds=200,
                                         short_lots=(ShortLot("old", "S", 2, 100, START),)))
    day1, day2 = START + timedelta(days=1), START + timedelta(days=2)
    ledger.accrue("a1", START, day1, Funding(0.10, 0.20, {"S": 0.03}))
    assert ledger.cash == pytest.approx(1000 + 104 / 365)
    components = ledger.components()
    assert components["interest"] == pytest.approx(110 / 365)
    assert components["borrow_charges"] == pytest.approx(6 / 365)
    ledger.set_marks({"S": 110})
    ledger.accrue("a2", day1, day2, Funding(0.20, 0.10, {"S": 0.06}))
    expected_second_interest = (0.2 * (500 + 104 / 365) + 30) / 365
    assert ledger.cash == pytest.approx(1000 + 104 / 365 + expected_second_interest - 13.2 / 365)
    assert ledger.components()["borrow_charges"] == pytest.approx(19.2 / 365)
    assert ledger.components()["interest"] == pytest.approx(110 / 365 + expected_second_interest)


def test_missing_accrual_input_does_not_consume_operation_id(instruments):
    ledger = Ledger(instruments, Opening("IMPORT", cash=1000, positions={"S": -2},
                                         marks={"S": 100}, opening_equity=800,
                                         restricted_short_proceeds=200,
                                         short_lots=(ShortLot("old", "S", 2, 100, START),)))
    before = snapshot(ledger)
    end = START + timedelta(days=1)
    with pytest.raises(DataGap):
        ledger.accrue("retry", START, end, Funding())
    assert snapshot(ledger) == before
    ledger.accrue("retry", START, end, Funding(borrow_rates={"S": 0.03}))
    assert ledger.cash == pytest.approx(1000 - 6 / 365)


@pytest.mark.parametrize("option_id,fixing,payoff", [
    ("P", 95, 5), ("P", 100, 0), ("P", 105, 0),
    ("C", 95, 0), ("C", 100, 0), ("C", 105, 5),
])
@pytest.mark.parametrize("quantity", [-2, 2])
def test_option_settlement_signed_cash_and_multiplier_once(instruments, option_id, fixing, payoff, quantity):
    ledger = Ledger(instruments, Opening("IMPORT", cash=1000, positions={option_id: quantity},
                                         marks={option_id: 3}, opening_equity=1000 + quantity * 150))
    assert instruments[option_id].payoff({"S": fixing}) == payoff
    ledger.set_marks({option_id: payoff})
    before = ledger.equity
    settlement = ledger.settle("expiry", EXPIRY, option_id, payoff)
    assert settlement.cashflow == quantity * 50 * payoff
    assert ledger.equity == before
    assert ledger.cash == 1000 + quantity * 50 * payoff
    assert ledger.positions == {}
    assert ledger.components()["hedge_pnl"] == quantity * 50 * (payoff - 3)
    assert ledger.settle("expiry", EXPIRY, option_id, payoff) == settlement
    assert ledger.cash == 1000 + quantity * 50 * payoff


def test_import_rejects_missing_fractional_short_lot_with_material_proceeds(instruments):
    with pytest.raises(InputError, match="short lots do not match"):
        Ledger(instruments, Opening("IMPORT", cash=1000, positions={"S": -5e-9},
                                    marks={"S": 1e10}, opening_equity=950))


@pytest.mark.parametrize("payoff", [0, 4])
def test_settlement_idempotence_and_conflicting_duplicate_ids(instruments, payoff):
    ledger = originate(instruments)
    first = ledger.settle("expiry-B", EXPIRY, "B", payoff)
    before = snapshot(ledger)
    assert ledger.settle("expiry-B", EXPIRY, "B", payoff) == first
    assert snapshot(ledger) == before
    assert ledger.positions == {}
    with pytest.raises(InputError, match="conflicting"):
        ledger.settle("expiry-B", EXPIRY, "B", payoff + 1)
    with pytest.raises(InputError, match="already settled"):
        ledger.settle("different-ID", EXPIRY, "B", payoff)
    assert snapshot(ledger) == before


def test_unfunded_mandatory_obligation_is_booked_in_full(instruments):
    ledger = Ledger(instruments, Opening("IMPORT", cash=30, positions={"P": -1},
                                         marks={"P": 1}, opening_equity=-20))
    record = ledger.settle("expiry", EXPIRY, "P", 2)
    assert record.cashflow == -100
    assert ledger.cash == -70
    assert ledger.positions == {}
    assert ledger.equity == -70
    assert ledger.components()["net_pnl"] == -50


def test_reconciliation_from_independent_trade_and_settlement_components(instruments):
    ledger = originate(instruments, fee=1)
    ledger.set_marks({"B": 2, "C": 2, "S": 100})
    ledger.fill("call-buy", START, "C", 2, 2.1, fee=3)
    ledger.fill("stock-short", START, "S", -10, 100, fee=2)
    assert ledger.equity == pytest.approx(984)
    ledger.set_marks({"B": 4, "C": 1, "S": 90})
    ledger.fill("stock-cover", START + timedelta(days=1), "S", 10, 90, fee=2)
    ledger.fill("call-sell", START + timedelta(days=1), "C", -2, 0.95, fee=1)
    assert ledger.cash == pytest.approx(1176)
    assert ledger.equity == pytest.approx(776)
    ledger.settle("expiry-B", EXPIRY, "B", 4)
    assert ledger.cash == pytest.approx(776)
    assert ledger.positions == {}
    parts = ledger.components()
    assert parts["liability_pnl"] == -200
    assert parts["hedge_pnl"] == 0  # stock +100 and calls -100
    assert parts["execution_cost"] == pytest.approx(15)
    assert parts["fees"] == 9
    assert parts["net_pnl"] == pytest.approx(-224)
    assert parts["stock_turnover"] == 1900
    assert parts["option_premium_turnover"] == 305
    # An unrecorded cash injection is detected, rather than absorbed by a residual component.
    ledger.cash += 1
    with pytest.raises(ReconciliationError):
        ledger.components()


def test_clone_preview_changes_no_live_accounting_or_settlement_ids(instruments):
    ledger = originate(instruments)
    ledger.set_marks({"B": 2, "P": 1, "S": 100})
    before = snapshot(ledger)
    preview = ledger.clone()
    preview.fill("preview-buy", START, "P", 10, 1, fee=7)
    preview.allocate_margin("preview-margin", START, 999)
    preview.settle("expiry-B", EXPIRY, "B", 5)
    assert snapshot(ledger) == before
    ledger.settle("expiry-B", EXPIRY, "B", 4)
    assert ledger.cash == 800


def test_generated_posting_id_collision_rejects_before_mutation(instruments):
    ledger = cash_book(instruments)
    ledger.set_marks({"P": 2})
    ledger.fill("trade", START, "P", 1, 2, fee=1)
    before = snapshot(ledger)
    with pytest.raises(InputError):
        ledger.allocate_margin("trade:fee", START, 500)
    assert snapshot(ledger) == before


def test_instrument_units_basket_shares_and_finite_inputs():
    basket = Instrument("B", "basket_put", multiplier=17, strike=100,
                        expiry=EXPIRY, coefficients=(("A", 2), ("Z", 0.5)))
    assert basket.payoff({"A": 30, "Z": 40}) == 20  # B=2*30+0.5*40=80.
    with pytest.raises(InputError):
        Instrument("S", "stock", multiplier=100)
    with pytest.raises(InputError):
        Instrument("S", "stock", multiplier=float("nan"))
    with pytest.raises(InputError):
        Funding(cash_rate=float("inf"))
    with pytest.raises(InputError):
        validate_instruments({"A": Instrument("A", "stock", currency="USD"),
                              "Z": Instrument("Z", "stock", currency="EUR")})


@pytest.mark.parametrize("mode", ["ORIGINATE", "IMPORT"])
def test_basket_call_accounting_units_and_settlement_are_liability_pnl(mode):
    instruments = {
        "A": Instrument("A", "stock"), "Z": Instrument("Z", "stock"),
        "B": Instrument("B", "basket_call", multiplier=10, strike=130,
                        expiry=EXPIRY, coefficients=(("A", 2), ("Z", .5))),
    }
    # Two short contracts at 15 per basket unit: cash +300, liability -300.
    if mode == "ORIGINATE":
        ledger = Ledger(instruments, Opening(mode, initial_capital=1000, liability_id="B"))
        ledger.originate("sale", START, "B", -2, 15, 15)
    else:
        ledger = Ledger(instruments, Opening(mode, cash=1300, positions={"B": -2},
                        marks={"B": 15}, opening_equity=1000, original_premium=300))
        assert ledger.postings == ()
    assert ledger.cash == 1300
    assert ledger.equity == 1000
    assert ledger.components()["liability_pnl"] == 0
    before = snapshot(ledger)
    with pytest.raises(InputError, match="never a hedge instrument"):
        ledger.fill("forbidden-close", START, "B", 2, 15)
    assert snapshot(ledger) == before

    # Official basket = 2*55 + .5*100 = 160; unit call payoff 30.
    payoff = instruments["B"].payoff({"A": 55, "Z": 100})
    assert payoff == 30
    ledger.set_marks({"B": payoff})
    assert ledger.equity == 700
    ledger.settle("settle", EXPIRY, "B", payoff)
    assert ledger.cash == ledger.equity == 700
    assert ledger.positions == {}
    assert ledger.components()["liability_pnl"] == -300
    assert ledger.components()["hedge_pnl"] == 0
    assert ledger.components()["net_pnl"] == -300
    settled = snapshot(ledger)
    ledger.settle("settle", EXPIRY, "B", payoff)
    assert snapshot(ledger) == settled


@pytest.mark.parametrize("kind,expected", [("basket_put", 20), ("basket_call", 0)])
def test_basket_direction_and_single_liability_validation(kind, expected):
    basket = Instrument("B", kind, multiplier=17, strike=100, expiry=EXPIRY,
                        coefficients=(("A", 2), ("Z", .5)))
    assert basket.is_basket
    assert basket.risk_stocks == ("A", "Z")
    assert basket.payoff({"A": 30, "Z": 40}) == expected
    other = Instrument("other", "basket_call", strike=100, expiry=EXPIRY,
                       coefficients=(("A", 1),))
    with pytest.raises(InputError, match="one basket liability"):
        validate_instruments({"A": Instrument("A", "stock"), "Z": Instrument("Z", "stock"),
                              "B": basket, "other": other})
