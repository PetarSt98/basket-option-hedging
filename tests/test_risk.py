"""Research-proxy risk calculations checked against hand-derived portfolios."""

from datetime import datetime, timedelta, timezone

import pytest

from basket_hedging.contracts import DataGap, Funding, InputError, Instrument, Opening, ShortLot
from basket_hedging.engine import Engine
from basket_hedging.ledger import Ledger
from basket_hedging.market import Borrow, Event, Quote, Rules, ValuationPacket
from basket_hedging.risk import UnnettedLiabilityStressMargin, assess, borrow_failures


START = datetime(2026, 1, 1, 16, tzinfo=timezone.utc)
EXPIRY = START + timedelta(days=30)


@pytest.fixture
def instruments():
    return {
        "S": Instrument("S", "stock"),
        "B": Instrument("B", "basket_put", multiplier=100, strike=100,
                        expiry=EXPIRY, coefficients=(("S", 1.0),)),
        "P": Instrument("P", "put", multiplier=50, strike=100,
                        expiry=EXPIRY, underlying="S"),
    }


def packet(time, marks, scenarios=None):
    return ValuationPacket(time, marks, scenarios if scenarios is not None else {"unchanged": marks.copy()},
                           observed_at={key: time for key in marks}, input_cutoff=time)


def portfolio(instruments):
    ledger = Ledger(instruments, Opening(
        "IMPORT", cash=1000, positions={"B": -1, "P": 1, "S": -2},
        marks={"B": 2, "P": 1, "S": 100}, opening_equity=650,
        restricted_short_proceeds=200,
        short_lots=(ShortLot("old-short", "S", 2, 100, START),),
    ))
    valuation = packet(START, {"B": 2, "P": 1, "S": 100}, {
        "unchanged": {"B": 2, "P": 1, "S": 100},
        "down": {"B": 5, "P": 4, "S": 80},
        "up": {"B": 1, "P": 0, "S": 120},
    })
    return ledger, valuation


def test_unnetted_liabilities_plus_signed_portfolio_stress(instruments):
    ledger, valuation = portfolio(instruments)
    model = UnnettedLiabilityStressMargin()
    # V=-350. Down V=-460 => loss110. Up V=-340 => loss0.
    # Gross short-option liability200 + worst portfolio loss110 =310.
    assert model.required_margin(ledger, valuation) == 310
    assert model.label == "uncalibrated research margin proxy"
    risk = assess(ledger, 310, 25, {"S": Borrow(True, 10)})
    assert risk.feasible
    assert risk.equity == 650
    assert risk.eligible_cash_capacity == 800
    assert risk.maximum_shortfall == 0
    ledger.allocate_margin("allocate", START, 310)
    assert ledger.available_cash == 490
    assert ledger.equity == 650


def test_preview_recomputes_stress_on_previewed_positions(instruments):
    ledger, valuation = portfolio(instruments)
    preview = ledger.clone()
    preview.fill("preview", START, "S", -2, 100)
    assert preview.cash == 1200
    assert preview.restricted_short_proceeds == 400
    assert preview.equity == 650
    # V=-550. Down V=-620 => loss70. Up V=-580 => loss30.
    assert UnnettedLiabilityStressMargin().required_margin(preview, valuation) == 270
    preview.allocate_margin("allocate", START, 270)
    assert preview.available_cash == 530
    assert (ledger.cash, ledger.restricted_short_proceeds, ledger.positions["S"]) == (1000, 200, -2)
    assert ledger.postings == ()


@pytest.mark.parametrize("long_contracts", [0, 1, 20])
def test_long_option_values_do_not_offset_gross_liability_reserve(instruments, long_contracts):
    ledger = Ledger(instruments, Opening(
        "IMPORT", cash=1000, positions={"B": -1, "P": long_contracts},
        marks={"B": 2, "P": 1}, opening_equity=800 + long_contracts * 50,
    ))
    assert UnnettedLiabilityStressMargin().required_margin(
        ledger, packet(START, {"B": 2, "P": 1})) == 200


def test_short_constituent_options_add_unnetted_liability_reserve(instruments):
    ledger = Ledger(instruments, Opening("IMPORT", cash=1000, positions={"B": -1, "P": -2},
                                         marks={"B": 2, "P": 1}, opening_equity=700))
    assert UnnettedLiabilityStressMargin().required_margin(
        ledger, packet(START, {"B": 2, "P": 1})) == 300


@pytest.mark.parametrize("scenarios,error", [
    ({"down": {"B": 5, "P": 4, "S": 80}}, DataGap),
    ({"unchanged": {"B": 2, "P": 1, "S": 100}, "down": {"B": 5, "S": 80}}, DataGap),
    ({"unchanged": {"B": 2.1, "P": 1, "S": 100}}, InputError),
])
def test_missing_or_conflicting_stress_inputs_are_not_silently_filled(instruments, scenarios, error):
    ledger, _ = portfolio(instruments)
    with pytest.raises(error):
        UnnettedLiabilityStressMargin().required_margin(
            ledger, packet(START, {"B": 2, "P": 1, "S": 100}, scenarios))


def test_margin_and_ledger_must_use_identical_current_marks(instruments):
    ledger, _ = portfolio(instruments)
    with pytest.raises(InputError, match="same accounting marks"):
        UnnettedLiabilityStressMargin().required_margin(
            ledger, packet(START, {"B": 3, "P": 1, "S": 100}))


def test_entire_declared_stress_cannot_disappear_from_a_packet(instruments):
    ledger, valuation = portfolio(instruments)
    model = UnnettedLiabilityStressMargin(("unchanged", "down", "up"))
    assert model.required_margin(ledger, valuation) == 310
    incomplete = packet(START, dict(valuation.marks), {
        "unchanged": dict(valuation.marks), "up": dict(valuation.stress_marks["up"]),
    })
    # The omitted down case supplied the 110 loss, so accepting this would
    # incorrectly reduce margin from 310 to 200 and may conceal a breach.
    with pytest.raises(DataGap, match="missing declared margin stress scenarios: down"):
        model.required_margin(ledger, incomplete)
    assert ledger.postings == ()
    assert ledger.equity == 650


@pytest.mark.parametrize("names", [(), ("down",), ("unchanged", "unchanged"),
                                  ("unchanged", ""), ("unchanged", 1)])
def test_invalid_declared_scenario_names_are_configuration_errors(names):
    with pytest.raises(InputError, match="declared margin scenarios"):
        UnnettedLiabilityStressMargin(names)


@pytest.mark.parametrize("cash,equity,margin_gap,buffer_gap,total_gap", [
    (150, 350, 50, 25, 75),
    (210, 410, 0, 15, 15),
    (225, 425, 0, 0, 0),
])
def test_margin_cash_and_incremental_buffer_shortfalls_are_separate(
    instruments, cash, equity, margin_gap, buffer_gap, total_gap,
):
    ledger = Ledger(instruments, Opening(
        "IMPORT", cash=cash, positions={"B": -1, "P": 4},
        marks={"B": 2, "P": 2}, opening_equity=equity,
    ))
    risk = assess(ledger, 200, 25, {})
    assert risk.margin_cash_shortfall == margin_gap
    assert risk.buffer_cash_shortfall == buffer_gap
    assert risk.cash_requirement_shortfall == total_gap
    assert risk.cash_requirement_shortfall_lower_bound == total_gap
    assert risk.margin_equity_shortfall == risk.buffer_equity_shortfall == 0
    assert risk.maximum_shortfall == total_gap
    assert risk.feasible == (total_gap == 0)


def test_equity_shortfalls_not_double_counted_with_margin_and_buffer(instruments):
    ledger = Ledger(instruments, Opening("IMPORT", cash=350, positions={"B": -1},
                                         marks={"B": 2}, opening_equity=150))
    risk = assess(ledger, 200, 25, {})
    assert risk.margin_equity_shortfall == 50
    assert risk.buffer_equity_shortfall == 25
    assert risk.equity_requirement_shortfall == 75
    assert risk.cash_requirement_shortfall == 0
    assert risk.maximum_shortfall == 75


def test_negative_cash_capacity_is_not_clamped_in_requirement_gaps(instruments):
    ledger = Ledger(instruments, Opening(
        "IMPORT", cash=190, positions={"S": -2}, marks={"S": 100}, opening_equity=-10,
        restricted_short_proceeds=200,
        short_lots=(ShortLot("old", "S", 2, 100, START),),
    ))
    risk = assess(ledger, 0, 0, {"S": Borrow(True, 2)})
    assert risk.eligible_cash_capacity == 0
    assert risk.margin_cash_shortfall == risk.margin_equity_shortfall == 10
    assert risk.available_cash_deficit == risk.unencumbered_cash_deficit == risk.equity_deficit == 10
    assert risk.maximum_shortfall == 10
    assert risk.buffer_cash_shortfall == risk.buffer_equity_shortfall == 0
    assert not risk.feasible


@pytest.mark.parametrize("borrow", [{}, {"S": Borrow(False, 10)}, {"S": Borrow(True, 1)}])
def test_borrow_failure_is_independent_of_cash_and_margin(instruments, borrow):
    ledger, _ = portfolio(instruments)
    risk = assess(ledger, 310, 25, borrow)
    assert risk.maximum_shortfall == 0
    assert risk.borrow_failures == ("S",)
    assert borrow_failures(ledger, borrow) == ("S",)
    assert not risk.feasible


def test_missing_valuation_keeps_values_unknown_and_known_cash_failure_visible(instruments):
    ledger = Ledger(instruments, Opening("IMPORT", cash=-5, opening_equity=-5))
    risk = assess(ledger, None, 25, {}, valued=False)
    assert risk.equity is None
    assert risk.required_margin is None
    assert risk.margin_cash_shortfall is None
    assert risk.maximum_shortfall is None
    assert risk.unencumbered_cash_deficit == 5
    assert risk.cash_requirement_shortfall_lower_bound == 30
    assert not risk.feasible


@pytest.mark.parametrize("cash,expected_status,known_deficit", [
    (100, "STOP_WITH_BREACH", 50),
    (150, "STOP_DATA_GAP", 0),
])
def test_missing_valuation_preserves_known_cash_buffer_breach(instruments, cash, expected_status, known_deficit):
    opening = Opening("IMPORT", cash=cash, opening_equity=cash)
    result = Engine(instruments, opening, Rules(START, (), 150), UnnettedLiabilityStressMargin()).run(
        [Event("missing", START, 0)], lambda view: None)
    risk = result.snapshots[-1].risk
    assert result.status == expected_status
    assert risk.cash == cash
    assert risk.cash_requirement_shortfall_lower_bound == known_deficit
    assert risk.unencumbered_cash_deficit == 0  # Cash is positive; the buffer is binding.
    assert risk.required_margin is risk.equity is risk.maximum_shortfall is None
    assert risk.cash_requirement_shortfall is risk.buffer_cash_shortfall is None
    assert risk.margin_cash_shortfall is None
    assert result.snapshots[-1].pnl is None
    assert len(result.data_gaps) == 1
    assert result.decisions == [] and result.fills == ()
    if known_deficit:
        assert not risk.feasible
        assert len(result.breaches) == 1 and result.breaches[0].data_gap
    else:
        assert result.breaches == []


def test_missing_valuation_cash_buffer_check_excludes_restricted_short_proceeds(instruments):
    ledger = Ledger(instruments, Opening(
        "IMPORT", cash=200, positions={"S": -1}, marks={"S": 100}, opening_equity=100,
        restricted_short_proceeds=100, short_lots=(ShortLot("short", "S", 1, 100, START),),
    ))
    risk = assess(ledger, None, 150, {"S": Borrow(True, 1)}, valued=False)
    assert not risk.feasible
    assert risk.cash_requirement_shortfall_lower_bound == 50  # 150 - (200 - 100).
    assert risk.borrow_failures == ()
    assert risk.required_margin is risk.cash_requirement_shortfall is None


@pytest.mark.parametrize("valued", [True, False])
def test_position_limits_are_independent_of_margin_cash_and_marks(instruments, valued):
    ledger, _ = portfolio(instruments)
    risk = assess(ledger, 310 if valued else None, 25, {"S": Borrow(True, 10)},
                  valued=valued, position_limits={"B": 1, "P": .5, "S": 1})
    assert risk.position_failures == ("P", "S")
    assert risk.borrow_failures == ()
    assert risk.maximum_shortfall == (0 if valued else None)
    assert not risk.feasible
    at_limit = assess(ledger, 310 if valued else None, 25, {"S": Borrow(True, 10)},
                      valued=valued, position_limits={"B": 1, "P": 1, "S": 2})
    assert at_limit.position_failures == ()
    assert at_limit.feasible


@pytest.mark.parametrize("limits", [{"unknown": 1}, {"S": -1}, {"S": float("nan")}])
def test_invalid_position_limit_is_an_input_error(instruments, limits):
    ledger, _ = portfolio(instruments)
    with pytest.raises(InputError):
        assess(ledger, 310, 0, {"S": Borrow(True, 10)}, position_limits=limits)


class PendingDeltaTest:
    """Fixed test target demonstrates ordering; it is not a production policy."""

    name = "pending-delta-test"

    def __init__(self):
        self.calls = 0

    def targets(self, view):
        self.calls += 1
        return {"S": -40}


@pytest.mark.parametrize("execution", [False, True])
def test_actual_gap_breach_before_pending_fill_or_unscheduled_rebalance(instruments, execution):
    opened = START + timedelta(hours=17)
    deadline = START + timedelta(days=1)
    horizon = START + timedelta(days=2)
    initial = packet(START, {"B": 2, "S": 100}, {
        "unchanged": {"B": 2, "S": 100}, "down": {"B": 3, "S": 80},
    })
    gap = packet(opened, {"B": 6, "S": 80}, {
        "unchanged": {"B": 6, "S": 80}, "down": {"B": 8, "S": 70},
    })
    events = [
        Event("inception", START, 0, initial, origination_price=2, decision=True,
              earliest_execution=opened, order_deadline=deadline),
        Event("gap", opened, 0, gap, execution=execution,
              quotes=(Quote("gap-quote", "S", opened, opened, 80, 80, 100, 100),)),
    ]
    rules = Rules(horizon, ("S",), 0, funding=Funding(borrow_rates={"S": 0}),
                  borrow={"S": Borrow(True, 100)})
    policy = PendingDeltaTest()
    result = Engine(instruments, Opening("ORIGINATE", initial_capital=1000, liability_id="B"),
                    rules, UnnettedLiabilityStressMargin()).run(iter(events), policy)
    assert result.status == "STOP_WITH_BREACH"
    assert result.ledger.cash == 1200
    assert result.ledger.positions == {"B": -1}
    assert result.ledger.posted_margin_cash == 300  # Last successful allocation is retained.
    assert result.fills == ()
    assert result.ledger.settlements == {}
    assert policy.calls == 1
    assert len(result.breaches) == 1
    breach = result.breaches[0]
    assert breach.event_id == "gap" and breach.phase == "pretrade"
    assert breach.risk.equity == 600
    assert breach.risk.required_margin == 800
    assert breach.risk.margin_equity_shortfall == 200
    assert breach.risk.maximum_shortfall == 200
    assert [outcome.outcome for outcome in result.order_outcomes] == ["submitted", "cancelled"]
    assert result.ledger.components()["net_pnl"] == -400


def test_valid_but_infeasible_import_produces_recorded_inception_breach(instruments):
    opening = Opening("IMPORT", cash=350, positions={"B": -1}, marks={"B": 2}, opening_equity=150)
    result = Engine(instruments, opening, Rules(START, ("S",), 0), UnnettedLiabilityStressMargin()).run(
        [Event("inception", START, 0, packet(START, {"B": 2}))], PendingDeltaTest())
    assert result.status == "STOP_WITH_BREACH"
    assert result.baseline == 150
    assert result.ledger.cash == 350
    assert result.ledger.components()["net_pnl"] == 0
    assert len(result.breaches) == 1
    assert result.breaches[0].event_id == "inception"
    assert result.breaches[0].risk.margin_equity_shortfall == 50
    assert not any(post.kind == "origination" for post in result.ledger.postings)


@pytest.mark.parametrize("borrow", [{}, {"S": Borrow(True, 0)}])
def test_fractional_short_with_material_value_still_requires_borrow(instruments, borrow):
    # 5e-9 shares at 1e10 currency/share is a 50-currency short, not a cash tolerance.
    opening = Opening("IMPORT", cash=1000, positions={"S": -5e-9},
                      marks={"S": 1e10}, opening_equity=950, restricted_short_proceeds=50,
                      short_lots=(ShortLot("fractional", "S", 5e-9, 1e10, START),))
    rules = Rules(START, ("S",), 0, funding=Funding(borrow_rates={"S": 0}), borrow=borrow)
    result = Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()).run(
        [Event("inception", START, 0, packet(START, {"S": 1e10}))], PendingDeltaTest())
    assert result.status == "STOP_WITH_BREACH"
    assert result.breaches[0].risk.borrow_failures == ("S",)
    assert result.ledger.equity == 950
