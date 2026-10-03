from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from basket_hedging.deterministic import deterministic_case, option_accounting_case
from basket_hedging.contracts import Funding, Instrument, Opening
from basket_hedging.engine import Engine
from basket_hedging.market import Borrow, Event, Fixing, Quote, Rules, ValuationPacket
from basket_hedging.policies import ScheduledDeltaPolicy, UnhedgedPolicy
from basket_hedging.replay import load_case, write_case
from basket_hedging.risk import UnnettedLiabilityStressMargin


def run_saved_case(tmp_path, components, policy):
    instruments, opening, rules, events = components
    path = write_case(tmp_path / "input.json", instruments, opening, rules, events, {"source": "hand-authored"})
    case = load_case(path)
    return Engine(case.instruments, case.opening, case.rules, UnnettedLiabilityStressMargin()).run(case.events(), policy)


def test_completed_three_stock_book_has_independent_expected_balances(tmp_path):
    result = run_saved_case(tmp_path, deterministic_case(), ScheduledDeltaPolicy())
    assert result.status == "COMPLETED"
    # K10000 + premium200; two batches of shorts at bid; settlement400;
    # normal later stock cover at ask. Each three-leg batch pays three in fees.
    cash_by_event = {s.event_id: s.risk.cash for s in result.snapshots if s.phase == "event"}
    assert cash_by_event == pytest.approx({"hand/0": 10200, "hand/1": 13194, "hand/2": 13194,
                                          "hand/3": 16098, "hand/4": 15698, "hand/5": 9629})
    assert result.ledger.positions == {}
    assert result.ledger.restricted_short_proceeds == 0
    assert result.ledger.posted_margin_cash == 0
    pnl = result.ledger.components()
    assert pnl["liability_pnl"] == -200
    assert pnl["hedge_pnl"] == pytest.approx(-150)
    assert pnl["execution_cost"] == pytest.approx(12)
    assert pnl["fees"] == 9
    assert pnl["net_pnl"] == pytest.approx(-371)
    assert pnl["stock_turnover"] == pytest.approx(11970)
    assert len(result.fills) == 9
    assert result.settlement_outcomes[0]["record"].cashflow == -400
    assert result.settlement_outcomes[0]["funded"]
    assert all(abs(s.pnl["reconciliation_error"]) < 1e-8 for s in result.snapshots)


def test_unhedged_same_inputs_has_no_stock_trades_and_retains_liability_loss(tmp_path):
    result = run_saved_case(tmp_path, deterministic_case(), UnhedgedPolicy())
    assert result.status == "COMPLETED"
    assert result.ledger.cash == 9800
    assert result.ledger.components()["net_pnl"] == -200
    assert result.fills == ()


@pytest.mark.parametrize("with_later_record", [False, True])
def test_horizon_execution_can_follow_a_mark_release_at_the_same_time(tmp_path, with_later_record):
    instruments, opening, rules, events = deterministic_case()
    final = events.pop()
    events.extend([
        replace(final, id="horizon-mark", sequence=0, execution=False, quotes=()),
        replace(final, sequence=1),
    ])
    if with_later_record:
        later = final.time + timedelta(days=1)
        events.append(replace(final, id="beyond-horizon", time=later, sequence=0,
                              valuation=None, quotes=(), execution=False))
    result = run_saved_case(tmp_path, (instruments, opening, rules, events), ScheduledDeltaPolicy())
    assert result.status == "COMPLETED"
    assert result.ledger.cash == 9629
    assert result.ledger.positions == {}
    assert len(result.fills) == 9
    assert not any(s.event_id == "beyond-horizon" for s in result.snapshots)
    assert not any(o.outcome == "cancelled" and o.reason == "horizon reached"
                   for o in result.order_outcomes)


@pytest.mark.parametrize("policy", [UnhedgedPolicy(), ScheduledDeltaPolicy()])
def test_gap_fixture_keeps_failed_strategy_and_liability(tmp_path, policy):
    result = run_saved_case(tmp_path, deterministic_case("gap"), policy)
    assert result.status == "STOP_WITH_BREACH"
    assert result.breaches[0].event_id == "hand/1"
    assert result.breaches[0].phase == "pretrade"
    assert result.breaches[0].risk.required_margin == 800
    assert result.breaches[0].risk.equity_requirement_shortfall == 250
    assert result.ledger.cash == 1200
    assert result.ledger.positions == {"BASKET": -1}
    assert not result.fills and not result.settlement_outcomes


@pytest.mark.parametrize("kind", ["missing_exit", "rejected_exit"])
def test_final_exit_failure_retains_marked_stock_inventory(tmp_path, kind):
    result = run_saved_case(tmp_path, deterministic_case(kind), ScheduledDeltaPolicy())
    assert result.status == "INCOMPLETE_EXIT"
    assert result.ledger.cash == 15698
    assert result.ledger.positions == {"S1": -20, "S2": -20, "S3": -20}
    assert result.ledger.restricted_short_proceeds == pytest.approx(5904)
    assert result.ledger.equity == 9638  # cash minus 20*(99+81+123)
    assert len(result.fills) == 6


@pytest.mark.parametrize("kind", ["put", "call"])
def test_scripted_constituent_option_purchase_uses_normal_execution_and_settlement(tmp_path, kind):
    class ScriptedTestTargets:
        name = "scripted option accounting test"

        def targets(self, view):
            return {"OPTION": 1}

    result = run_saved_case(tmp_path, option_accounting_case(kind), ScriptedTestTargets())
    assert result.status == "COMPLETED"
    assert result.fills[0].cash_change == -21
    assert result.settlement_outcomes[0]["record"].cashflow == 50
    assert result.ledger.cash == 1028
    pnl = result.ledger.components()
    assert pnl["hedge_pnl"] == 30
    assert pnl["execution_cost"] == 1
    assert pnl["fees"] == 1
    assert pnl["net_pnl"] == 28


def test_missing_delta_skips_only_decision_and_never_reuses_old_target(tmp_path):
    instruments, opening, rules, events = deterministic_case()
    events[0] = replace(events[0], valuation=replace(events[0].valuation, deltas={}))
    result = run_saved_case(tmp_path, (instruments, opening, rules, events), ScheduledDeltaPolicy())
    assert result.status == "COMPLETED"
    assert result.decisions[0]["status"] == "skipped"
    assert not any(fill.time == events[1].time for fill in result.fills)


def test_policy_cannot_trade_the_basket_even_with_valid_mark(tmp_path):
    class InvalidTestTargets:
        name = "invalid test target"

        def targets(self, view):
            return {"BASKET": 0}

    result = run_saved_case(tmp_path, deterministic_case(), InvalidTestTargets())
    assert result.status == "COMPLETED"
    assert not result.fills
    assert any(x.outcome == "rejected" and x.instrument_id == "BASKET" for x in result.order_outcomes)
    assert result.settlement_outcomes[0]["record"].cashflow == -400


def test_mixed_book_cash_equity_and_pnl_match_hand_calculation(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(days=day) for day in range(5)]
    instruments = {
        "S": Instrument("S", "stock"),
        "B": Instrument("B", "basket_put", multiplier=10, strike=100,
                        expiry=times[3], coefficients=(("S", 1),)),
        "P": Instrument("P", "put", multiplier=10, strike=95,
                        expiry=times[3], underlying="S"),
    }

    def valuation(time, marks):
        return ValuationPacket(time, marks, {"unchanged": marks},
                               observed_at={key: time for key in marks})

    def quote(key, time, bid, ask):
        return Quote(f"{key}/{time.isoformat()}", key, time, time, bid, ask, 100, 100)

    events = [
        Event("start", times[0], 0, valuation(times[0], {"S": 100, "B": 2, "P": 1}),
              origination_price=2),
        Event("enter", times[1], 0, valuation(times[1], {"S": 100, "B": 2, "P": 1}),
              quotes=(quote("S", times[1], 99, 101), quote("P", times[1], .9, 1.1)),
              funding=Funding(cash_rate=.365, borrow_rates={"S": .365})),
        Event("adverse", times[2], 0, valuation(times[2], {"S": 90, "B": 12, "P": 5}),
              quotes=(quote("S", times[2], 89, 91),)),
        Event("expiry", times[3], 0, valuation(times[3], {"S": 90}),
              fixings=(Fixing("fix-B", "B", times[3], times[3], {"S": 90}),
                       Fixing("fix-P", "P", times[3], times[3], {"S": 90}))),
        Event("exit", times[4], 0, valuation(times[4], {"S": 90}),
              quotes=(quote("S", times[4], 89, 91),)),
    ]

    def scripted_targets(view):
        return {"start": {"S": -2, "P": 1}, "enter": {"S": -1, "P": 1}}.get(view.event_id)

    opening = Opening("ORIGINATE", initial_capital=1000, liability_id="B")
    rules = Rules(times[4], ("S", "P"), 0, fee_per_order=1,
                  funding=Funding(borrow_rates={"S": 0}), borrow={"S": Borrow(True, 10)},
                  expected_stress_scenarios=("unchanged",))
    result = run_saved_case(tmp_path, (instruments, opening, rules, events), scripted_targets)
    # Cash/equity/Q/R after each event. Origination adds 20 cash and -20 liability.
    # Entry: buy put for 11, short stock for 198, pay two fees. First accrual is zero.
    # Later daily interest: .987, .894787, .965591787; borrow: .2, .09, .09.
    expected = {
        "start": (1020, 1000, 0, 20),
        "enter": (1205, 995, 198, 20),
        "adverse": (1113.787, 953.787, 99, 120),
        "expiry": (1064.591787, 974.591787, 99, 0),
        "exit": (973.467378787, 973.467378787, 0, 0),
    }
    checkpoints = [snapshot for snapshot in result.snapshots if snapshot.phase == "event"]
    assert [snapshot.event_id for snapshot in checkpoints] == list(expected)
    for snapshot in checkpoints:
        risk = snapshot.risk
        assert (risk.cash, risk.equity, risk.restricted_short_proceeds,
                risk.posted_margin_cash) == pytest.approx(expected[snapshot.event_id], rel=0, abs=1e-9)
    # Liability: 20-100=-80. Gross stock P&L20 plus put P&L(50-10)=40.
    components = result.ledger.components()
    for key, value in {"liability_pnl": -80, "hedge_pnl": 60, "execution_cost": 5,
                       "fees": 4, "interest": 2.847378787, "borrow_charges": .38,
                       "net_pnl": -26.532621213}.items():
        assert components[key] == pytest.approx(value, rel=0, abs=1e-9)
    assert result.status == "COMPLETED" and result.ledger.positions == {}
    assert len(result.fills) == 4
    assert sorted(item["record"].cashflow for item in result.settlement_outcomes) == [-100, 50]
    assert all(item["funded"] for item in result.settlement_outcomes)
    assert result.ledger.restricted_short_proceeds == result.ledger.posted_margin_cash == 0
