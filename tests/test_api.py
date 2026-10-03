"""The public pseudocode-shaped API, with independent economic expectations."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest

from basket_hedging import BacktestConfig, Engine, backtest_engine, no_hedge, stock_delta_hedge
from basket_hedging.contracts import InputError, Instrument, Opening
from basket_hedging.market import Event, Fixing, Observation, Quote, Rules, ValuationPacket
from basket_hedging.reporting import write_reports
from basket_hedging.risk import UnnettedLiabilityStressMargin


START = datetime(2026, 1, 1, 16, tzinfo=timezone.utc)
OPEN = START + timedelta(days=1)
EXPIRY = START + timedelta(days=2)
HORIZON = START + timedelta(days=3)


def call_case():
    stock = Instrument("S", "stock")
    basket = Instrument("B", "basket_call", multiplier=100, strike=100,
                        expiry=EXPIRY, coefficients=(("S", 1),))
    rules = Rules(HORIZON, ("S",), 0, fee_per_order=1, start_time=START,
                  expected_stress_scenarios=("unchanged",))
    events = []
    for i, (time, spot, option, delta) in enumerate([
        (START, 100, 2, .4), (OPEN, 101, 3, .6), (EXPIRY, 105, 5, 1), (HORIZON, 106, 0, 0)
    ]):
        marks = {"S": spot, "B": option}
        valuation = ValuationPacket(time, marks, {"unchanged": marks}, {"B": {"S": delta}},
                                     {key: time for key in marks}, time)
        events.append(Event(f"event/{i}", time, 0, valuation,
                            observations=(Observation(f"spot/{i}", "S", spot, time, time),),
                            quotes=(Quote(f"quote/{i}", "S", time, time, spot-.1, spot+.1, 100, 100),),
                            fixings=(Fixing("fixing/B", "B", time, time, {"S": spot}),) if time == EXPIRY else (),
                            origination_price=2 if time == START else None))
    return basket, events, BacktestConfig(10000, (stock,), rules)


def test_plain_function_policy_and_basket_call_complete_with_independent_balances():
    basket, events, config = call_case()
    result = backtest_engine(basket, iter(events), stock_delta_hedge, config)
    assert result.status == "COMPLETED"
    # Receive 200, buy 40 at 101.1 + fee 1; settle -500; sell at 105.9 - fee 1.
    assert [fill.quantity for fill in result.fills] == [40, -40]
    cash = [snapshot.risk.cash for snapshot in result.snapshots if snapshot.phase == "event"]
    assert cash == pytest.approx([10200, 6155, 5655, 9890])
    pnl = result.ledger.components()
    assert pnl["liability_pnl"] == -300
    assert pnl["hedge_pnl"] == 200
    assert pnl["execution_cost"] == pytest.approx(8)
    assert pnl["fees"] == 2
    assert pnl["net_pnl"] == pytest.approx(-110)
    assert result.ledger.posted_margin_cash == result.ledger.restricted_short_proceeds == 0
    assert result.ledger.positions == {}


def test_no_hedge_callable_still_settles_the_liability_once():
    basket, events, config = call_case()
    result = backtest_engine(basket, events, no_hedge, config)
    assert result.status == "COMPLETED"
    assert result.ledger.cash == 9700  # 10,000 + 200 premium - 500 payoff.
    assert result.fills == ()
    assert len(result.ledger.settlements) == 1


def test_inception_sale_price_is_required_even_with_a_model_mark():
    basket, events, config = call_case()
    events[0] = replace(events[0], origination_price=None)
    result = backtest_engine(basket, events, no_hedge, config)
    assert result.status == "STOP_DATA_GAP"
    assert result.ledger.cash == 10000
    assert not result.ledger.positions
    assert not any(post.kind == "origination" for post in result.ledger.postings)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "one", True])
def test_invalid_policy_quantity_remains_a_serializable_recorded_rejection(tmp_path, bad):
    basket, events, config = call_case()
    result = backtest_engine(basket, events, lambda view: {"S": bad}, config)
    assert result.status == "COMPLETED"
    assert result.fills == ()
    assert any(decision["status"] == "rejected" for decision in result.decisions)
    write_reports([result], tmp_path)
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert summary["current"]["total_cash"] == 9700


def test_public_api_rejects_basket_in_hedge_registry():
    basket, _, config = call_case()
    with pytest.raises(InputError, match="exclude the basket"):
        replace(config, hedge_instruments=(*config.hedge_instruments, basket))


def test_stock_baseline_rounds_shares_to_configured_lot_only():
    basket, events, config = call_case()
    config = replace(config, rules=replace(config.rules, lot_sizes={"S": 3}))
    result = backtest_engine(basket, events, stock_delta_hedge, config)
    # 40 desired shares => nearest 3-share lot = 39; same normal later exit.
    assert [fill.quantity for fill in result.fills] == [39, -39]
    assert result.ledger.cash == pytest.approx(9885.2)


@pytest.mark.parametrize("liability_id", ["S", "unknown", "P"])
def test_imported_liability_identity_cannot_trigger_wrong_completion_or_unwind(liability_id):
    basket, _, config = call_case()
    instruments = {"B": basket, "S": config.hedge_instruments[0],
                   "P": Instrument("P", "put", strike=100, expiry=EXPIRY, underlying="S")}
    opening = Opening("IMPORT", cash=1000, positions={"B": -1}, marks={"B": 2},
                      opening_equity=800, liability_id=liability_id)
    with pytest.raises(InputError, match="liability_id.*basket"):
        Engine(instruments, opening, config.rules, UnnettedLiabilityStressMargin())


def test_imported_explicit_basket_id_requires_an_actual_position():
    basket, _, config = call_case()
    with pytest.raises(InputError, match="held basket"):
        Engine({"B": basket, "S": config.hedge_instruments[0]},
               Opening("IMPORT", cash=1000, opening_equity=1000, liability_id="B"),
               config.rules, UnnettedLiabilityStressMargin())
