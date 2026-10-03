from datetime import datetime, timedelta, timezone

import pytest

from basket_hedging.contracts import InputError
from basket_hedging.engine import Engine
from basket_hedging.policies import ScheduledDeltaPolicy, UnhedgedPolicy
from basket_hedging.replay import jsonable, load_case
from basket_hedging.risk import UnnettedLiabilityStressMargin
from basket_hedging.synthetic import generate_synthetic_case, market_path


def run(case, policy):
    return Engine(case.instruments, case.opening, case.rules, UnnettedLiabilityStressMargin()).run(case.events(), policy)


def test_pricing_sample_count_and_seed_do_not_change_market_replay(tmp_path):
    first = load_case(generate_synthetic_case(tmp_path / "one.json", samples=256, days=2))
    second = load_case(generate_synthetic_case(tmp_path / "two.json", samples=512, pricing_seed=999, days=2))
    rows_one, rows_two = list(first.events()), list(second.events())
    assert [row.observations for row in rows_one] == [row.observations for row in rows_two]
    assert [row.quotes for row in rows_one] == [row.quotes for row in rows_two]
    assert [row.fixings for row in rows_one] == [row.fixings for row in rows_two]
    assert rows_one[0].valuation.marks["BASKET"] != rows_two[0].valuation.marks["BASKET"]


def test_three_stock_fixture_repeats_financial_outputs_and_matches_policy_valuations(tmp_path):
    first_path = generate_synthetic_case(tmp_path / "one.json", samples=512, days=3)
    second_path = generate_synthetic_case(tmp_path / "two.json", samples=512, days=3)
    first, second = load_case(first_path), load_case(second_path)
    assert first.events_path.read_bytes() == second.events_path.read_bytes()
    unhedged = run(first, UnhedgedPolicy())
    delta = run(first, ScheduledDeltaPolicy())
    repeat = run(second, ScheduledDeltaPolicy())
    assert unhedged.status == delta.status == repeat.status == "COMPLETED"
    assert unhedged.valuations == delta.valuations
    assert delta.ledger.postings == repeat.ledger.postings
    assert delta.decisions == repeat.decisions
    assert delta.snapshots == repeat.snapshots
    assert len(delta.fills) > 0
    assert delta.ledger.positions == {}


def test_ten_constituent_smoke_uses_the_same_engine_and_interfaces(tmp_path):
    case = load_case(generate_synthetic_case(tmp_path / "ten.json", constituents=10, days=2, samples=256))
    result = run(case, ScheduledDeltaPolicy())
    assert len(case.rules.hedge_universe) == 10
    assert result.status == "COMPLETED"
    assert result.ledger.positions == {}
    assert len({fill.instrument_id for fill in result.fills}) == 10
    assert abs(result.ledger.components()["reconciliation_error"]) < 1e-8


def test_market_generator_does_not_repair_invalid_inputs():
    time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(InputError, match="positive"):
        list(market_path(("S",), (0,), (0.2,), ((1,),), (time,), 1))
    with pytest.raises(InputError, match="correlation"):
        list(market_path(("A", "B"), (100, 100), (0.2, 0.2), ((1, 2), (2, 1)), (time,), 1))
    with pytest.raises(InputError, match="strictly increase"):
        list(market_path(("S",), (100,), (0.2,), ((1,),), (time, time), 1))
