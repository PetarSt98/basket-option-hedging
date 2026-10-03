import json
from datetime import datetime, timedelta, timezone

import pytest

from basket_hedging.contracts import InputError, Instrument, Opening
from basket_hedging.engine import Engine
from basket_hedging.market import Event, Rules, ValuationPacket
from basket_hedging.replay import load_case, write_case


class ZeroTestMargin:
    label = "test-only fixed margin"

    def required_margin(self, ledger, valuation):
        return 0.0


class NoOrders:
    name = "test input stream"

    def targets(self, view):
        return {}


def test_jsonl_roundtrip_and_engine_does_not_read_beyond_stop(tmp_path):
    time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    instruments = {"S": Instrument("S", "stock")}
    opening = Opening("IMPORT", cash=-1, opening_equity=-1)
    rules = Rules(time + timedelta(days=1), ("S",), 0)
    packet = ValuationPacket(time, {"S": 100}, {"unchanged": {"S": 100}},
                             observed_at={"S": time}, input_ids=("manual",))
    path = write_case(tmp_path / "case.json", instruments, opening, rules,
                      [Event("opening", time, 0, packet)], {"source": "hand-authored"})
    case = load_case(path)
    with case.events_path.open("a", encoding="utf-8") as output:
        output.write("not valid future JSON\n")
    result = Engine(case.instruments, case.opening, case.rules, ZeroTestMargin()).run(case.events(), NoOrders())
    assert result.status == "STOP_WITH_BREACH"
    assert result.ledger.cash == -1
    assert result.baseline == -1
    assert result.last_valid_snapshot.pnl["net_pnl"] == 0
    # Fully consuming the same file does encounter the intentionally bad future row.
    with pytest.raises(InputError, match="case.jsonl:2"):
        list(case.events())


def test_loader_rejects_nonfinite_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1, "x": NaN}')
    with pytest.raises(InputError, match="nonfinite"):
        load_case(path)


def test_malformed_financial_payload_beyond_horizon_raises_without_releasing_it(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    horizon = start + timedelta(days=1)
    instruments = {"S": Instrument("S", "stock")}
    events = []
    for identifier, time, sequence in (("opening", start, 0),
                                        ("horizon", horizon, 0),
                                        ("horizon-later", horizon, 1)):
        packet = ValuationPacket(time, {"S": 100}, {"unchanged": {"S": 100}},
                                 observed_at={"S": time})
        events.append(Event(identifier, time, sequence, packet))
    path = write_case(tmp_path / "case.json", instruments,
                      Opening("IMPORT", cash=1000, opening_equity=1000),
                      Rules(horizon, ("S",), 0), events, {"source": "hand-authored"})
    case = load_case(path)
    completed = Engine(case.instruments, case.opening, case.rules, ZeroTestMargin()).run(
        case.events(), NoOrders())
    assert completed.status == "COMPLETED"

    future = (horizon + timedelta(days=30)).isoformat()
    # The availability and JSON are valid; only the future financial payload is malformed.
    malformed = dict(id="future", time=future, sequence=0,
                     valuation=dict(as_of=future, marks={"S": -1},
                                    stress_marks={"unchanged": {"S": -1}},
                                    observed_at={"S": future}))
    with case.events_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(malformed) + "\n")
    seen = []

    def observe(view):
        seen.append((view.event_id, view.time, dict(view.valuation.marks)))
        return None

    engine = Engine(case.instruments, case.opening, case.rules, ZeroTestMargin())
    with pytest.raises(InputError, match=r"case\.jsonl:4:.*mark S"):
        engine.run(case.events(), observe)
    assert seen == [("opening", start, {"S": 100}),
                    ("horizon", horizon, {"S": 100}),
                    ("horizon-later", horizon, {"S": 100})]
    assert engine.ledger.cash == 1000
    assert engine.ledger.positions == {}
