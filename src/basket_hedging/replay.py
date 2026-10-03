"""Small JSON configuration plus streaming JSONL market records; no simulation required."""

import json
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol

from .contracts import Funding, InputError, Instrument, Opening, ShortLot, timestamp
from .market import Borrow, Event, Fixing, Observation, Quote, Rules, ValuationPacket


class _ScheduledOrder(Protocol):
    earliest_execution: datetime
    deadline: datetime


def replay_clock(events: Iterable[Event], pending_orders: Callable[[], Iterable[_ScheduledOrder]],
                 instruments: Mapping[str, Instrument], rules: Rules) -> Iterator[Event]:
    """Merge released records with live order and contractual clock boundaries.

    Source records keep their availability/sequence order. The next source row
    is read only after the current row has been yielded: a stopped consumer need
    not parse later data. Clock-only records carry no invented marks or quotes.
    """
    source = iter(events)
    following = next(source, None)
    start = rules.start_time or (min(following.time, rules.horizon) if following else rules.horizon)
    fixed_times = {rules.horizon, *(instrument.expiry for instrument in instruments.values()
                                   if instrument.expiry is not None and start <= instrument.expiry <= rules.horizon)}
    previous: datetime | None = None
    while True:
        if following is not None and following.time < (previous if previous is not None else start):
            raise InputError("replay records precede the start or move backward in availability time")
        # Consume every source sequence at H before finalizing; never add an
        # extra H timer after an actual source record already processed H.
        if previous == rules.horizon and (following is None or following.time > rules.horizon):
            return
        boundary = previous if previous is not None else start
        candidates = [time for time in fixed_times if time > boundary]
        if previous is None:
            candidates.append(start)
        for order in pending_orders():
            candidates.extend(time for time in (order.earliest_execution, order.deadline)
                              if boundary < time <= rules.horizon)
        if following is not None and following.time <= rules.horizon:
            candidates.append(following.time)
        if not candidates:
            return
        time = min(candidates)
        previous = time
        if following is not None and following.time == time:
            yield following
            following = next(source, None)
        else:
            yield Event(f"clock/{time.isoformat()}", time, 0, None)


def jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _invalid_constant(value: str) -> None:
    raise InputError(f"nonfinite JSON number {value}")


def _loads(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_invalid_constant)
    except (json.JSONDecodeError, TypeError) as error:
        raise InputError(f"invalid JSON: {error}") from error


def _time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise InputError(f"invalid timestamp {value!r}") from error
    timestamp(parsed)
    return parsed


def parse_event(record: Mapping[str, Any]) -> Event:
    data = dict(record)
    for key in ("time", "earliest_execution", "order_deadline", "next_rebalance_time"):
        if data.get(key) is not None:
            data[key] = _time(data[key])
    for key, constructor, time_keys in (
        ("observations", Observation, ("observed_at", "available_at")),
        ("quotes", Quote, ("time", "available_at")),
        ("fixings", Fixing, ("observed_at", "available_at")),
    ):
        items = []
        for raw in data.get(key, []):
            item = dict(raw)
            for time_key in time_keys:
                item[time_key] = _time(item[time_key])
            items.append(constructor(**item))
        data[key] = tuple(items)
    if data.get("valuation") is not None:
        valuation = dict(data["valuation"])
        valuation["as_of"] = _time(valuation["as_of"])
        for key in ("input_cutoff", "calibration_cutoff"):
            if valuation.get(key) is not None:
                valuation[key] = _time(valuation[key])
        valuation["observed_at"] = {key: _time(value) for key, value in valuation.get("observed_at", {}).items()}
        valuation["input_ids"] = tuple(valuation.get("input_ids", []))
        data["valuation"] = ValuationPacket(**valuation)
    if data.get("funding") is not None:
        data["funding"] = Funding(**data["funding"])
    if data.get("borrow") is not None:
        data["borrow"] = {key: Borrow(**value) for key, value in data["borrow"].items()}
    try:
        return Event(**data)
    except (TypeError, KeyError) as error:
        raise InputError(f"invalid event fields: {error}") from error


@dataclass(frozen=True)
class Case:
    instruments: Mapping[str, Instrument]
    opening: Opening
    rules: Rules
    events_path: Path
    metadata: Mapping[str, Any]

    def events(self) -> Iterator[Event]:
        with self.events_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        yield parse_event(_loads(line))
                    except (InputError, TypeError, KeyError, ValueError) as error:
                        raise InputError(f"{self.events_path.name}:{line_number}: {error}") from error


def load_case(path: str | Path) -> Case:
    path = Path(path)
    data = _loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise InputError("unsupported case schema version")
    try:
        instruments = {}
        for raw in data["instruments"]:
            item = dict(raw)
            if item.get("expiry") is not None:
                item["expiry"] = _time(item["expiry"])
            item["coefficients"] = tuple(tuple(pair) for pair in item.get("coefficients", []))
            instrument = Instrument(**item)
            if instrument.id in instruments:
                raise InputError("duplicate instrument ID")
            instruments[instrument.id] = instrument
        opening_data = dict(data["opening"])
        opening_data["short_lots"] = tuple(ShortLot(**{**lot, "opened_at": _time(lot["opened_at"])})
                                           for lot in opening_data.get("short_lots", []))
        opening = Opening(**opening_data)
        rules_data = dict(data["rules"])
        rules_data["horizon"] = _time(rules_data["horizon"])
        if rules_data.get("start_time") is not None:
            rules_data["start_time"] = _time(rules_data["start_time"])
        rules_data["hedge_universe"] = tuple(rules_data["hedge_universe"])
        rules_data["funding"] = Funding(**rules_data.get("funding", {}))
        rules_data["borrow"] = {key: Borrow(**value) for key, value in rules_data.get("borrow", {}).items()}
        return Case(instruments, opening, Rules(**rules_data), path.parent / data["events_file"], data.get("metadata", {}))
    except (TypeError, KeyError) as error:
        raise InputError(f"invalid case fields: {error}") from error


def write_case(path: str | Path, instruments: Mapping[str, Instrument], opening: Opening,
               rules: Rules, events: Iterable[Event], metadata: Mapping[str, Any]) -> Path:
    path = Path(path)
    events_path = path.with_suffix(".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("w", encoding="utf-8", newline="\n") as target:
        for event in events:
            target.write(json.dumps(jsonable(event), separators=(",", ":"), allow_nan=False) + "\n")
    write_json(path, dict(schema_version=1, instruments=list(instruments.values()), opening=opening,
                         rules=rules, events_file=events_path.name, metadata=metadata))
    return path
