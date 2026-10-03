"""Hand-authored fixtures: prices/Greeks are test inputs, not model forecasts."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .contracts import Funding, Instrument, Opening
from .market import Borrow, Event, Fixing, Observation, Quote, Rules, ValuationPacket
from .replay import write_case


def hand_packet(time: datetime, marks: dict[str, float], deltas: dict[str, dict[str, float]],
                stock_ids: tuple[str, ...], *, source: str) -> ValuationPacket:
    down = {key: value * 0.9 if key in stock_ids else value + 2 for key, value in marks.items()}
    up = {key: value * 1.1 if key in stock_ids else max(value - 2, 0) for key, value in marks.items()}
    return ValuationPacket(time, marks, {"unchanged": marks.copy(), "toy_down": down, "toy_up": up},
                           deltas, {key: time for key in marks}, time, (source,),
                           {"model": "hand-authored ledger fixture", "source": source})


def deterministic_case(kind: str = "complete") -> tuple[dict[str, Instrument], Opening, Rules, list[Event]]:
    if kind not in {"complete", "gap", "missing_exit", "rejected_exit"}:
        raise ValueError("unknown deterministic fixture")
    stocks = ("S1", "S2", "S3")
    times = [datetime(2026, 1, day, hour, tzinfo=timezone.utc)
             for day, hour in ((5, 16), (6, 9), (6, 16), (7, 9), (7, 16), (8, 9))]
    expiry, horizon = times[-2:]
    instruments = {key: Instrument(key, "stock") for key in stocks}
    instruments["BASKET"] = Instrument("BASKET", "basket_put", multiplier=100, strike=100,
                                        expiry=expiry, coefficients=tuple((key, 1 / 3) for key in stocks))
    opening = Opening("ORIGINATE", initial_capital=1000 if kind == "gap" else 10000,
                      liability_id="BASKET", liability_quantity=-1)
    rules = Rules(horizon, stocks, 50, Funding(0, 0, {key: 0 for key in stocks}),
                  {key: Borrow(True, 1000) for key in stocks}, fee_per_order=1,
                  maximum_order_quantities={key: 1000 for key in stocks}, start_time=times[0],
                  expected_stress_scenarios=("unchanged", "toy_down", "toy_up"))
    prices = [(100, 80, 120), (99, 81, 120), (96, 78, 120),
              (95, 77, 119), (94, 76, 118), (99, 81, 123)]
    basket_marks = [2, 6 if kind == "gap" else 2, 3, 3.5, 4, 0]
    events = []
    for index, (time, prices_now, mark) in enumerate(zip(times, prices, basket_marks)):
        spots = dict(zip(stocks, prices_now))
        marks = {**spots, "BASKET": mark}
        decision = index in (0, 2, 4)
        execution = index in (1, 3, 5)
        delta = -0.1 if index < 2 else -0.2
        packet = hand_packet(time, marks, {"BASKET": {key: delta for key in stocks}}, stocks,
                             source=f"hand/{kind}/{index}")
        quotes = tuple(Quote(f"q/{index}/{key}", key, time, time, value - 0.1, value + 0.1,
                             1000, 0 if kind == "rejected_exit" and index == 5 else 1000)
                       for key, value in spots.items()) if execution else ()
        if kind == "missing_exit" and index == 5:
            quotes = ()
        fixings = (Fixing("official/BASKET", "BASKET", expiry, expiry, spots),) if index == 4 else ()
        events.append(Event(f"hand/{index}", time, index, packet,
                            observations=tuple(Observation(f"spot/{index}/{key}", key, value, time, time)
                                               for key, value in spots.items()),
                            quotes=quotes, fixings=fixings, decision=decision, execution=execution,
                            earliest_execution=times[index + 1] if decision else None,
                            order_deadline=(times[index + 2] if index < 4 else horizon) if decision else None,
                            terminal_decision=index == 4, origination_price=2 if index == 0 else None))
    return instruments, opening, rules, events


def option_accounting_case(kind: str = "put") -> tuple[dict[str, Instrument], Opening, Rules, list[Event]]:
    """The test supplies scripted option targets; this function is only input data."""
    start = datetime(2026, 1, 5, 16, tzinfo=timezone.utc)
    expiry = start + timedelta(days=1)
    horizon = expiry + timedelta(hours=17)
    times = (start, start + timedelta(hours=17), expiry, horizon)
    instruments = {"S": Instrument("S", "stock"),
                   "OPTION": Instrument("OPTION", kind, multiplier=10, strike=100, expiry=expiry, underlying="S")}
    opening = Opening("IMPORT", cash=1000, opening_equity=1000)
    rules = Rules(horizon, ("OPTION",), 0, fee_per_order=1, start_time=start,
                  expected_stress_scenarios=("unchanged", "toy_down", "toy_up"))
    events = []
    for index, time in enumerate(times):
        spot = 100 if index < 2 else (95 if kind == "put" else 105)
        marks = {"S": spot, "OPTION": 2 if index < 2 else (5 if index == 2 else 0)}
        packet = hand_packet(time, marks, {"OPTION": {"S": -0.4 if kind == "put" else 0.4}}, ("S",),
                             source=f"option-test/{kind}/{index}")
        events.append(Event(f"option/{index}", time, index, packet,
                            quotes=(Quote("option-quote", "OPTION", time, time, 1.9, 2.1, 10, 10),) if index == 1 else (),
                            fixings=(Fixing("official/OPTION", "OPTION", time, time, {"S": spot}),) if index == 2 else (),
                            decision=index == 0, execution=index in (1, 3),
                            earliest_execution=times[1] if index == 0 else None,
                            order_deadline=expiry if index == 0 else None))
    return instruments, opening, rules, events


def write_deterministic_cases(directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    paths = {}
    for kind in ("complete", "gap", "missing_exit", "rejected_exit"):
        instruments, opening, rules, events = deterministic_case(kind)
        paths[kind] = write_case(directory / f"deterministic_{kind}.json", instruments, opening, rules, events,
                                 {"source": "hand-authored deterministic fixture v1", "kind": kind,
                                  "margin_label": "uncalibrated research margin proxy",
                                  "stress": "unchanged; stocks +/-10%, option marks +/-2 floored at zero",
                                  "purpose": "accounting and timing only; no profitability inference"})
    return paths
