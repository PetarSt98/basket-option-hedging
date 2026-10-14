"""Point-in-time records. All times are timezone-aware; replay time is availability."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol

from .contracts import DataGap, Funding, InputError, Instrument, number, readonly, timestamp


@dataclass(frozen=True)
class Observation:
    id: str
    instrument_id: str
    value: float
    observed_at: datetime
    available_at: datetime

    def __post_init__(self) -> None:
        timestamp(self.observed_at)
        timestamp(self.available_at)
        number(self.value, "observation")
        if not self.id or self.observed_at > self.available_at:
            raise InputError("invalid observation ID/timestamps")


@dataclass(frozen=True)
class Quote:
    id: str
    instrument_id: str
    time: datetime
    available_at: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    def __post_init__(self) -> None:
        timestamp(self.time)
        timestamp(self.available_at)
        for name in ("bid", "ask", "bid_size", "ask_size"):
            number(getattr(self, name), name, minimum=0)
        if not self.id or self.bid > self.ask or self.time > self.available_at:
            raise InputError("invalid/crossed quote or timestamps")


@dataclass(frozen=True)
class Fixing:
    id: str
    instrument_id: str
    observed_at: datetime
    available_at: datetime
    spots: Mapping[str, float]

    def __post_init__(self) -> None:
        timestamp(self.observed_at)
        timestamp(self.available_at)
        if not self.id or self.observed_at > self.available_at:
            raise InputError("invalid fixing ID/timestamps")
        for value in self.spots.values():
            number(value, "fixing spot")
            if value <= 0:
                raise InputError("fixing spots must be positive")
        object.__setattr__(self, "spots", readonly(self.spots))


@dataclass(frozen=True)
class Borrow:
    located: bool
    limit: float

    def __post_init__(self) -> None:
        if not isinstance(self.located, bool):
            raise InputError("borrow locate must be boolean")
        number(self.limit, "borrow limit", minimum=0)


@dataclass(frozen=True)
class ValuationPacket:
    as_of: datetime
    marks: Mapping[str, float]
    stress_marks: Mapping[str, Mapping[str, float]]
    deltas: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    observed_at: Mapping[str, datetime] = field(default_factory=dict)
    input_cutoff: datetime | None = None
    input_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    calibration_cutoff: datetime | None = None

    def __post_init__(self) -> None:
        timestamp(self.as_of)
        if self.calibration_cutoff is not None:
            timestamp(self.calibration_cutoff, "calibration cutoff")
            if self.calibration_cutoff > self.as_of:
                raise InputError("valuation uses future calibration")
        if self.input_cutoff is not None:
            timestamp(self.input_cutoff)
            if self.input_cutoff > self.as_of:
                raise InputError("valuation uses unreleased inputs")
        for key, value in self.marks.items():
            number(value, f"mark {key}", minimum=0)
        for scenario, marks in self.stress_marks.items():
            if not scenario:
                raise InputError("stress scenarios need names")
            for value in marks.values():
                number(value, "stress mark", minimum=0)
        for deltas in self.deltas.values():
            for value in deltas.values():
                number(value, "unit delta")
        for time in self.observed_at.values():
            timestamp(time)
            if time > self.as_of:
                raise InputError("valuation uses future observations")
        for name in ("marks", "stress_marks", "deltas", "observed_at", "metadata"):
            object.__setattr__(self, name, readonly(getattr(self, name)))
        object.__setattr__(self, "input_ids", tuple(self.input_ids))

    def policy_copy(self) -> "ValuationPacket":
        return ValuationPacket(self.as_of, readonly(self.marks), readonly(self.stress_marks),
                               readonly(self.deltas), readonly(self.observed_at), self.input_cutoff,
                               tuple(self.input_ids), readonly(self.metadata), self.calibration_cutoff)


@dataclass(frozen=True)
class Event:
    id: str
    time: datetime
    sequence: int
    valuation: ValuationPacket | None = None
    observations: tuple[Observation, ...] = ()
    quotes: tuple[Quote, ...] = ()
    fixings: tuple[Fixing, ...] = ()
    decision: bool = False
    execution: bool = False
    earliest_execution: datetime | None = None
    order_deadline: datetime | None = None
    terminal_decision: bool = False
    funding: Funding | None = None
    borrow: Mapping[str, Borrow] | None = None
    origination_price: float | None = None
    next_rebalance_time: datetime | None = None
    dividends: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamp(self.time)
        for name in ("decision", "execution", "terminal_decision"):
            if not isinstance(getattr(self, name), bool):
                raise InputError(f"{name} must be boolean")
        if not self.id or isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise InputError("events need ID and nonnegative integer sequence")
        if self.origination_price is not None:
            number(self.origination_price, "origination price", minimum=0)
        for value in (self.earliest_execution, self.order_deadline):
            if value is not None:
                timestamp(value)
        if self.next_rebalance_time is not None:
            timestamp(self.next_rebalance_time, "next scheduled rebalance")
            if self.next_rebalance_time <= self.time:
                raise InputError("next scheduled rebalance must follow the current event")
        if self.decision:
            if (self.earliest_execution is None or self.order_deadline is None
                    or self.earliest_execution <= self.time or self.order_deadline < self.earliest_execution):
                raise InputError("decision requires explicit future execution/deadline")
        if self.terminal_decision and not self.decision:
            raise InputError("terminal decision must also be a decision event")
        for item in (*self.observations, *self.quotes, *self.fixings):
            if item.available_at > self.time:
                raise InputError("event contains unreleased data")
        for name in ("observations", "quotes", "fixings"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.borrow is not None:
            object.__setattr__(self, "borrow", readonly(self.borrow))
        for stock, amount in self.dividends.items():
            if not isinstance(stock, str) or not stock:
                raise InputError("dividends require stock IDs")
            number(amount, "cash dividend per share", minimum=0)
        object.__setattr__(self, "dividends", readonly(self.dividends))


@dataclass(frozen=True)
class Rules:
    horizon: datetime
    hedge_universe: tuple[str, ...]
    safety_buffer: float
    funding: Funding = field(default_factory=Funding)
    borrow: Mapping[str, Borrow] = field(default_factory=dict)
    fee_per_order: float = 0.0
    fee_per_unit: float = 0.0
    slippage_per_unit: float = 0.0
    origination_fee: float = 0.0
    settlement_fee: float = 0.0
    lot_sizes: Mapping[str, float] = field(default_factory=dict)
    maximum_order_quantities: Mapping[str, float] = field(default_factory=dict)
    max_mark_age_seconds: float = 0.0
    start_time: datetime | None = None
    execution_delay_seconds: float = 0.0
    order_lifetime_seconds: float | None = None
    maximum_position_quantities: Mapping[str, float] = field(default_factory=dict)
    expected_stress_scenarios: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        timestamp(self.horizon)
        if self.start_time is not None:
            timestamp(self.start_time, "start time")
            if self.start_time > self.horizon:
                raise InputError("start time must not follow horizon")
        for name in ("safety_buffer", "fee_per_order", "fee_per_unit", "slippage_per_unit",
                     "origination_fee", "settlement_fee", "max_mark_age_seconds", "execution_delay_seconds"):
            number(getattr(self, name), name, minimum=0)
        if self.order_lifetime_seconds is not None:
            number(self.order_lifetime_seconds, "order lifetime")
            if self.order_lifetime_seconds <= self.execution_delay_seconds:
                raise InputError("order lifetime must exceed execution delay")
        if len(set(self.hedge_universe)) != len(self.hedge_universe):
            raise InputError("duplicate hedge instrument")
        for value in (*self.lot_sizes.values(), *self.maximum_order_quantities.values(),
                      *self.maximum_position_quantities.values()):
            number(value, "trading quantity constraint", minimum=0)
        if self.expected_stress_scenarios is not None:
            scenarios = tuple(self.expected_stress_scenarios)
            if (not scenarios or any(not isinstance(name, str) or not name for name in scenarios)
                    or len(set(scenarios)) != len(scenarios) or "unchanged" not in scenarios):
                raise InputError("expected stress scenarios must be unique names including unchanged")
            object.__setattr__(self, "expected_stress_scenarios", scenarios)
        for name in ("borrow", "lot_sizes", "maximum_order_quantities", "maximum_position_quantities"):
            object.__setattr__(self, name, readonly(getattr(self, name)))
        object.__setattr__(self, "hedge_universe", tuple(self.hedge_universe))


@dataclass(frozen=True)
class PricingView:
    """Released inputs only; no portfolio, future feed, or replay random generator."""

    time: datetime
    instruments: Mapping[str, Instrument]
    market: Mapping[str, Observation]
    quotes: Mapping[str, Quote]
    fixings: Mapping[str, Fixing]
    history: tuple[Observation, ...]
    supplied_packet: ValuationPacket | None


class Pricer(Protocol):
    def value(self, view: PricingView) -> ValuationPacket: ...


class SuppliedValuations:
    """Use explicit as-of packets; a timer alone does not refresh old marks."""

    def value(self, view: PricingView) -> ValuationPacket:
        if view.supplied_packet is None:
            raise DataGap("missing valuation packet; clock boundaries require as-of pricing")
        return view.supplied_packet
