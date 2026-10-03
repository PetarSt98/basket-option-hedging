"""Persistent whole orders with immediate cancellation and deterministic priority."""

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .contracts import QUANTITY_TOL


@dataclass(frozen=True)
class Order:
    id: str
    instrument_id: str
    quantity: float
    submitted_at: datetime
    earliest_execution: datetime
    deadline: datetime
    terminal: bool = False  # Includes its deadline's execution phase.


class OrderQueue:
    def __init__(self) -> None:
        self.pending: dict[str, Order] = {}

    def snapshot(self) -> tuple[Order, ...]:
        return tuple(sorted(self.pending.values(),
                            key=lambda order: (order.instrument_id, order.submitted_at, order.id)))

    def eligible_at(self, time: datetime) -> Iterable[Order]:
        return (order for order in self.snapshot()
                if time > order.submitted_at and time >= order.earliest_execution
                and (time <= order.deadline if order.terminal else time < order.deadline))

    def signed_pending_quantity(self, instrument_id: str) -> float:
        return sum(order.quantity for order in self.pending.values()
                   if order.instrument_id == instrument_id)

    def cancel_obsolete(self, instrument_id: str, required: float) -> tuple[Order, ...]:
        """Keep compatible whole orders; replace excessive orders rather than resize."""
        cancelled = []
        remaining = required
        for order in self.snapshot():
            if order.instrument_id != instrument_id:
                continue
            if order.quantity * remaining > 0 and abs(order.quantity) <= abs(remaining) + QUANTITY_TOL:
                remaining -= order.quantity
            else:
                cancelled.append(self.pending.pop(order.id))
        return tuple(cancelled)
