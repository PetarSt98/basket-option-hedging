"""One shared feasibility calculation; margin requirement is replaceable in-process."""

from dataclasses import dataclass
from typing import Mapping, Protocol

from .contracts import ABS_TOL, QUANTITY_TOL, DataGap, InputError, close, number
from .ledger import Ledger
from .market import Borrow, ValuationPacket


class MarginModel(Protocol):
    label: str

    def required_margin(self, ledger: Ledger, valuation: ValuationPacket) -> float: ...


class UnnettedLiabilityStressMargin:
    """ULSCP: gross short-option marked liability plus largest portfolio stress loss."""

    label = "uncalibrated research margin proxy"
    name = "Unnetted Liability plus Stress Cash Proxy"

    def __init__(self, expected_scenarios: tuple[str, ...] | None = None):
        self.expected_scenarios = tuple(expected_scenarios) if expected_scenarios is not None else ("unchanged",)
        if (not self.expected_scenarios or "unchanged" not in self.expected_scenarios
                or any(not isinstance(name, str) or not name for name in self.expected_scenarios)
                or len(set(self.expected_scenarios)) != len(self.expected_scenarios)):
            raise InputError("declared margin scenarios must have unique names and include unchanged")

    def required_margin(self, ledger: Ledger, valuation: ValuationPacket) -> float:
        missing = set(self.expected_scenarios) - valuation.stress_marks.keys()
        if missing:
            raise DataGap(f"missing declared margin stress scenarios: {', '.join(sorted(missing))}")
        for key in ledger.positions:
            if key not in valuation.marks or any(key not in marks for marks in valuation.stress_marks.values()):
                raise DataGap(f"missing margin mark/stress input for {key}")
            if not close(valuation.stress_marks["unchanged"][key], valuation.marks[key]):
                raise InputError("unchanged scenario must equal the accounting marks")
            if not close(ledger.marks[key], valuation.marks[key]):
                raise InputError("margin and ledger must use the same accounting marks")
        values = ledger.position_values()
        portfolio_value = sum(values.values())
        liability_reserve = sum(max(-value, 0.0) for key, value in values.items()
                                if ledger.instruments[key].is_option)
        stress_loss = max([0.0] + [portfolio_value - sum(
            quantity * ledger.instruments[key].multiplier * marks[key]
            for key, quantity in ledger.positions.items()) for marks in valuation.stress_marks.values()])
        return liability_reserve + stress_loss


@dataclass(frozen=True)
class Risk:
    equity: float | None
    cash: float
    posted_margin_cash: float
    restricted_short_proceeds: float
    available_cash: float
    eligible_cash_capacity: float
    required_margin: float | None
    safety_buffer: float
    margin_equity_shortfall: float | None
    margin_cash_shortfall: float | None
    buffer_equity_shortfall: float | None
    buffer_cash_shortfall: float | None
    equity_requirement_shortfall: float | None
    cash_requirement_shortfall: float | None
    available_cash_deficit: float
    equity_deficit: float | None
    unencumbered_cash_deficit: float
    borrow_failures: tuple[str, ...]
    maximum_shortfall: float | None
    feasible: bool
    position_failures: tuple[str, ...] = ()
    cash_requirement_shortfall_lower_bound: float = 0.0


def borrow_failures(ledger: Ledger, borrow: Mapping[str, Borrow]) -> tuple[str, ...]:
    failures = []
    for key, quantity in sorted(ledger.positions.items()):
        if ledger.instruments[key].kind == "stock" and quantity < -QUANTITY_TOL:
            info = borrow.get(key)
            if info is None or not info.located or -quantity > info.limit + QUANTITY_TOL:
                failures.append(key)
    return tuple(failures)


def assess(ledger: Ledger, margin: float | None, safety_buffer: float,
           borrow: Mapping[str, Borrow], *, valued: bool = True,
           position_limits: Mapping[str, float] | None = None) -> Risk:
    number(safety_buffer, "safety buffer", minimum=0)
    if margin is not None:
        number(margin, "required margin", minimum=0)
    if valued and margin is None:
        raise InputError("valued risk requires a margin requirement")
    equity = ledger.equity if valued else None
    cash_capacity = ledger.cash - ledger.restricted_short_proceeds
    failures = borrow_failures(ledger, borrow)
    exceeded = []
    for key, limit in sorted((position_limits or {}).items()):
        if key not in ledger.instruments:
            raise InputError(f"position limit references unknown instrument {key}")
        number(limit, "maximum absolute position", minimum=0)
        if abs(ledger.positions.get(key, 0.0)) > limit + QUANTITY_TOL:
            exceeded.append(key)
    margin_equity = max(margin - equity, 0.0) if valued else None
    margin_cash = max(margin - cash_capacity, 0.0) if valued else None
    total_equity = max(margin + safety_buffer - equity, 0.0) if valued else None
    total_cash = max(margin + safety_buffer - cash_capacity, 0.0) if valued else None
    maximum = max(total_equity, total_cash) if valued else None
    # Margin is nonnegative: this cash requirement can fail even without marks.
    # Keep the unknown full margin/shortfall separate from its known lower bound.
    cash_lower_bound = total_cash if valued else max(safety_buffer - cash_capacity, 0.0)
    feasible = ((maximum <= ABS_TOL) if valued else (cash_lower_bound <= ABS_TOL)) and not failures and not exceeded
    return Risk(equity=equity, cash=ledger.cash, posted_margin_cash=ledger.posted_margin_cash,
                restricted_short_proceeds=ledger.restricted_short_proceeds,
                available_cash=ledger.available_cash, eligible_cash_capacity=max(cash_capacity, 0.0),
                required_margin=margin, safety_buffer=safety_buffer,
                margin_equity_shortfall=margin_equity, margin_cash_shortfall=margin_cash,
                buffer_equity_shortfall=total_equity - margin_equity if valued else None,
                buffer_cash_shortfall=total_cash - margin_cash if valued else None,
                equity_requirement_shortfall=total_equity, cash_requirement_shortfall=total_cash,
                available_cash_deficit=max(-ledger.available_cash, 0.0),
                equity_deficit=max(-equity, 0.0) if valued else None,
                unencumbered_cash_deficit=max(-cash_capacity, 0.0), borrow_failures=failures,
                maximum_shortfall=maximum, feasible=feasible, position_failures=tuple(exceeded),
                cash_requirement_shortfall_lower_bound=cash_lower_bound)
