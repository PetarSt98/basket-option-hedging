"""Explicit instrument units, opening records and numerical validation."""

from dataclasses import dataclass, field
from datetime import datetime
from math import isclose, isfinite
from types import MappingProxyType
from typing import Any, Literal, Mapping

ABS_TOL = 1e-8
REL_TOL = 1e-10
QUANTITY_TOL = 1e-12


class InputError(ValueError):
    """Malformed or inconsistent input (not financial infeasibility)."""


class DataGap(ValueError):
    """Required point-in-time information is unavailable."""


class ReconciliationError(ArithmeticError):
    """Independently accumulated P&L does not reconcile with equity."""


def number(value: float, name: str, *, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise InputError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise InputError(f"{name} must be >= {minimum}")


def timestamp(value: datetime, name: str = "time") -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise InputError(f"{name} must be timezone-aware")


def close(left: float, right: float) -> bool:
    return isclose(left, right, abs_tol=ABS_TOL, rel_tol=REL_TOL)


def readonly(value: Any) -> Any:
    """Detach and freeze nested input containers, including their source aliases."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: readonly(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(readonly(item) for item in value)
    return value


@dataclass(frozen=True)
class Instrument:
    id: str
    kind: Literal["stock", "put", "call", "basket_put", "basket_call"]
    currency: str = "USD"
    multiplier: float = 1.0
    strike: float | None = None
    expiry: datetime | None = None
    underlying: str | None = None
    coefficients: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", tuple(tuple(pair) for pair in self.coefficients))
        if not self.id or not self.currency or self.kind not in {"stock", "put", "call", "basket_put", "basket_call"}:
            raise InputError("invalid instrument identity/kind/currency")
        number(self.multiplier, "multiplier")
        if self.multiplier <= 0:
            raise InputError("multiplier must be positive")
        if self.kind == "stock":
            if self.multiplier != 1 or any(x is not None for x in (self.strike, self.expiry, self.underlying)) or self.coefficients:
                raise InputError("stocks use signed shares, multiplier 1, and no option terms")
            return
        if self.strike is None or self.expiry is None:
            raise InputError("options require strike and expiry")
        number(self.strike, "strike", minimum=0)
        timestamp(self.expiry, "expiry")
        if self.is_basket:
            if self.underlying is not None or not self.coefficients:
                raise InputError("basket requires fixed coefficients, not a single underlying")
            if len(dict(self.coefficients)) != len(self.coefficients):
                raise InputError("duplicate basket constituent")
            for _, coefficient in self.coefficients:
                number(coefficient, "basket coefficient", minimum=0)
            if sum(a for _, a in self.coefficients) <= 0:
                raise InputError("basket must contain positive exposure")
        elif not self.underlying or self.coefficients:
            raise InputError("constituent options require one underlying")

    @property
    def is_option(self) -> bool:
        return self.kind != "stock"

    @property
    def is_basket(self) -> bool:
        return self.kind in {"basket_put", "basket_call"}

    @property
    def risk_stocks(self) -> tuple[str, ...]:
        if self.kind == "stock":
            return (self.id,)
        if self.is_basket:
            return tuple(i for i, _ in self.coefficients)
        assert self.underlying is not None
        return (self.underlying,)

    def payoff(self, official_spots: Mapping[str, float]) -> float:
        if not self.is_option:
            raise InputError("stocks have no option payoff")
        for stock in self.risk_stocks:
            if stock not in official_spots:
                raise DataGap(f"missing official fixing for {stock}")
            number(official_spots[stock], "official spot")
            if official_spots[stock] <= 0:
                raise InputError("official spots must be positive")
        level = (sum(a * official_spots[i] for i, a in self.coefficients)
                 if self.is_basket else official_spots[self.underlying])
        assert self.strike is not None
        return max(level - self.strike, 0.0) if self.kind in {"call", "basket_call"} else max(self.strike - level, 0.0)


def validate_instruments(instruments: Mapping[str, Instrument]) -> None:
    if not instruments or any(key != value.id for key, value in instruments.items()):
        raise InputError("instrument map must have matching, unique IDs")
    if len({x.currency for x in instruments.values()}) != 1:
        raise InputError("only one currency is supported")
    if sum(x.is_basket for x in instruments.values()) > 1:
        raise InputError("v1 supports one basket liability")
    for instrument in instruments.values():
        for stock in instrument.risk_stocks:
            if stock not in instruments or instruments[stock].kind != "stock":
                raise InputError(f"unknown constituent stock {stock}")


@dataclass(frozen=True)
class ShortLot:
    id: str
    stock_id: str
    quantity: float  # positive borrowed shares, FIFO order is list order
    sale_price: float
    opened_at: datetime

    def __post_init__(self) -> None:
        timestamp(self.opened_at)
        number(self.quantity, "short lot quantity")
        number(self.sale_price, "short lot price")
        if not self.id or self.quantity <= 0 or self.sale_price <= 0:
            raise InputError("short lots require ID, positive shares and price")


@dataclass(frozen=True)
class Opening:
    mode: Literal["ORIGINATE", "IMPORT"]
    initial_capital: float | None = None
    cash: float = 0.0
    positions: Mapping[str, float] = field(default_factory=dict)
    marks: Mapping[str, float] = field(default_factory=dict)
    posted_margin_cash: float = 0.0
    restricted_short_proceeds: float = 0.0
    short_lots: tuple[ShortLot, ...] = ()
    opening_equity: float | None = None
    liability_id: str | None = None
    liability_quantity: float = -1.0
    original_premium: float | None = None  # IMPORT metadata, never a posting

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", readonly(self.positions))
        object.__setattr__(self, "marks", readonly(self.marks))
        object.__setattr__(self, "short_lots", tuple(self.short_lots))


@dataclass(frozen=True)
class Funding:
    cash_rate: float = 0.0
    collateral_rate: float = 0.0
    borrow_rates: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        number(self.cash_rate, "cash rate")
        number(self.collateral_rate, "collateral rate")
        for rate in self.borrow_rates.values():
            number(rate, "borrow rate", minimum=0)
        object.__setattr__(self, "borrow_rates", readonly(self.borrow_rates))
