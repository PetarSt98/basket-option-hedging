"""The sole accounting implementation, used unchanged for preview and execution."""

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Mapping

from .contracts import (DataGap, Funding, InputError, Instrument, Opening,
                        QUANTITY_TOL, ReconciliationError, ShortLot, close,
                        number, timestamp, validate_instruments)


@dataclass(frozen=True)
class Posting:
    id: str
    time: datetime
    kind: str
    cash_change: float = 0.0
    instrument_id: str | None = None
    quantity: float = 0.0
    price: float = 0.0
    reference_mark: float = 0.0
    restriction_change: float = 0.0


@dataclass(frozen=True)
class Settlement:
    id: str
    time: datetime
    instrument_id: str
    quantity: float
    payoff: float
    cashflow: float


class Ledger:
    def __init__(self, instruments: Mapping[str, Instrument], opening: Opening):
        validate_instruments(instruments)
        if opening.liability_id is not None:
            if opening.liability_id not in instruments or not instruments[opening.liability_id].is_basket:
                raise InputError("opening liability_id must identify the basket")
            if opening.mode == "IMPORT" and not opening.positions.get(opening.liability_id, 0):
                raise InputError("import liability_id must identify a held basket; omit it for a book without one")
        self.instruments = dict(instruments)
        self.mode = opening.mode
        self.cash = 0.0
        self.positions: dict[str, float] = {}
        self.marks: dict[str, float] = {}
        self.posted_margin_cash = 0.0
        self.short_lots: list[ShortLot] = []
        self._postings: list[Posting] = []
        self._operations: dict[str, tuple] = {}
        self.settlements: dict[str, Settlement] = {}
        self.original_premium = opening.original_premium
        if self.original_premium is not None:
            number(self.original_premium, "original premium", minimum=0)
        if opening.mode == "ORIGINATE":
            if opening.initial_capital is None:
                raise InputError("ORIGINATE requires positive initial capital")
            number(opening.initial_capital, "initial capital")
            if opening.initial_capital <= 0:
                raise InputError("ORIGINATE requires positive initial capital")
            if (opening.positions or opening.marks or opening.short_lots or opening.cash != 0
                    or opening.posted_margin_cash != 0 or opening.restricted_short_proceeds != 0
                    or opening.opening_equity is not None or opening.original_premium is not None):
                raise InputError("ORIGINATE must start with capital and no opening book")
            if opening.liability_id not in instruments or not instruments[opening.liability_id].is_basket:
                raise InputError("ORIGINATE requires the basket liability")
            number(opening.liability_quantity, "liability quantity")
            if opening.liability_quantity >= 0:
                raise InputError("liability origination must be short")
            self.cash = opening.initial_capital
            self.baseline = opening.initial_capital
        elif opening.mode == "IMPORT":
            if opening.initial_capital is not None:
                raise InputError("IMPORT has opening equity, not a new capital contribution")
            number(opening.cash, "opening cash")
            number(opening.posted_margin_cash, "opening margin cash", minimum=0)
            number(opening.restricted_short_proceeds, "opening short proceeds", minimum=0)
            self.cash = opening.cash
            self.posted_margin_cash = opening.posted_margin_cash
            for key, quantity in opening.positions.items():
                self._instrument(key)
                number(quantity, "opening quantity")
                if instruments[key].is_basket and quantity > 0:
                    raise InputError("the basket position must be a short liability")
                if quantity != 0:
                    self.positions[key] = quantity
            self.set_marks(opening.marks)
            self.short_lots = list(opening.short_lots)
            self._validate_lots()
            if not close(self.restricted_short_proceeds, opening.restricted_short_proceeds):
                raise InputError("opening restricted proceeds do not equal lot proceeds")
            if opening.opening_equity is None:
                raise InputError("IMPORT requires declared opening marked equity")
            number(opening.opening_equity, "opening equity")
            if not close(self.equity, opening.opening_equity):
                raise InputError("opening equity is inconsistent with cash/positions/marks")
            self.baseline = opening.opening_equity
        else:
            raise InputError("opening mode must be ORIGINATE or IMPORT")
        self.opening_values = self.position_values()

    @property
    def postings(self) -> tuple[Posting, ...]:
        return tuple(self._postings)

    @property
    def restricted_short_proceeds(self) -> float:
        return sum(lot.quantity * lot.sale_price for lot in self.short_lots)

    @property
    def available_cash(self) -> float:
        return self.cash - self.posted_margin_cash - self.restricted_short_proceeds

    @property
    def equity(self) -> float:
        return self.cash + sum(self.position_values().values())

    def position_values(self) -> dict[str, float]:
        if any(key not in self.marks for key in self.positions):
            raise DataGap("missing mark for an open position")
        return {key: q * self.instruments[key].multiplier * self.marks[key]
                for key, q in self.positions.items()}

    def clone(self) -> "Ledger":
        other = object.__new__(Ledger)
        other.__dict__ = self.__dict__.copy()
        for key in ("positions", "marks", "_operations", "settlements", "opening_values"):
            setattr(other, key, getattr(self, key).copy())
        other.short_lots = self.short_lots.copy()
        other._postings = self._postings.copy()
        return other

    def _instrument(self, key: str) -> Instrument:
        if key not in self.instruments:
            raise InputError(f"unknown instrument {key}")
        return self.instruments[key]

    def _validate_lots(self) -> None:
        if len({lot.id for lot in self.short_lots}) != len(self.short_lots):
            raise InputError("duplicate short lot ID")
        prior: dict[str, datetime] = {}
        for lot in self.short_lots:
            if self._instrument(lot.stock_id).kind != "stock":
                raise InputError("proceeds lots must reference stocks")
            if lot.stock_id in prior and lot.opened_at < prior[lot.stock_id]:
                raise InputError("opening lots must be FIFO ordered")
            prior[lot.stock_id] = lot.opened_at
        for key, instrument in self.instruments.items():
            if instrument.kind == "stock":
                shares = sum(lot.quantity for lot in self.short_lots if lot.stock_id == key)
                if abs(shares - max(-self.positions.get(key, 0.0), 0.0)) > QUANTITY_TOL:
                    raise InputError(f"short lots do not match outstanding shares of {key}")

    def _new_operation(self, key: str, payload: tuple) -> bool:
        if not key:
            raise InputError("posting operation needs a nonempty ID")
        if key in self._operations:
            if self._operations[key] != payload:
                raise InputError(f"conflicting duplicate posting ID {key}")
            return False
        if any(key.startswith(old + ":") or old.startswith(key + ":") for old in self._operations):
            raise InputError(f"posting operation namespace collision {key}")
        self._operations[key] = payload
        return True

    def _post(self, posting: Posting) -> None:
        if any(p.id == posting.id for p in self._postings):
            raise InputError(f"posting ID collision {posting.id}")
        self._postings.append(posting)
        self.cash += posting.cash_change

    def set_marks(self, marks: Mapping[str, float]) -> None:
        checked = dict(marks)
        for key, value in checked.items():
            instrument = self._instrument(key)
            number(value, "mark", minimum=0)
            if instrument.kind == "stock" and value <= 0:
                raise InputError("stock marks must be positive")
        if any(key not in checked for key in self.positions):
            raise DataGap("missing current mark for held instrument")
        self.marks = checked

    def originate(self, key: str, time: datetime, instrument_id: str,
                  quantity: float, price: float, mark: float, fee: float = 0.0) -> None:
        timestamp(time)
        instrument = self._instrument(instrument_id)
        for value, name in ((quantity, "quantity"), (price, "premium"), (mark, "mark"), (fee, "fee")):
            number(value, name)
        if self.mode != "ORIGINATE" or not instrument.is_basket or quantity >= 0 or min(price, mark, fee) < 0:
            raise InputError("invalid liability origination")
        payload = ("origination", time, instrument_id, quantity, price, mark, fee)
        if key in self._operations:
            self._new_operation(key, payload)
            return
        if any(p.kind == "origination" for p in self._postings):
            raise InputError("liability already originated")
        self._new_operation(key, payload)
        self.positions[instrument_id] = quantity
        self.marks[instrument_id] = mark
        self.original_premium = -quantity * instrument.multiplier * price
        self._post(Posting(key, time, "origination", self.original_premium, instrument_id, quantity, price, mark))
        self._post(Posting(key + ":fee", time, "fee", -fee, instrument_id))

    def fill(self, key: str, time: datetime, instrument_id: str,
             quantity: float, price: float, fee: float = 0.0) -> None:
        timestamp(time)
        instrument = self._instrument(instrument_id)
        for value, name in ((quantity, "fill quantity"), (price, "fill price"), (fee, "fee")):
            number(value, name)
        if instrument.is_basket:
            raise InputError("the basket liability is never a hedge instrument")
        if quantity == 0 or price < 0 or fee < 0 or (instrument.kind == "stock" and price <= 0):
            raise InputError("invalid fill quantity/price/fee")
        if instrument.expiry is not None and time >= instrument.expiry:
            raise InputError("cannot trade an option at or after fixing")
        if instrument_id not in self.marks:
            raise DataGap("fill requires a reference accounting mark")
        reference = self.marks[instrument_id]
        payload = ("fill", time, instrument_id, quantity, price, fee, reference)
        if not self._new_operation(key, payload):
            return
        old = self.positions.get(instrument_id, 0.0)
        new = old + quantity
        old_restriction = self.restricted_short_proceeds
        if instrument.kind == "stock":
            cover = min(max(-old, 0.0), max(quantity, 0.0))
            remaining: list[ShortLot] = []
            for lot in self.short_lots:
                release = min(lot.quantity, cover) if lot.stock_id == instrument_id else 0.0
                cover -= release
                if lot.quantity - release > QUANTITY_TOL:
                    remaining.append(replace(lot, quantity=lot.quantity - release))
            self.short_lots = remaining
            opened = max(-new, 0.0) - max(-old, 0.0) if quantity < 0 else 0.0
            if opened > QUANTITY_TOL:
                self.short_lots.append(ShortLot(key + ":lot", instrument_id, opened, price, time))
        if abs(new) <= QUANTITY_TOL:
            self.positions.pop(instrument_id, None)
        else:
            self.positions[instrument_id] = new
        self._post(Posting(key, time, "fill", -quantity * instrument.multiplier * price,
                           instrument_id, quantity, price, reference))
        self._post(Posting(key + ":fee", time, "fee", -fee, instrument_id))
        restriction = self.restricted_short_proceeds - old_restriction
        if restriction:
            self._post(Posting(key + ":proceeds", time, "short_proceeds", instrument_id=instrument_id,
                               restriction_change=restriction))

    def dividend(self, key: str, time: datetime, stock_id: str, per_share: float) -> None:
        """Released cash distribution on pretrade holdings; shorts owe the amount.

        The replay declares payment timing. No historical distribution is inferred
        from adjusted prices, and the same operation cannot be paid twice.
        """
        timestamp(time)
        if self._instrument(stock_id).kind != "stock":
            raise InputError("cash distributions require a stock instrument")
        number(per_share, "cash dividend per share", minimum=0)
        if not self._new_operation(key, ("dividend", time, stock_id, per_share)):
            return
        quantity = self.positions.get(stock_id, 0.0)
        self._post(Posting(key, time, "dividend", quantity * per_share,
                           stock_id, quantity, per_share))

    def allocate_margin(self, key: str, time: datetime, required: float) -> None:
        timestamp(time)
        number(required, "posted margin", minimum=0)
        if not self._new_operation(key, ("margin", time, required)):
            return
        change = required - self.posted_margin_cash
        self.posted_margin_cash = required
        self._post(Posting(key, time, "collateral", restriction_change=change))

    def accrue(self, key: str, start: datetime, end: datetime, funding: Funding) -> None:
        timestamp(start)
        timestamp(end)
        if end < start:
            raise InputError("accrual cannot run backward")
        years = (end - start).total_seconds() / (365 * 24 * 60 * 60)
        # Engine calls only from a previously feasible state, except zero interval at opening.
        interest = (funding.cash_rate * self.available_cash + funding.collateral_rate * self.posted_margin_cash) * years
        charges = []
        for stock, quantity in self.positions.items():
            if self.instruments[stock].kind == "stock" and quantity < 0:
                if stock not in funding.borrow_rates or stock not in self.marks:
                    raise DataGap(f"missing previous-interval borrow rate/mark for {stock}")
                charges.append((stock, funding.borrow_rates[stock] * -quantity * self.marks[stock] * years))
        if not self._new_operation(key, ("accrual", start, end, funding)):
            return
        self._post(Posting(key + ":interest", end, "interest", interest))
        for stock, amount in charges:
            self._post(Posting(key + ":borrow:" + stock, end, "borrow", -amount, stock))

    def settle(self, key: str, time: datetime, instrument_id: str, payoff: float,
               fee: float = 0.0) -> Settlement:
        timestamp(time)
        instrument = self._instrument(instrument_id)
        number(payoff, "payoff", minimum=0)
        number(fee, "settlement fee", minimum=0)
        if not instrument.is_option or time < instrument.expiry:
            raise InputError("invalid option settlement time")
        payload = ("settlement", time, instrument_id, payoff, fee)
        if key in self._operations:
            self._new_operation(key, payload)
            return self.settlements[instrument_id]
        if instrument_id in self.settlements:
            raise InputError("instrument already settled under a different ID")
        self._new_operation(key, payload)
        quantity = self.positions.get(instrument_id, 0.0)
        self.marks[instrument_id] = payoff
        cashflow = quantity * instrument.multiplier * payoff
        record = Settlement(key, time, instrument_id, quantity, payoff, cashflow)
        self._post(Posting(key, time, "settlement", cashflow, instrument_id, quantity, payoff, payoff))
        self._post(Posting(key + ":fee", time, "fee", -fee, instrument_id))
        self.positions.pop(instrument_id, None)
        self.settlements[instrument_id] = record
        return record

    def components(self) -> dict[str, float]:
        """Derive P&L legs from instrument values and actual postings, never a residual."""
        current = self.position_values()
        liability = hedge = execution = fees = interest = borrow = 0.0
        stock_turnover = option_turnover = 0.0
        for key, instrument in self.instruments.items():
            change = current.get(key, 0.0) - self.opening_values.get(key, 0.0)
            if instrument.is_basket:
                liability += change
            else:
                hedge += change
        for posting in self._postings:
            if posting.kind == "origination":
                liability += posting.cash_change
            elif posting.kind == "settlement":
                if self.instruments[posting.instrument_id].is_basket:
                    liability += posting.cash_change
                else:
                    hedge += posting.cash_change
            elif posting.kind == "fill":
                instrument = self.instruments[posting.instrument_id]
                reference_value = posting.quantity * instrument.multiplier * posting.reference_mark
                hedge -= reference_value
                execution += -posting.cash_change - reference_value
                if instrument.kind == "stock":
                    stock_turnover += abs(posting.cash_change)
                else:
                    option_turnover += abs(posting.cash_change)
            elif posting.kind == "fee":
                fees -= posting.cash_change
            elif posting.kind == "dividend":
                hedge += posting.cash_change
            elif posting.kind == "interest":
                interest += posting.cash_change
            elif posting.kind == "borrow":
                borrow -= posting.cash_change
        net = self.equity - self.baseline
        explained = liability + hedge - execution - fees - borrow + interest
        if not close(net, explained):
            raise ReconciliationError(f"equity P&L {net} != independent components {explained}")
        return dict(net_pnl=net, liability_pnl=liability, hedge_pnl=hedge,
                    execution_cost=execution, fees=fees, trading_costs=execution + fees,
                    interest=interest, borrow_charges=borrow, financing_costs=borrow - interest,
                    stock_turnover=stock_turnover, option_premium_turnover=option_turnover,
                    reconciliation_error=net - explained)
