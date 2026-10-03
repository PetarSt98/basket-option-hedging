"""Incremental replay, causal execution and recorded experimental failure states."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Mapping, Protocol

from .contracts import (ABS_TOL, DataGap, Funding, InputError, Instrument, Opening,
                        QUANTITY_TOL, close, number)
from .ledger import Ledger, Posting, Settlement
from .market import (Event, Fixing, Observation, Pricer, PricingView, Quote, Rules,
                     SuppliedValuations, ValuationPacket, readonly)
from .orders import Order, OrderQueue
from .replay import replay_clock
from .risk import MarginModel, Risk, assess


@dataclass(frozen=True)
class OrderOutcome:
    event_id: str
    time: datetime
    order_id: str
    instrument_id: str
    outcome: str
    reason: str
    quantity: float
    price: float | None = None
    fee: float | None = None
    quote_id: str | None = None
    projected_risk: Risk | None = None


@dataclass(frozen=True)
class PolicyView:
    event_id: str
    time: datetime
    sequence: int
    instruments: Mapping[str, Instrument]
    positions: Mapping[str, float]
    cash: float
    posted_margin_cash: float
    restricted_short_proceeds: float
    available_cash: float
    valuation: ValuationPacket
    history: tuple[Observation, ...]
    order_history: tuple[OrderOutcome, ...]
    hedge_universe: tuple[str, ...]
    terminal_decision: bool
    rules: Mapping[str, Any]
    pending_orders: tuple[Order, ...] = ()
    market: Mapping[str, Observation] = field(default_factory=dict)
    quotes: Mapping[str, Quote] = field(default_factory=dict)
    decision_event: bool = True
    basket_settled: bool = False


class Policy(Protocol):
    name: str

    def targets(self, view: PolicyView) -> Mapping[str, float] | None: ...


@dataclass(frozen=True)
class Snapshot:
    event_id: str
    time: datetime
    sequence: int
    phase: str
    positions: Mapping[str, float]
    risk: Risk
    pnl: Mapping[str, float] | None
    residual_delta: Mapping[str, float] | None


@dataclass(frozen=True)
class Breach:
    event_id: str
    time: datetime
    phase: str
    risk: Risk
    data_gap: bool = False


@dataclass
class Result:
    policy: str
    status: str
    reason: str
    opening_mode: str
    baseline: float
    horizon: datetime
    ledger: Ledger
    snapshots: list[Snapshot] = field(default_factory=list)
    order_outcomes: list[OrderOutcome] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    breaches: list[Breach] = field(default_factory=list)
    settlement_outcomes: list[dict[str, Any]] = field(default_factory=list)
    data_gaps: list[dict[str, Any]] = field(default_factory=list)
    valuations: list[dict[str, Any]] = field(default_factory=list)
    margin_label: str = ""

    @property
    def fills(self) -> tuple[Posting, ...]:
        return tuple(post for post in self.ledger.postings if post.kind == "fill")

    @property
    def last_valid_snapshot(self) -> Snapshot | None:
        return next((s for s in reversed(self.snapshots) if s.pnl is not None), None)


class Engine:
    """Single-use engine; policies and valuation providers are separate components."""

    def __init__(self, instruments: Mapping[str, Instrument], opening: Opening,
                 rules: Rules, margin_model: MarginModel, pricer: Pricer | None = None):
        self.ledger = Ledger(instruments, opening)
        self.instruments = self.ledger.instruments
        self.opening = opening
        self.rules = rules
        self.margin_model = margin_model
        self.pricer = pricer if pricer is not None else SuppliedValuations()
        self._expected_stresses = rules.expected_stress_scenarios
        self.liability_id = opening.liability_id or next(
            (key for key in self.ledger.positions if instruments[key].is_basket), None)
        for key in rules.hedge_universe:
            if key not in instruments or instruments[key].is_basket:
                raise InputError("hedge universe must contain known instruments, never the basket")
        for key in (*rules.borrow, *rules.funding.borrow_rates, *rules.lot_sizes,
                    *rules.maximum_order_quantities, *rules.maximum_position_quantities):
            if key not in instruments:
                raise InputError(f"unknown rule instrument {key}")
        self.funding = rules.funding
        self.borrow = dict(rules.borrow)
        self.orders = OrderQueue()
        self.pending = self.orders.pending
        self.history: list[Observation] = []
        self._seen_data: dict[tuple[str, str], Any] = {}
        self._quotes: dict[str, Quote] = {}
        self.market: dict[str, Observation] = {}
        self._quote_used: dict[tuple[str, str], float] = {}
        self._fixings: dict[str, Fixing] = {}
        self._started = False
        self._interval_start: datetime | None = None
        self._mandatory_committed = False

    def _unique(self, category: str, key: str, value: Any) -> bool:
        identity = (category, key)
        if identity in self._seen_data:
            if self._seen_data[identity] != value:
                raise InputError(f"conflicting duplicate {category} ID {key}")
            return False
        self._seen_data[identity] = value
        return True

    def _release(self, event: Event) -> None:
        if event.valuation is not None and event.valuation.as_of > event.time:
            raise InputError("future valuation packet")
        for observation in event.observations:
            if observation.instrument_id not in self.instruments:
                raise InputError("unknown observation instrument")
            if self.instruments[observation.instrument_id].kind == "stock" and observation.value <= 0:
                raise InputError("stock observations must be positive")
            if self._unique("observation", observation.id, observation):
                self.history.append(observation)
                prior = self.market.get(observation.instrument_id)
                if prior is None or observation.observed_at >= prior.observed_at:
                    self.market[observation.instrument_id] = observation
        for quote in event.quotes:
            if quote.instrument_id not in self.instruments:
                raise InputError("unknown quote instrument")
            if self.instruments[quote.instrument_id].kind == "stock" and quote.bid <= 0:
                raise InputError("stock quotes must be positive")
            self._unique("quote", quote.id, quote)
            prior_quote = self._quotes.get(quote.instrument_id)
            if prior_quote is None or quote.time >= prior_quote.time:
                self._quotes[quote.instrument_id] = quote
        for fixing in event.fixings:
            if fixing.instrument_id not in self.instruments:
                raise InputError("unknown fixing instrument")
            instrument = self.instruments[fixing.instrument_id]
            if fixing.observed_at != instrument.expiry:
                raise InputError("official fixing time must equal contractual expiry")
            # All contracts share the same official stock value at a fixing time.
            for prior in self._fixings.values():
                if prior.observed_at == fixing.observed_at:
                    for stock in sorted(prior.spots.keys() & fixing.spots.keys()):
                        if prior.spots[stock] != fixing.spots[stock]:
                            raise InputError(f"conflicting official fixing for {stock} at {fixing.observed_at}")
            self._unique("fixing", fixing.id, fixing)
            old = self._fixings.get(fixing.instrument_id)
            if old is not None and old != fixing:
                raise InputError("conflicting official fixing for instrument")
            self._fixings[fixing.instrument_id] = fixing
        if event.borrow is not None:
            for stock in event.borrow:
                if stock not in self.instruments or self.instruments[stock].kind != "stock":
                    raise InputError("borrow changes must reference stocks")
            self.borrow.update(event.borrow)
        if event.funding is not None:
            for stock in event.funding.borrow_rates:
                if stock not in self.instruments or self.instruments[stock].kind != "stock":
                    raise InputError("borrow rates must reference stocks")

    def _valid_packet(self, event: Event, ledger: Ledger) -> ValuationPacket:
        packet = event.valuation
        if packet is None:
            raise DataGap("missing valuation packet")
        if packet.as_of > event.time:
            raise InputError("future valuation packet")
        if packet.as_of != event.time:
            raise DataGap("valuation packet is not as-of this event")
        if self._expected_stresses is None:
            # Compatibility for old fixtures: bind once, never shrink on later packets.
            self._expected_stresses = tuple(packet.stress_marks)
        missing_stresses = set(self._expected_stresses) - packet.stress_marks.keys()
        if missing_stresses:
            raise DataGap(f"missing declared stress scenarios: {sorted(missing_stresses)}")
        for key, mark in packet.marks.items():
            if key not in self.instruments:
                raise InputError("unknown valuation instrument")
            if self.instruments[key].kind == "stock" and mark <= 0:
                raise InputError("stock marks must be positive")
        for key in ledger.positions:
            observed = packet.observed_at.get(key)
            if key not in packet.marks or observed is None:
                raise DataGap(f"missing mark or observation timestamp for {key}")
            if (event.time - observed).total_seconds() > self.rules.max_mark_age_seconds:
                raise DataGap(f"stale valuation input for {key}")
        return packet

    def _value_portfolio(self, event: Event) -> Event:
        view = PricingView(event.time, readonly(self.instruments), readonly(self.market),
                           readonly(self._quotes), readonly(self._fixings), tuple(self.history),
                           event.valuation)
        packet = self.pricer.value(view)
        if not isinstance(packet, ValuationPacket):
            raise InputError("pricer must return a ValuationPacket")
        return replace(event, valuation=packet)

    def _mandatory(self, event: Event, previous: datetime | None) -> list[Settlement]:
        # Validate the ENTIRE mandatory phase before touching the actual ledger.
        if previous is None and any(lot.opened_at > event.time for lot in self.ledger.short_lots):
            raise InputError("imported short lot cannot originate after inception")
        payoffs: dict[str, float] = {}
        for key in self.ledger.positions:
            instrument = self.instruments[key]
            if instrument.expiry is not None and instrument.expiry <= event.time:
                fixing = self._fixings.get(key)
                if fixing is None or event.time != instrument.expiry:
                    raise DataGap(f"missing timely fixing/settlement event for {key}")
                payoffs[key] = instrument.payoff(fixing.spots)
        if previous is not None:
            for key, quantity in self.ledger.positions.items():
                if self.instruments[key].kind == "stock" and quantity < 0:
                    if key not in self.funding.borrow_rates or key not in self.ledger.marks:
                        raise DataGap(f"missing prior funding input for {key}")
        if previous is None and self.opening.mode == "ORIGINATE":
            packet = event.valuation
            key = self.opening.liability_id
            if (packet is None or packet.as_of != event.time or key not in packet.marks
                    or key not in packet.observed_at or event.origination_price is None):
                raise DataGap("missing released inception mark/premium")
            if (event.time - packet.observed_at[key]).total_seconds() > self.rules.max_mark_age_seconds:
                raise DataGap("stale origination inputs")
            if event.time >= self.instruments[key].expiry:
                raise InputError("origination must precede expiry")
        self._expire(event)
        prospective = self.ledger.clone()
        if previous is not None:
            prospective.accrue(f"accrual/{event.id}", previous, event.time, self.funding)
        if previous is None and self.opening.mode == "ORIGINATE":
            key = self.opening.liability_id
            prospective.originate("origination", event.time, key, self.opening.liability_quantity,
                                  event.origination_price, event.valuation.marks[key], self.rules.origination_fee)
        settlements = []
        for key, payoff in sorted(payoffs.items()):
            settlements.append(prospective.settle(f"settlement/{key}", event.time, key, payoff,
                                                   self.rules.settlement_fee))
        self.ledger = prospective
        self._mandatory_committed = True
        if event.funding is not None:
            self.funding = event.funding
        return settlements

    def _outcome(self, event: Event, order: Order, outcome: str, reason: str, **details: Any) -> None:
        self.result.order_outcomes.append(OrderOutcome(event.id, event.time, order.id,
                                                       order.instrument_id, outcome, reason,
                                                       order.quantity, **details))

    def _cancel(self, event: Event, reason: str) -> None:
        for order in self.pending.values():
            self._outcome(event, order, "cancelled", reason)
        self.pending.clear()

    def _expire(self, event: Event) -> None:
        for key, order in list(self.pending.items()):
            instrument = self.instruments[order.instrument_id]
            expired = (event.time > order.deadline if order.terminal else event.time >= order.deadline)
            fixing_cutoff = instrument.expiry is not None and event.time >= instrument.expiry
            if expired or fixing_cutoff:
                self._outcome(event, order, "expired", "fixing cutoff" if fixing_cutoff else "order deadline")
                del self.pending[key]

    def _risk(self, event: Event, ledger: Ledger) -> Risk:
        packet = self._valid_packet(event, ledger)
        required = self.margin_model.required_margin(ledger, packet)
        return assess(ledger, required, self.rules.safety_buffer, self.borrow,
                      position_limits=self.rules.maximum_position_quantities)

    def _residual_delta(self, packet: ValuationPacket | None) -> dict[str, float] | None:
        if packet is None:
            return None
        result = {key: self.ledger.positions.get(key, 0.0) for key, inst in self.instruments.items() if inst.kind == "stock"}
        for key, quantity in self.ledger.positions.items():
            instrument = self.instruments[key]
            if instrument.is_option:
                deltas = packet.deltas.get(key, {})
                if any(stock not in deltas for stock in instrument.risk_stocks):
                    return None
                for stock in instrument.risk_stocks:
                    result[stock] += quantity * instrument.multiplier * deltas[stock]
        return result

    def _snapshot(self, event: Event, phase: str, risk: Risk, *, valued: bool = True) -> None:
        self.result.snapshots.append(Snapshot(event.id, event.time, event.sequence, phase,
                                             readonly(self.ledger.positions), risk,
                                             readonly(self.ledger.components()) if valued else None,
                                             readonly(self._residual_delta(event.valuation)) if valued else None))

    def _check(self, event: Event, phase: str) -> bool:
        risk = self._risk(event, self.ledger)
        # Snapshot and breach are recorded BEFORE any response or allocation.
        self._snapshot(event, phase, risk)
        if not risk.feasible:
            self.result.breaches.append(Breach(event.id, event.time, phase, risk))
            self.result.status = "STOP_WITH_BREACH"
            self.result.reason = f"actual account infeasible at {phase}"
            self._cancel(event, "account breach")
            return False
        self.ledger.allocate_margin(f"margin/{event.id}/{phase}", event.time, risk.required_margin)
        self.ledger.components()
        return True

    def _data_gap(self, event: Event, reason: str) -> None:
        risk = assess(self.ledger, None, self.rules.safety_buffer, self.borrow, valued=False,
                      position_limits=self.rules.maximum_position_quantities)
        known_failure = not risk.feasible
        self._snapshot(event, "data_gap", risk, valued=False)
        self.result.data_gaps.append(dict(event_id=event.id, time=event.time, reason=reason,
                                         outstanding_positions=dict(self.ledger.positions),
                                         mandatory_phase="committed" if self._mandatory_committed else "unposted",
                                         accrual_interval_start=self._interval_start,
                                         accrual_interval_end=event.time,
                                         due_option_ids=[key for key in self.ledger.positions
                                                         if self.instruments[key].expiry is not None
                                                         and self.instruments[key].expiry <= event.time],
                                         required_horizon=self.rules.horizon))
        if known_failure:
            self.result.breaches.append(Breach(event.id, event.time, "data_gap", risk, True))
        self.result.status = "STOP_WITH_BREACH" if known_failure else "STOP_DATA_GAP"
        self.result.reason = reason
        self._cancel(event, "data gap")

    def _quantity_error(self, key: str, quantity: float, *, check_maximum: bool = True) -> str | None:
        lot = self.rules.lot_sizes.get(key, 0.0)  # zero explicitly means fractional quantities
        if lot and abs(quantity - round(quantity / lot) * lot) > QUANTITY_TOL:
            return "invalid lot size"
        maximum = self.rules.maximum_order_quantities.get(key)
        if check_maximum and maximum is not None and abs(quantity) > maximum + QUANTITY_TOL:
            return "maximum order quantity exceeded"
        return None

    def _execute(self, event: Event) -> bool:
        for order in self.orders.eligible_at(event.time):
            candidates = [quote for quote in event.quotes if quote.instrument_id == order.instrument_id
                          and quote.time == quote.available_at == event.time and quote.time > order.submitted_at]
            if not candidates:
                self._outcome(event, order, "pending", "no fresh contemporaneous execution quote")
                continue
            if len({q.id for q in candidates}) > 1:
                raise InputError("execution event must designate one quote per instrument")
            quote = candidates[0]
            side = "ask" if order.quantity > 0 else "bid"
            price = quote.ask + self.rules.slippage_per_unit if order.quantity > 0 else quote.bid - self.rules.slippage_per_unit
            fee = self.rules.fee_per_order + abs(order.quantity) * self.rules.fee_per_unit
            capacity = (quote.ask_size if side == "ask" else quote.bid_size) - self._quote_used.get((quote.id, side), 0.0)
            error = self._quantity_error(order.instrument_id, order.quantity)
            if abs(order.quantity) > capacity + QUANTITY_TOL:
                error = "insufficient remaining quoted size"
            if price < 0 or (self.instruments[order.instrument_id].kind == "stock" and price <= 0):
                error = "invalid execution price after slippage"
            projected_risk = None
            prospective = self.ledger.clone()
            if error is None:
                try:
                    prospective.fill(f"fill/{order.id}", event.time, order.instrument_id, order.quantity, price, fee)
                    self._valid_packet(event, prospective)
                    if (self.instruments[order.instrument_id].kind == "stock"
                            and prospective.positions.get(order.instrument_id, 0.0) < 0
                            and order.instrument_id not in self.funding.borrow_rates):
                        error = "missing stock borrow rate"
                    projected_risk = self._risk(event, prospective)
                    if not projected_risk.feasible:
                        error = "projected account infeasible"
                except DataGap as gap:
                    error = f"missing candidate valuation: {gap}"
            del self.pending[order.id]
            if error is not None:
                self._outcome(event, order, "rejected", error, projected_risk=projected_risk)
                continue
            # The preview used the same ledger operation; real execution calls it once on actual state.
            self.ledger.fill(f"fill/{order.id}", event.time, order.instrument_id, order.quantity, price, fee)
            self._quote_used[(quote.id, side)] = self._quote_used.get((quote.id, side), 0.0) + abs(order.quantity)
            self._outcome(event, order, "filled", "whole order", price=price, fee=fee, quote_id=quote.id)
            if not self._check(event, "postfill/" + order.id):
                return False
        return True

    def _earliest_execution(self, event: Event) -> datetime:
        return max(event.earliest_execution or event.time,
                   event.time + timedelta(seconds=self.rules.execution_delay_seconds))

    def _strategy_view(self, event: Event) -> PolicyView:
        return PolicyView(event.id, event.time, event.sequence, readonly(self.instruments),
                          readonly(self.ledger.positions), self.ledger.cash, self.ledger.posted_margin_cash,
                          self.ledger.restricted_short_proceeds, self.ledger.available_cash,
                          event.valuation.policy_copy(), tuple(self.history), tuple(self.result.order_outcomes),
                          tuple(self.rules.hedge_universe), event.terminal_decision,
                          readonly(dict(safety_buffer=self.rules.safety_buffer,
                                        earliest_execution=self._earliest_execution(event),
                                        next_rebalance_time=event.next_rebalance_time,
                                        fee_per_order=self.rules.fee_per_order,
                                        fee_per_unit=self.rules.fee_per_unit,
                                        slippage_per_unit=self.rules.slippage_per_unit,
                                        lot_sizes=self.rules.lot_sizes,
                                        maximum_order_quantities=self.rules.maximum_order_quantities,
                                        maximum_position_quantities=self.rules.maximum_position_quantities,
                                        funding=dict(cash_rate=self.funding.cash_rate,
                                                     collateral_rate=self.funding.collateral_rate,
                                                     borrow_rates=self.funding.borrow_rates),
                                        borrow={key: dict(located=value.located, limit=value.limit)
                                                for key, value in self.borrow.items()})),
                          self.orders.snapshot(), readonly(self.market), readonly(self._quotes),
                          event.decision, self.liability_id in self.ledger.settlements)

    def _decide(self, event: Event, policy: Policy | Callable) -> None:
        view = self._strategy_view(event)
        timing = dict(event_id=event.id, time=event.time,
                      earliest_execution=view.rules["earliest_execution"],
                      next_rebalance_time=event.next_rebalance_time)
        try:
            if view.basket_settled:
                targets = {key: 0.0 for key in self.rules.hedge_universe}
            else:
                targets = policy.targets(view) if hasattr(policy, "targets") else policy(view)
        except DataGap as gap:
            self.result.decisions.append(dict(timing, status="skipped", reason=str(gap)))
            return
        if targets is None:
            self.result.decisions.append(dict(timing, status="no_decision"))
            return
        if not isinstance(targets, Mapping) or any(not isinstance(key, str) for key in targets):
            raise InputError("policy must return instrument-to-quantity targets or None")
        valid_targets = {}
        for key, target in targets.items():
            try:
                number(target, "policy target")
            except InputError as error:
                self.result.decisions.append(dict(timing, status="rejected",
                                                  instrument_id=key, reason=str(error), target=repr(target)))
            else:
                valid_targets[key] = target
        targets = valid_targets
        self.result.decisions.append(dict(timing, status="targets", targets=targets.copy()))
        self._adjust_pending_orders(event, targets, terminal=view.basket_settled)

    def _adjust_pending_orders(self, event: Event, targets: Mapping[str, float], *, terminal: bool) -> None:
        earliest = self._earliest_execution(event)
        deadline = event.order_deadline or self.rules.horizon
        if event.order_deadline is None and self.rules.order_lifetime_seconds is not None:
            deadline = min(deadline, event.time + timedelta(seconds=self.rules.order_lifetime_seconds))
        inclusive = terminal or event.terminal_decision or (
            event.order_deadline is None and deadline == self.rules.horizon)
        for key, target in sorted(targets.items()):
            quantity = target - self.ledger.positions.get(key, 0.0)
            order = Order(f"{event.id}/{key}", key, quantity, event.time, earliest, deadline, inclusive)
            if key not in self.rules.hedge_universe or key not in self.instruments or self.instruments[key].is_basket:
                self._outcome(event, order, "rejected", "instrument outside permitted hedge universe")
                continue
            instrument = self.instruments[key]
            error = self._quantity_error(key, target, check_maximum=False)
            limit = self.rules.maximum_position_quantities.get(key)
            if limit is not None and abs(target) > limit + QUANTITY_TOL:
                error = "target exceeds position limit"
            if error:
                self._outcome(event, order, "rejected", error)
                continue
            if instrument.expiry is not None and event.time >= instrument.expiry and abs(quantity) > QUANTITY_TOL:
                self._outcome(event, order, "rejected", "option at or after fixing cutoff")
                continue
            for cancelled in self.orders.cancel_obsolete(key, quantity):
                self._outcome(event, cancelled, "cancelled", "superseded target")
            quantity -= self.orders.signed_pending_quantity(key)
            if abs(quantity) <= QUANTITY_TOL:
                continue
            order = replace(order, quantity=quantity)
            error = self._quantity_error(key, quantity)
            if earliest > deadline or (earliest == deadline and not inclusive):
                error = "no eligible execution before order deadline"
            if error:
                self._outcome(event, order, "rejected", error)
            else:
                self.pending[order.id] = order
                self._outcome(event, order, "submitted", "target minus holdings minus compatible pending quantity")

    def run(self, events: Iterable[Event], policy: Policy | Callable[[PolicyView], Mapping[str, float] | None]) -> Result:
        if self._started:
            raise InputError("instantiate a fresh engine for each run")
        self._started = True
        name = getattr(policy, "name", getattr(policy, "__name__", type(policy).__name__))
        self.result = Result(name, "RUNNING", "", self.opening.mode, self.ledger.baseline,
                             self.rules.horizon, self.ledger, margin_label=self.margin_model.label)
        previous_key: tuple[datetime, int] | None = None
        previous_time: datetime | None = None
        last_event: Event | None = None
        for event in replay_clock(events, self.orders.snapshot, self.instruments, self.rules):
            # Later sequences at H may contain its execution phase. Consume all
            # released H records, but never release a record beyond the horizon.
            if event.time > self.rules.horizon:
                break
            if not self._unique("event", event.id, event):
                continue
            key = (event.time, event.sequence)
            if previous_key is not None and key <= previous_key:
                raise InputError("events must increase in availability time and sequence")
            previous_key = key
            last_event = event
            self._interval_start = previous_time
            self._mandatory_committed = False
            self._release(event)
            try:
                # Origination needs its released price/mark before booking the liability.
                if previous_time is None and self.opening.mode == "ORIGINATE":
                    event = self._value_portfolio(event)
                # Validate, expire orders, accrue and atomically commit in that order.
                settlements = self._mandatory(event, previous_time)
                for settlement in settlements:
                    self.result.settlement_outcomes.append(dict(record=settlement,
                        fixing=self._fixings[settlement.instrument_id],
                        funded=self.ledger.cash - self.ledger.restricted_short_proceeds >= -ABS_TOL,
                        funding_convention="atomic same-event net cash, released old margin available"))
                if any(s.instrument_id == self.liability_id for s in settlements):
                    self._cancel(event, "basket settled; replace opening orders with exits")
                if previous_time is not None or self.opening.mode != "ORIGINATE":
                    event = self._value_portfolio(event)
                packet = self._valid_packet(event, self.ledger)
                self.ledger.set_marks(packet.marks)
                for instrument_id, quantity in self.ledger.positions.items():
                    if (self.instruments[instrument_id].kind == "stock" and quantity < 0
                            and instrument_id not in self.funding.borrow_rates):
                        raise DataGap(f"missing current stock borrow rate for {instrument_id}")
                if previous_time is None and self.opening.mode == "IMPORT":
                    for instrument_id in self.opening.positions:
                        if self.opening.positions[instrument_id] != 0 and not close(
                                self.opening.marks[instrument_id], packet.marks.get(instrument_id, self.opening.marks[instrument_id])):
                            raise InputError("inception marks conflict with imported opening marks")
                self.result.valuations.append(dict(event_id=event.id, as_of=packet.as_of,
                                                   input_ids=packet.input_ids, metadata=dict(packet.metadata)))
                if not self._check(event, "pretrade"):
                    break
                if (event.execution or event.quotes) and not self._execute(event):
                    break
                if not self._check(event, "post_execution"):
                    break
                self._decide(event, policy)
                self._snapshot(event, "event", self._risk(event, self.ledger))
            except DataGap as gap:
                self._data_gap(event, str(gap))
                break
            previous_time = event.time
        if self.result.status == "RUNNING":
            if last_event is not None and last_event.time == self.rules.horizon:
                self._cancel(last_event, "horizon reached")
                settled = self.liability_id is None or self.liability_id in self.ledger.settlements
                completed = not self.ledger.positions and settled
                self.result.status = "COMPLETED" if completed else "INCOMPLETE_EXIT"
                self.result.reason = "settled, zero inventory" if completed else "unsettled liability or residual inventory at horizon"
            elif last_event is not None:
                self._data_gap(last_event, "replay ended before declared horizon")
            else:
                self.result.status = "STOP_DATA_GAP"
                self.result.reason = "empty replay"
                self.result.data_gaps.append(dict(reason="empty replay", outstanding_positions=dict(self.ledger.positions)))
        self.result.ledger = self.ledger
        return self.result
