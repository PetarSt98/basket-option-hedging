# Hedge backtest pseudocode

One currency; one short European arithmetic-basket call or put; constituent shares, calls and puts as permitted hedges. All options are cash-settled at official expiry fixings. These are demonstration assumptions, not confirmed Polykern contract terms. Pricing and policy are separate in-process components; the basket cannot be traded as a hedge.

## Accounting

```text
Position_value = signed_quantity * multiplier * unit_mark
Equity = total_cash + sum(Position_value)
Available_cash = total_cash - posted_margin - restricted_short_proceeds
Fill: quantity += dq; cash -= dq * multiplier * fill_price + fee
Settlement: mark to payoff; cash += quantity * multiplier * payoff;
            remove the option once, using its settlement ID
P&L = Equity - opening_baseline
```

ORIGINATE starts with risk capital and no positions; its baseline is capital. Book the released premium and short liability together once. IMPORT loads cash, positions, restrictions and consistent FIFO short-proceeds lots; its baseline is opening marked equity, with no repeated premium. A valid but infeasible opening records a breach. No later external capital is added.

## Engine

```python
def backtest_engine(instruments, opening, market_data, strategy, config):
    validate_contracts_units_and_rules(instruments, opening, config)
    book, baseline = open_account(opening)
    orders, market, journal = empty_orders(), empty_market(), new_journal()

    try:
        for event in replay_clock(market_data, orders, instruments, config):
            previous = snapshot(book, market, funding_rates)
            market.release(event)                  # Preserve original timestamps.
            history = market.released_history()    # Read-only; no future iterator.

            # Validate the entire phase first, including official fixings.
            # ORIGINATE only: include inception sale/mark from released inputs.
            cashflows = prepare_due_cashflows(book, previous, market, event, config)
            orders.expire_deadlines_and_option_fixing_cutoffs(event)
            book.post_once(cashflows)               # Atomic; debit unfunded payments too.
            apply_released_rates_for_next_interval(event)
            if basket_settled_in(cashflows):
                orders.cancel_all()

            values = value_portfolio(book, market, history, event, config.pricer)
            check_and_reserve_margin(book, values)  # Record breach BEFORE any fill.

            execute_pending_orders(orders, book, market, values, event, config)
            check_and_reserve_margin(book, values)
            view = read_only(book, orders, market, history, values, event, config)

            if book.basket_settled:
                targets = zero_holdings(config.hedge_universe)
            else:
                targets = evaluate_strategy_or_record_missing_inputs(strategy, view)

            if targets is not None:
                validate_targets(targets, config)
                for instrument, target in targets:
                    required = target - book.quantity(instrument)
                    orders.keep_compatible_and_cancel_obsolete(instrument, required)
                    additional = required - orders.pending_quantity(instrument)
                    if additional != 0:
                        orders.submit(instrument, additional, after=event, rules=config)

            journal.record_and_reconcile(event, book, orders, values, baseline)

        status = COMPLETED if book.basket_settled and book.is_flat else INCOMPLETE_EXIT
    except (MissingData, RiskBreach) as error:
        breach = is_risk_breach(error) or known_cash_or_borrow_breach(book)
        status = STOP_WITH_BREACH if breach else STOP_DATA_GAP
        journal.record_stop(error, status, book)    # Last valid values retain their times.

    orders.cancel_all()
    return report(book, journal, baseline, status)


def execute_pending_orders(orders, book, market, values, event, config):
    for order in orders.eligible_in_instrument_order(event):
        fill = propose_whole_fill(order, market, event, config)
        if fill is missing_quote:
            continue                              # Pending until its deadline.
        if fill is rejected:
            orders.record_rejection(order)
            continue

        trial = copy(book)
        trial.apply_fill(fill)                    # Same ledger method as actual execution.
        if not account_feasible(trial, values, config):
            orders.reject(order, projected_shortfalls(trial, values))
            continue                              # No actual fee, cash movement or breach.

        book.apply_fill(fill)
        orders.record_fill(fill)
        market.consume_quote_size(fill)
        check_and_reserve_margin(book, values)     # Earlier fills survive later rejections.
```

## Policy, clock and constraints

```python
def stock_delta_hedge(view):
    targets = zero_holdings(view.hedge_universe)
    for stock in view.permitted_stocks:
        delta = sum(q * multiplier * unit_delta_to(stock)
                    for each currently held NONSPOT position)
        targets[stock] = round_to_configured_lot(-delta)
    return targets
```

**Policy:** all held options contribute until they settle or actually close; option targets are zero for this stock baseline. With no constituent options held, this is negative basket delta. A short put with quantity −1, multiplier 100 and unit delta −0.40 targets −40 shares; from −10 held and no pending order, sell 30. Omitted targets, `None` and missing policy inputs preserve valid orders. The engine evaluates every boundary; the scheduled policy returns `None` outside flagged reviews. After actual basket settlement, ordinary closing orders unwind the permitted hedges.

**Clock and pricing:** merge released records with start/horizon, registered expiries, funding/rule records, and live order activation/deadline times. Process all same-time source sequences before finishing the horizon. Privately validating the following source row can raise `InputError` for a malformed future payload, even beyond the horizon; its contents never reach the strategy. Timer boundaries invent no quotes or marks. Supplied valuation packets must match the current time; otherwise an in-process pricer must use released inputs within freshness/calibration cutoffs. Missing essential marks or fixings stop the run.

**Execution and cash:** fresh contemporaneous quotes must be observed strictly after submission, even with zero delay. Buy at ask, sell at bid; include slippage and fees once, enforce whole-order size/lot/borrow/position limits, and cancel immediately. Gross stock-short proceeds are segregated and released FIFO on covers. Accrue interest and borrow using prior balances, marks and rates. Interest credited at each boundary compounds in later intervals, so extra order timers can change financing without fills. Collateral changes restrictions, not equity.

**Risk and results:** use the uncalibrated Unnetted Liability plus Stress Cash Proxy: gross short-option marked liability plus worst portfolio stress loss over the declared complete set. Require both equity and cash excluding short proceeds to cover margin plus buffer, plus borrow/position limits. Record actual breaches before canceling orders; retain positions without liquidation or funding injection. Reconcile `P&L = liability P&L + hedge P&L - trading costs - net financing costs`. Report costs, funding, settlements, orders, residual inventory and distinct completed/breach/data-gap/incomplete outcomes. Compare matched valid checkpoints; a synthetic path validates mechanics, not real-world profitability.
