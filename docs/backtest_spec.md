# Hedge backtest specification

## 1. Scope and source status

**Facts attributed to Daniel, as relayed by the user:** the collaboration preparation includes 1–2 pages of engine pseudocode and reading 5–10 papers. The stated preliminary problem is short basket-volatility exposure, no liquid basket hedge, and hedging through constituent stocks/options while retaining premium subject to margin, cash and trading constraints. These statements do not specify the actual contract or broker arrangements.

**Sources inspected:** the user's message is the available project brief. At the original design inspection, this repository contained only Git metadata: no project description, papers, code or `AGENTS.md`. No separate source bundle was visible. The design review consulted the relevant methodological sections of [Cont and Vuletić, *Data-driven hedging with generative models* (2025), §2 and Remark 2.3](https://link.springer.com/article/10.1007/s10479-025-06867-3): pricing functions, hedge holdings and self-financing cash are distinct; a regression intercept does not determine the cash account. This is background, not a reproduction of their algorithm or results. The missing project materials and broader 5–10-paper reading task remain outstanding.

**User-required design:** a strategy-independent ledger and execution timeline; unhedged and scheduled constituent-delta baselines; no forecasting, training, neural networks, reinforcement learning, learned scenario generator or hedge optimizer. Pricing models are permitted. The existing liability is never an eligible hedge instrument and cannot be closed at its model mark, including after a breach.

**Demonstration assumptions:** one currency; fixed arithmetic basket; one short European basket put or call, with a put in the original fixtures; stock-only baseline hedging; European constituent puts/calls available to other policies and deterministic accounting tests; all options cash-settled with simultaneous fixing and settlement; immediate cash settlement for stock trades; same-event mandatory cashflows netted atomically; no dividends, splits or other corporate actions; fractional shares unless lots are configured; explicit stock borrow; a named research margin proxy. These are not Polykern facts or conventions asserted for listed equity options. Unsupported conventions must fail input validation.

**Final-outline update:** the engine evaluates policies at every clock boundary, preserves compatible pending orders, and permits a policy to return `None`. The scheduled baseline remains available. The stock-delta rule includes all held options until their actual closing fills; this extends the outline's basket-only example to an existing option book without ignoring residual exposure. It is a research convention, not an inferred Polykern requirement. The engine remains a research prototype.

**Subsequent Stage 1 extension:** the user authorized an optional conditional-variance/weighted-L1 policy using synthetic scenarios, documented in [cont_vuletic_stage1.md](cont_vuletic_stage1.md). The earlier prohibition on hedge optimization describes the original baseline scope; the engine/accounting contract below remains unchanged.

**Subsequent Stage 2 extension:** the user authorized a learned conditional generator for synthetic constituent returns, documented in [cont_vuletic_stage2.md](cont_vuletic_stage2.md). It replaces only the scenario source; the Stage 1 optimizer and the engine/accounting contract remain unchanged. It is a conditional Gaussian neural model, not a VolGAN reproduction.

## 2. Positions and units

For stock prices `S_i(t)` in currency/share, define `B(t) = sum_i a_i S_i(t)`. Fixed `a_i >= 0` means shares of stock i per basket unit, not a rebalanced portfolio weight. `B` and strike `K_B` are currency/basket unit. Unit payoff `h_B` is `max(K_B-B(T),0)` for a put or `max(B(T)-K_B,0)` for a call. One long contract pays `m_B h_B`, where `m_B` is basket units/contract. Signed quantity `q_B < 0` denotes the short liability; its cashflow is `q_B m_B h_B`.

Stock quantities are signed shares, with multiplier `m_i = 1`. Constituent-option quantities are signed contracts; `m_j` is shares/contract. Their unit payoff is `max(K_j-S_i,0)` for a put or `max(S_i-K_j,0)` for a call. Positive quantities are assets/long positions; negative quantities are shorts. Every option quote and mark `v_j` is per underlying unit, and its contract price is `m_j v_j`; never multiply twice. Stock marks are per share.

Fixture proposal: three stocks initially `(100,80,120)`, coefficients `(1/3,1/3,1/3)`, so `B(0)=100`; `K_B=100`, `q_B=-1`, `m_B=100`. Define initial reference notional `N_0=|q_B|m_B B(0)=10,000` currency, and strike notional `N_K=|q_B|m_B K_B`. Neither is cash capital or an extra payoff multiplier. Constituent-option tests use a declared multiplier, including a non-unit example.

## 3. Accounting identities — before the algorithm

`C` is total cash, including disjoint restricted cash balances: posted margin `R` and segregated stock-short proceeds `Q`. `A` is available cash; `E` is equity. For current signed positions:

```text
MV_j = q_j m_j v_j                  V = sum_j MV_j
A = C - R - Q                      E = C + V

Trade dq_j at unit fill p_j, fee f >= 0:
  q_j' = q_j + dq_j                C' = C - dq_j m_j p_j - f
  At unchanged marks: E' - E = -dq_j m_j(p_j - v_j) - f

Mandatory signed cashflow CF:       C' = C + CF
Cash interest I, borrow charges D:  C' = C + I - D
Margin cash transfer x:            R' = R + x; C and E unchanged
```

**ORIGINATE:** start with declared initial risk capital `K_cap > 0` and no positions. This is the sole external contribution. At inception, book the liability and gross premium `P_0 = -q_B m_B p_B > 0` exactly once from released inputs. Immediately afterward, `C=K_cap+P_0-f_0` and `E=K_cap+P_0+q_B m_B v_B(0)-f_0`. Selling at the mark leaves equity `K_cap-f_0`; premium is not free profit or the entire capital base. The fixture sets `p_B=v_B(0)` as an origination assumption, not evidence of a tradable basket quote. The P&L baseline is `K_cap`.

**IMPORT:** load opening cash, signed positions, restrictions and opening marked equity, together with opening stock-short proceeds lots required for FIFO release. Do not credit historical premiums again. Validate instrument units, opening equity against opening marks, and restrictions against the lot records (including outstanding short quantities and their gross proceeds). Malformed/inconsistent records are input errors; valid records describing insufficient cash, equity or borrow produce a recorded inception breach, not an input-validation exclusion. The P&L baseline is opening marked equity. In either mode denote the applicable baseline by `E_base`.

Buying an option exchanges cash for its marked asset; selling exchanges cash for a short liability. Actual fill cash includes spread/slippage; never debit these a second time. Fees and borrow charges have separate, uniquely identified postings. A collateral transfer only reallocates existing cash. There are no later contributions, withdrawals or automatic financing injections. The cash ledger has no relationship to any future regression intercept.

**Borrow/proceeds policy:** short stock requires a known locate and sufficient per-stock borrow limit. Lock the gross proceeds of each newly opened short lot in `Q`; fees are paid from available cash. Those proceeds earn no rebate, cannot fund other purchases and are ineligible for margin. A cover releases the matching proceeds using FIFO lots, pro rata for partial covers; any profit/loss changes available cash. Crossing zero closes existing exposure before opening a new lot. No naked stock shorting or assumed unlimited borrow. A borrow recall/limit breach stops the run; no assumed buy-in.

Accrue over `[previous event,t)` using holdings, balances and rates effective in that interval: ACT/365 simple interest `I=(r_A A + r_R R) dt` for feasible nonnegative balances, and stock borrow charge `D=sum_i b_i max(-q_i,0) S_i(previous) dt`. Rate changes create interval boundaries. Rates and borrow fees are explicit toy inputs, not estimated funding terms; `Q` earns zero. Discretionary cash overdrafts are prohibited. Mandatory obligations are still booked in full if unfunded, exposing a cash deficit and stopping the run; this is not successful financed settlement.

Interest is credited at each event boundary, so subsequent intervals accrue on that updated cash. Extra strategy-generated order timers can therefore change financing even without fills: 1,000 cash at annual rate 0.365 becomes 1,002 after one two-day interval, versus 1,002.001 after two one-day intervals. This declared toy compounding convention has no separate broker interest-crediting schedule.

At expiry, first mark the option to its observed contractual unit payoff `h_j`, then atomically post `CF_j=q_j m_j h_j` and remove it. Equity is unchanged by this cash-for-position conversion; the final mark movement already records the economic gain/loss. Process even zero payoff and enforce a unique settlement ID. Official fixing values for the same stock and fixing time must agree exactly across contracts; conflicting supplied values are malformed input and rejected before that event's mandatory postings. The fixture has no settlement lag; a real lag would require a receivable/payable, not premature cash or disappearance of value.

## 4. Data and valuation contract

Only four boundaries are needed: replay supplies released observations; an in-process pricing component supplies marks/deltas/stress marks; the policy supplies target holdings; the engine owns cash, positions, orders, execution checks and reporting. Targets are absolute signed quantities in registry units (shares or contracts). Omitted instruments retain their holdings and valid pending orders; `None` leaves all valid orders alone. The configured hedge universe restricts what a policy may trade, always excluding the basket. Pricing mathematics stays outside the accounting/event loop. Replacing pricing or the policy must not change the ledger. The replay clock privately reads one following record to merge due times; unreleased contents never enter valuation, accounting or policy views.

That following record is parsed and validated before release. A malformed financial payload can raise `InputError` before an earlier generated timer or a completed horizon returns, including when the record's valid availability time is beyond the horizon. This is malformed input, not a recorded data-gap outcome. All same-time horizon records must still be consumed before completion; this parsing convention does not expose future contents to the strategy.

Required inputs, frozen and identified for each run:

| Record | Required contents |
| --- | --- |
| Instruments/opening state | IDs, currency, units, multipliers, basket coefficients, payoff/strike, expiry/fixing/settlement times, tradability, opening quantities/cash/restrictions or toy origination terms |
| Market observations | Source, instrument/factor, observation time, `available_at`, sequence ID, value, quality and freshness; distinct official fixing records |
| Execution quotes | Instrument, contemporaneous bid/ask, executable time, availability, valid size and quote ID; never a basket model mark |
| Valuation packet | As-of time, input and calibration cutoffs, model/version/parameters, unit marks, unit constituent deltas, stress marks, source observation times and pricing RNG configuration |
| Rules/calendar | Start/horizon, optional decision flags, execution delay and order lifetime/deadlines, lot/order/position limits, borrow availability/fees, cash rates, fees, freshness limits, margin proxy/declared scenarios/buffer |

The policy receives an immutable view of released history (`available_at <= decision time`), current observations/quotes with their original timestamps, the matching valuation packet, positions/cash/restrictions, live pending orders, rule configuration and order/fill history. Parameters/calibration must also have been available then. It cannot read future realized spots, future fixings, the source iterator or the RNG used to generate the replay.

Pricing implements `value(PricingView) -> ValuationPacket` in the same Python process. The view contains only released inputs and a supplied packet if present, not portfolio mutations or future data. Explicit packets support hand-authored fixtures; they must have the current as-of time. A generated clock boundary without a supplied packet requires a pricer, otherwise it is a data gap. Never refresh an old packet's timestamp to invent a current valuation. Previously released pricing inputs are usable only within configured age limits.

Process by availability time with deterministic sequence ordering. Quote event time and availability are separate: a delayed historical observation is not an executable price. V1 execution events require contemporaneous actionable quotes (`quote time = available_at = execution time`); a bar adapter must explicitly supply later executable prices, such as next-open quotes. A close used to make a decision cannot also fill that decision's order. Do not infer spread or intrabar paths from OHLC data.

Reject invalid units, duplicate conflicting IDs, crossed quotes, or unsupported corporate actions. Reject fills using stale/missing quotes; leave orders pending only until their expiry. No interpolation from future data. Carry-forward mark inputs are allowed only within the configured age limit, preserving their original timestamps. If an essential mark, fixing or margin input cannot be validly obtained, cancel orders and stop as `STOP_DATA_GAP`; report latest known balances, unprocessed obligations and time-stamped last valid equity, leaving current unavailable values unknown. Record any independently known cash, borrow or position-limit breach first; then use `STOP_WITH_BREACH` with an accompanying data-gap flag. Missing execution quotes alone do not prevent a margin check with otherwise valid marks. Missing deltas alone skip the decision with a recorded reason.

With valuation missing, `max(B_safe - (C - Q), 0)` is a known lower bound on the cash requirement shortfall because margin is nonnegative; a positive bound beyond tolerance is already a breach. Report it as `cash_requirement_shortfall_lower_bound`, while required margin and full shortfalls remain unknown. When fully valued, this field equals the actual cash requirement shortfall.

**Valuation proposal:** fixed-parameter correlated lognormal dynamics with zero dividends. Under the pricing measure, `dS_i/S_i = r dt + sigma_i dW_i`, with declared constant correlation. Basket value is discounted Monte Carlo average payoff; central spot differences estimate each delta using common pricing draws across bumps, with bump size and sample count recorded. Constituent European options use the corresponding Black–Scholes formula; terminal values use the official fixing exactly. Synthetic quotes are centered on replay spots or these option model marks. Hand-authored accounting tests instead supply explicit marks and quotes centered on them, without calling a pricer. No future realized replay values enter pricing. Pricing seeds/streams are independent of market-path seeds, deterministic by event/model identity, and shared across strategy comparisons for identical inputs. Stress marks use the same valuation boundary. Monte Carlo numerical tolerance is a pricing check, not evidence about hedge profitability.

**Synthetic replay proposal:** one seeded three-stock correlated GBM path, twenty toy business days with open/close observations, fixed declared toy volatility/correlation and zero replay drift. The calendar includes an inception observation and one post-expiry execution event. The price path, bid/ask spreads, fees, borrow limits/rates and capital are recorded fixture inputs; untrained simulation supplies test data, not a policy forecast. Separate deterministic perturbations test shocks and failures. No real market access is assumed. Toy pricing and replay sharing a model validates mechanics and can understate model risk.

## 5. Margin and infeasibility

Use **Unnetted Liability plus Stress Cash Proxy (ULSCP)**, explicitly an uncalibrated research rule, not broker/exchange compliance. Let `V` include all marked noncash positions. For an explicitly supplied finite toy stress set `Omega` (including the unchanged state), let `V^s` be their value under stress s:

```text
M = sum_nonspot max(-q_j m_j v_j, 0) + max_s max(V - V^s, 0)
G = max(C - Q, 0)                       eligible cash capacity, including R
B_safe >= 0                            required safety buffer in currency

Required: E >= M + B_safe and C - Q >= M + B_safe
If feasible: R := M; A := C - Q - R      buffer remains available cash

s_equity = max(M + B_safe - E, 0)
s_cash   = max(M + B_safe - (C - Q), 0)
shortfall = max(s_equity, s_cash)        do not add duplicate deficits
```

The first term reserves gross short-option marked liabilities without long-option offsets; the second reserves additional portfolio stress loss. Both cash backing and net equity are required. `G`, `M`, and `B_safe` are not extra assets added to equity; `G` is capacity, not another cash balance. Report margin-only gaps using the same formulas with `B_safe=0`, and buffer gaps separately. Negative equity or insufficient cash fails even when margin and buffer are zero. Stress shocks and buffer are mandatory, named toy inputs chosen before the run; no calibrated percentages or broker thresholds are asserted here. No variation margin is additionally charged as P&L under this proxy.

Declare the complete finite stress names in `expected_stress_scenarios`, including `unchanged`; missing whole scenarios are data gaps, just as missing instrument stress marks are. For legacy cases without a declaration, freeze the inception packet's names for the rest of the run. This compatibility mode cannot detect a scenario already omitted at inception; new cases should declare the set explicitly.

At every market record and clock boundary, evaluate the actual state after its mandatory posting phase and before pending fills or policy calls; also check after every fill. Capture current `E,C,A,R,Q,G,M,B_safe` and shortfalls **before any response**. If infeasible, record the breach and return `STOP_WITH_BREACH`, canceling orders and retaining positions. Routine redistribution of existing collateral occurs only after the capacity check; it cannot create equity or cure a resource shortfall. No forced liquidation, basket closeout, order shrinking, or cash injection.

For each eligible order, preview its entire fill at the current bid/ask, including size/lot limits, fees, FIFO proceeds release/lock, borrow capacity and required collateral. Reject insufficient quoted size or other infeasibility without partial fills, ledger postings or fees; record projected shortfalls separately from actual breaches. Process instruments sequentially in fixed instrument-ID order, checking actual risk after each fill; earlier fills remain if a later order fails. Never assume margin is monotonic in trade size or that a multi-stock fill is atomic. Scripted option-accounting tests explicitly enable constituent puts/calls in this same execution path; the two active baselines enable only stocks. The basket is always excluded.

## 6. Timeline and mechanical baselines

The replay clock merges market availability/sequence records, registered contract dates, live order activation/deadline times, configured start and horizon. Funding/borrow changes arrive as timestamped records. Source records at the same time retain their sequence order and replace a redundant clock-only boundary. Every source sequence at horizon `H` is processed before finalization. Generated boundaries contain no observations, fixings or executable quotes.

The engine evaluates the strategy at every boundary. `ScheduledDeltaPolicy` returns `None` outside source decision flags; the original fixtures flag closes, including inception and expiry, and provide next-open quotes. The callable stock-delta rule evaluates each boundary. Earliest execution respects both the configured delay and any later time supplied by the source record. Use an explicit source deadline when supplied; otherwise use configured order lifetime, capped at `H`. Ordinary deadlines are exclusive; terminal/default-horizon orders include `H`'s execution phase. Even zero delay requires a later quote observed strictly after submission. With coarser data, market moves and breaches between observations remain unknown.

The toy event's mandatory phase nets accrued charges/interest and simultaneous contractual cashflows atomically, retaining separate ledger entries. Check the resulting state; do not claim observation of intermediate deficits inside that declared netting phase. Different timestamps/events are never netted. This funding convention must be replaced if actual settlement rules require sequential funding.

At each available event, in this order:

1. Publish only this event's released observations/rule changes. Validate required fixings and mandatory-flow inputs before committing the posting phase; if missing, leave the whole phase unposted and report the data gap/outstanding obligations. Record any independently known cash/borrow breach even with missing valuation data.
2. Expire orders whose deadline has arrived and cancel option orders at their fixing cutoff. Accrue interest/borrow charges using the previous interval's state and rates. In ORIGINATE mode only, obtain the inception valuation from released inputs and book its actual sale premium and liability exactly once; IMPORT starts from the validated opening book without another premium. Apply due contractual cashflows and expiry settlement once, using official fixings; commit this mandatory phase atomically. Newly released rates apply to subsequent accrual intervals. Basket settlement cancels all remaining opening hedge orders before execution.
3. Apply the as-of valuation packet to remaining positions. Check and log actual margin, funding and borrow feasibility **before any trade**, even when no decision is scheduled. Stop immediately on a breach. Otherwise adjust the margin cash allocation from existing funds.
4. At an execution opportunity with fresh quotes, consider previously submitted, unexpired orders whose execution time is strictly later than submission and whose quote is newer than submission. Validate and fill sequentially as above, checking immediately after each fill. A timer alone supplies no execution price.
5. After those checks/fills, evaluate the policy on its immutable view. Missing policy inputs or a `None` result are recorded separately and preserve valid pending orders. For supplied targets, validate permissions/lots/position limits; keep compatible pending orders and immediately cancel obsolete ones. Submit target minus holdings minus compatible pending quantity, once, with eligibility/deadline attached. After actual basket settlement, the engine requests zero hedge holdings without consulting the policy. New orders cannot execute in this event. Record and reconcile the complete state.

For a live delta hedge, define for each stock i:

```text
D_nonspot,i = sum_j in ALL NON-SPOT positions q_j m_j (partial v_j / partial S_i)
stock_target_i = -D_nonspot,i
additional_order_i = stock_target_i - current_stock_shares_i - compatible_pending_i
```

This includes the liability and any currently held constituent options; exclude existing stock from the target calculation. The stock baseline targets zero constituent options but includes their delta until they actually close. With no constituent options held, this equals negative basket delta. Round stock targets to the nearest configured lot, with ties to even; zero lot size permits fractional shares. One short put with long-option unit delta `-0.40`, quantity `-1` and multiplier `100` has nonspot delta `+40` shares. Target `-40` shares; with `-10` held and no pending order, sell `30` more. Repeating that target while the sell order remains pending creates no duplicate. Log residual delta after delayed execution; exact neutrality is not promised. The unhedged rule targets zero permitted hedge holdings. Both baselines receive identical capital, liability, replay, valuation, costs and constraints; neither selects hedge options.

After actual basket settlement, the engine targets zero permitted hedge holdings. It submits ordinary closing orders for later execution, subject to the same quotes and constraints, including constituent options that have not expired. The common horizon `H` allows financing until actual exit. Completion requires the basket settled and zero residual inventory; open options remain obligations until settlement or an actual closing trade. Missing or rejected final exits produce `INCOMPLETE_EXIT`, with residual marked inventory; never silently liquidate it at a mark.

## 7. Reporting and reconciliation

**Net account P&L at t is `E(t)-E_base`**, in currency, including unrealized value: `E_base=K_cap` for ORIGINATE and opening marked equity for IMPORT. A completed cash-only run has `E(H)=C(H)`, with `R=Q=0`. At inception and after every posting, reconcile:

```text
L(t), ORIGINATE = P_0 + MV_B(t) + sum signed liability settlements through t
L(t), IMPORT = MV_B(t) - MV_B(opening) + sum subsequent signed settlements
Hedge(t) = sum_hedges [MV_j(t) - MV_j(0-)
                      - sum_fills dq_j m_j v_ref,j + sum lifecycle CF_j]
X = sum_hedge fills dq_j m_j (fill_j - v_ref,j)
TC = X + all fees (including inception/settlement fees if specified)
FC = borrow charges - available-cash and collateral interest
E(t) - E_base = L(t) + Hedge(t) - TC(t) - FC(t)
```

`v_ref` is the contemporaneous pre-fill accounting mark; tradable fixture/test marks are quote midpoints. Report `X` as the signed fill-to-mark execution cost and fees separately. Future model/quote basis differences need separate disclosure; they are not automatically bid/ask costs. `L` and `Hedge` are gross economic P&L components, not sums of cash spent; inception premium/mark mismatch belongs to `L`. An imported book instead uses opening marked values and subsequent cashflows, with no synthetic premium credit.

Report cash/collateral trajectories and extrema: minimum `C`, minimum `A`, peak `R`, peak `Q`, peak locked cash `R+Q`, peak required margin, and minimum buffer headroom `A-B_safe` after allocation. Financing reports interest credits, borrow charges and their net `FC`. Stock turnover is `sum_fills |dq_i| p_i` currency traded; option-test premium turnover is `sum |dq_j|m_j p_j`. Report separately and optionally divide hedge turnover by fixed `N_0`; fees are excluded and cover trades count. Liability origination/settlement are reported separately from hedge turnover.

Report `accepted_hedge_fill_count` as the number of executed hedge fills, including exits; exclude origination and contractual settlement.

Record every observed breach with phase/time/reason, required resources, actual resources, margin-only and buffer shortfalls; report their maximum **through the observed stopping time**, not an unobserved future maximum. With immediate stop, at most one actual breach event occurs per strategy run. Keep infeasible-order rejections separate. Report each settlement's fixing, signed amount, posting ID, funding status and removed quantity, plus final unwind fills and remaining inventory.

Optional **net premium retained** means completed-run net economic P&L `E(H)-E_base`; **premium-retention ratio** is that amount divided by known gross original `P_0>0`. For IMPORT, historical premium is denominator metadata only, never a cash credit; the numerator measures only the imported observation horizon. It includes hedge gains/losses, all costs and interest. It can be negative or exceed one; do not clamp it or use premium minus purchases. Label it a normalization of net P&L, not literal unspent premium cash.

Include all strategies and statuses. A stop reports marked P&L at stop and open obligations, with terminal settlement/P&L/retention unavailable unless actually completed. For unequal horizons, show a matched last-common valid checkpoint comparison alongside each outcome; do not rank stop-time and maturity P&L as comparable. A data gap reports last valid equity. One synthetic path supports ledger traces, not statistical confidence, distributional RMSE, VaR/CVaR, or claims about real profitability, tail protection, optimality or margin compliance.

## 8. Acceptance tests for the first implementation

1. **Capital/premium and option purchases:** with capital 1,000, sell a marked option for 200: cash 1,200, liability -200, equity 1,000 before fees. Buy a marked option for 200: cash falls 200 and assets rise 200. Test both puts/calls, long/short, non-unit multipliers and all moneyness cases. Importing a short credits no historical premium.
2. **Self-financing and costs:** unchanged-mark trades/round trips change equity only by signed fill-to-mark costs and fees, each once. Reconcile the reporting identity after every event. Rejected orders leave balances/fees unchanged.
3. **Delta sign/units and live orders:** the short-put example targets -40 shares and orders -30 from -10; repeating the target while an order is pending or after its fill places no duplicate. Include additional constituent options in nonspot delta until they actually close. Target option holdings are zero for the stock baseline. Reject any basket hedge order. `None`, omitted targets and missing deltas preserve valid pending orders; changed targets cancel obsolete quantities immediately.
4. **Restricted funds:** a stock short at its mark raises `C` and `Q` equally, leaving equity/free cash unchanged before fees and any margin change. Partial FIFO cover releases only matching proceeds. Collateral transfers preserve equity. Test borrow caps and known interest/borrow accrual across a rate change.
5. **Expiry once:** a short option revalued from 3 to 4 with multiplier 100 loses 100; settling at 4 and removing it adds no further equity jump. Duplicate settlement events are idempotent. Test zero payoff and insufficient cash: book the full obligation, flag unfunded settlement and stop without capital injection.
6. **Causality and clock:** after observing close 100, the order can fill only at later open 110 bid/ask, never close 100, even with zero delay. Changing future rows leaves all earlier marks/decisions/orders unchanged. Test delayed availability, quote age, dynamically added activation/deadline boundaries and missing critical data. A timer cannot refresh old marks or create a quote; all source sequences at the horizon remain eligible for processing.
7. **Breach before rebalance:** a gap at an unscheduled event triggers a pretrade breach with an old hedge order pending. No pending fill or new decision may cure/erase it; preserve the liability and the strategy's failed result. Test a cash-only deficit and a negative-equity state when margin/buffer are zero. Omitting a declared whole stress scenario produces a data gap, rather than lowering required margin silently.
8. **Execution/horizon:** an infeasible complete order is rejected without resizing; an earlier valid fill survives a later rejection. Test successful normal exit and missing terminal quotes with residual inventory and incomplete status. No fake model-price liquidation.
9. **Pricing isolation:** distinct replay/pricing RNG streams; identical valuation packets across baselines; basket one-stock reduction agrees with European put valuation within declared numerical tolerance; positive scaling of quantity/multiplier scales value/delta once. A change to MC sample count must not change the replay path.

## 9. Five material questions for Daniel

1. What is the actual liability payoff, quantity/notional convention, lifecycle and settlement, and can we obtain the project brief and intended paper list?
2. Which point-in-time stock/option quotes, historical contract metadata, funding/borrow data and official fixings are available, at what frequency and with what timestamps?
3. What are the broker's margin, collateral eligibility/haircut, netting, settlement-liquidity and breach-response rules?
4. What capital/credit is committed, and what are the stock-borrow, short-proceeds, permitted-instrument, execution, size and trading-calendar constraints?
5. What horizon, economic objective and success criteria should govern comparison, including treatment of failures and the meaning of premium retention?
