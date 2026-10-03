# Cont–Vuletić optimization prototype: Stage 1

This stage adds conditional scenario hedging behind the existing `strategy(view) -> target holdings` interface. It implements the regression and turnover penalty in [Cont and Vuletić, *Data-driven hedging with generative models*, Eq. (20) and the coordinate-descent appendix](https://link.springer.com/article/10.1007/s10479-025-06867-3). Scenarios come from a declared synthetic model. There is no trained generator, VolGAN, neural network or reproduction of the paper's empirical results.

## Economic signs and units

The candidate set is an explicit subset of the configured hedge universe: constituent stocks and optionally European constituent calls/puts. The basket is never eligible. Quantities remain signed shares or signed option contracts.

Let `L` be the signed marked value of held positions outside the candidate set. Normally this is the short basket; any omitted held positions also remain fixed exposure for this decision. Candidate holdings enter through the optimization variables, so they are not counted again in `L`.

```text
L_k = sum_fixed_positions q_j * m_j * v_j(scenario k)
DeltaV_k = -(L_k - L_now)
DeltaH[k, i] = m_i * (v_i(scenario k) - v_i(now))
phi_prev[i] = actual current holding of candidate i
```

The minus sign in `DeltaV` is essential. The paper's tracking residual is target value minus tracking-portfolio value. The engine instead adds hedge P&L to its signed liability. With this mapping:

```text
fixed-book change + hedge change = -(DeltaV - DeltaH @ phi)
```

Thus the optimizer's `phi` can be returned directly as actual target holdings. For example, a short put whose signed book delta is +40 shares requires a stock target near -40. Feeding +40 shares of book exposure directly into the minus-sign regression would reverse the hedge.

Option value changes include the contract multiplier once. Cost weights `c_i` are currency per corresponding holding unit: per share or per contract. The solver uses these physical units without feature standardization, hidden notional normalization or output rounding.

## Objective and gross scale

For `N` joint scenarios and `J` candidates, solve:

```text
min over A, phi:
    mean_k (DeltaV_k - A - sum_i phi_i * DeltaH[k, i])**2
    + alpha * g0 * sum_i c_i * abs(phi_i - phi_prev_i)
```

`alpha >= 0` is dimensionless; `c_i >= 0` is a configured proportional transaction-cost proxy. The squared residual has currency-squared units. Multiplying the turnover-cost proxy by currency-valued `g0` gives the penalty the same units.

The paper describes `g0` as initial gross position and identifies the gross value of its long straddle with the straddle's value. [Source: Eq. (20) and section 4.2.](https://link.springer.com/article/10.1007/s10479-025-06867-3) For this prototype, the corresponding rule is:

```text
g0 = sum_fixed_positions abs(q_j * m_j * initial_unit_mark_j)
```

For one basket this is `abs(q_B * m_B * initial_unit_mark_B)`. This explicit extension to signed books uses gross marked target value, not risk capital or underlying notional. Automatic initialization latches the scale on the first strategy callback and holds it fixed thereafter; the demonstration starts at inception. For a strategy started later, that callback is its opening reference rather than the original contract's inception. A positive gross scale is required.

The quadratic term has a positive-semidefinite Hessian, including when candidates are redundant. The nonnegative weighted absolute-value penalty is convex. Their sum is a convex generalized-LASSO objective in changes of position. Redundancy can make optimal holdings nonunique; convexity does not imply uniqueness.

## Solver and intercept

Use the change of position and adjusted response:

```text
beta = phi - phi_prev
y = DeltaV - DeltaH @ phi_prev
X = DeltaH
Xc = X - column_means(X)
yc = y - mean(y)
lambda_i = alpha * g0 * c_i
```

Centering removes the unpenalized intercept. With other coordinates fixed, the update is:

```text
partial_residual = yc - Xc @ beta + Xc[:, i] * beta_i
z = mean(Xc[:, i] * partial_residual)
d = mean(Xc[:, i]**2)
beta_i = sign(z) * max(abs(z) - lambda_i / 2, 0) / d

phi = phi_prev + beta
A = mean(y) - column_means(X) @ beta
```

The threshold is `lambda_i / 2` because the loss uses `1/N`, not `1/(2*N)`. Constant centered columns need no change and avoid division by zero. When all penalties are zero, least squares uses a numerical decomposition; for redundant columns it selects the minimum-norm change in holdings. This is an ordinary-least-squares optimum, although its coefficients need not match a different tie-breaking convention.

Coordinate descent otherwise uses a fixed instrument order. With residual `r = yc - Xc @ beta` and gradient `g = -2 * Xc.T @ r / N`, the stopping test checks:

```text
beta_i != 0: abs(g_i + lambda_i * sign(beta_i))
beta_i == 0: max(abs(g_i) - lambda_i, 0)
```

Each violation is divided by `max(1, abs(2 * Xc[:, i].T @ yc / N), lambda_i)`. The maximum must be at most `1e-9` by default; the default limit is 20,000 sweeps. Failure to satisfy the tolerance raises a convergence error. These are documented floating-point checks, not exact optimality certificates.

`A` estimates the scenario mean residual. It is never booked as cash, financing income or external capital. The engine alone maintains the self-financing cash account. The penalty uses current **actual holdings**, not the previous requested targets or unfilled orders; the engine separately reconciles compatible pending quantities.

## Conditional scenarios and execution

At each scheduled decision, the generator uses only released observations, declared parameters and schedule timestamps. It generates joint GBM spot scenarios with zero physical drift and fixed toy volatilities/correlation. The current event supplies `next_rebalance_time`; the existing read-only `view.rules` supplies this timestamp and `earliest_execution`, using the same effective eligibility calculation as order submission. Neither field is a future price or quote.

```text
decision_time = view.time
earliest_execution = max(event's declared eligibility, decision_time + configured delay)
horizon = min(next_rebalance_time, earliest relevant option expiry)
require horizon > earliest_execution and horizon > decision_time
delta_t = (horizon - decision_time) in ACT/365 calendar years
```

Missing schedule information or an expiry/review leaving no interval after execution eligibility raises a policy `DataGap`: the engine records a skipped decision and preserves compatible pending orders. Malformed or past timestamps are input errors. There is no fixed 24-hour fallback and no scan of future replay records. Expiry's terminal unwind bypasses optimization; its next rebalance is absent because no subsequent hedge review is scheduled.

The synthetic calendar is constructed before price generation. Ordinary close decisions model the next business-day close, 24 hours later, with execution eligible after 17 hours. Friday decisions model Monday's close, 72 hours later, with execution eligible after 65 hours. The GBM drift correction and diffusion use this actual elapsed time, including weekends. Unchanged 24-hour intervals retain the original random draws and numerical scenario values.

The policy still approximates decision-to-review changes with one target holding. Actual current holdings remain in place before fills; missing quotes or constraints can further delay or prevent execution. The change aligns the modeled endpoint with the declared trading schedule but does not model separate pre-fill/post-fill P&L or establish minimum realized variance under delayed execution. Decision records and optimizer diagnostics expose the timing for audit.

Scenario randomness is separate from replay randomness and pricing randomness, with stable seed derivation from configured seed, decision time and scenario identity. Changing the scenario count or pricing sample count cannot change the replay path. The existing `BasketValuator` reprices the hypothetical states under its fixed risk-neutral parameters, sharing pricing draws across comparable scenario valuations. Hypothetical expiry values use exact contractual payoffs; actual settlement still requires the replay's official fixing.

The solver returns unconstrained continuous targets. Borrow, cash, margin, position limits, lots, quotes, execution delay and actual costs remain engine responsibilities. Infeasible orders are visibly rejected; the strategy does not resize them or mutate the account. Configured lots can reject an optimizer output: this stage does not round it and then claim the rounded holdings solve the convex problem.

The linear cost proxy influences target selection. It is not another ledger charge or an exact representation of fixed per-order fees. Execution continues to book the actual configured spreads, slippage and fees once. L1 encourages sparse **changes**; previously held instruments may remain nonzero. For a fixed scenario problem, increasing alpha weakly reduces the weighted turnover penalty, not necessarily every turnover measure along an entire backtest.

## Hand-checkable examples

With `DeltaH = [-1, 1]`, `DeltaV = 2 * DeltaH`, previous holding zero and alpha zero, the solution is `phi = 2`, `A = 0`, residual MSE zero. Adding a constant 3 to `DeltaV` changes only the intercept to 3. With `alpha * g0 * c = 1`, the latter example gives `phi = 1.5`, which checks the threshold's factor of two independently.

For two identical columns `X1 = X2 = [-1, 1]`, response `2 * X1`, previous holdings zero, `alpha * g0 = 1` and costs `(1, 2)`, the optimum is `(phi1, phi2) = (1.5, 0)`. Both instruments hedge the same change, but the second has the larger turnover penalty. With zero penalty, their individual coefficients are not uniquely identified.

## Running and interpreting the comparison

After installation, run from the repository root:

```powershell
python -m basket_hedging compare-scenarios --case fixtures/synthetic.json --output reports/cont_vuletic_stage1 --scenarios 256 --scenario-seed 9173 --pricing-samples 1024 --alpha 0.1 --cost-proxy 0.02
```

The four policies are `unhedged`, `delta`, `variance` (alpha zero) and `cont_vuletic_l1` (positive alpha). They share the same saved artificial path, opening account and execution/risk rules. The stated toy parameters are chosen before the run, without fitting alpha to make a policy win.

Older synthetic files without `next_rebalance_time` must be regenerated with the existing `generate` command to exercise optimization; otherwise the missing schedule is reported as skipped policy decisions. The old fixed-duration `step_seconds`/`--step-hours` configuration has been removed. The strategy call remains `strategy(view) -> target holdings`.

Ordinary engine reports preserve the account and order traces. The additional `research_summary.json` reports P&L, turnover, actual trading costs, accepted trades, instruments used and failure status. Its observed gross hedge error is:

```text
gross_hedge_error_currency = -(liability_pnl + hedge_pnl)
```

This is a cumulative gross tracking residual with costs and financing excluded and reported separately. It is distinct from the optimizer's in-sample scenario residual MSE. No regression intercept is subtracted from the realized account. Comparisons retain failures and use common valid pretrade checkpoints; a stopped path receives no invented maturity result. One path does not establish statistical confidence, tail protection or profitability.

## Differences from the paper and Stage 2

This stage uses fixed-parameter synthetic spot scenarios and model repricing, with no learned joint dynamics of spots and implied-volatility surfaces. Its execution is delayed and constrained by the existing engine. Alpha and proportional cost proxies are explicit inputs; no AIC selection or empirical calibration is performed. The demonstration liability, capital, cash settlement, borrow and research margin conventions remain repository assumptions rather than Polykern facts.

The subsequent [Stage 2 synthetic prototype](cont_vuletic_stage2.md) supplies a learned conditional Gaussian scenario provider with released inputs and an explicit training cutoff. It preserves this optimizer and the scenario units, timestamps and repricing path. Historical spot/option data and learned joint implied-volatility surfaces remain outside that prototype. The strategy still returns targets, and accounting, execution and margin remain unchanged.
