# Basket hedging research backtester

This package replays a signed derivatives book with causal execution, self-financing accounting and recorded funding/margin failures. It supports a short arithmetic-basket call or put, constituent shares and European constituent calls/puts. The reusable engine handles the account; a separate policy chooses absolute holdings. The original baselines are unhedged and mechanical stock delta. An opt-in Stage 1 comparison adds variance and transaction-cost-regularized hedges using synthetic conditional scenarios, with no trained model.

The financial contract is [the specification](docs/backtest_spec.md), with [short pseudocode](docs/backtest_pseudocode.md). These reflect the final outline: persistent pending orders, a clock that includes contract and order dates, and replaceable in-process pricing. ORIGINATE and IMPORT retain their distinct P&L baselines. This is an auditable research prototype, not a production-readiness or broker-compliance claim.

## Install, test and reproduce

The new **historical SPY/QQQ/IWM VolGAN adaptation** has a separate
[data and run guide](docs/historical_benchmark.md). It uses the specified six
hash-verified release files and a joint adversarial return/surface generator.
The original files lack IWM dividend and split history. A separate verified
iShares distribution supplement and two-source split check supply those fields;
prices, model inputs and original file hashes are preserved. The historical
command uses the frozen trained models. The synthetic demonstration below
remains separate and unchanged.

The [completed historical comparison](reports/historical/benchmark/benchmark.md)
contains 71 scheduled contracts and five policies. Its common terminal sample
has 62 episodes: five shared data-gap episodes and four additional VolGAN solver
failures remain visible. Delta has lower hedge-error SD than VolGAN on this
sample. The guide explains the four charts and the unchanged weak calibration.

For the **Cont–Vuletić workflow demonstration**, run [examples/paper_demo.py](examples/paper_demo.py):

```powershell
python -m pip install -e ".[demo]"
python -B examples/paper_demo.py
Start-Process reports/paper_demo/next_day.png
Start-Process reports/paper_demo/backtest.png
```

This loads the saved model in `reports/stage2/model.json`; it does not retrain it.
At each scheduled decision, the model generates 256 joint scenarios for the next
review time. The existing pricer values the basket in each scenario; the existing
regression chooses stock positions with the Stage 2 settings (alpha 0.1, cost
weight 0.02 per share, 1,024 pricing draws). The engine then submits orders and
executes them against later quotes from the saved synthetic replay. The scenario
horizon follows the schedule, including weekends. No engine logic is changed.

Show `next_day.png` first: on 5 January at 16:00 UTC, the targets are approximately
-15.91, -15.55 and -16.07 shares. They fill on 6 January at 09:00 UTC; the scenarios
end at that day's 16:00 review. The black diamonds show the later replay outcome,
added only after the backtest. Each scenario is one possible joint future.
Then show `backtest.png`: market prices,
net account P&L for three policies, and actual holdings through settlement and exit.

All three runs complete: terminal P&L is -55.38 unhedged, -144.60 mechanical delta,
and -145.12 with the generated-scenario hedge. The last policy makes 61 fills.
The model is a conditional Gaussian generator trained on synthetic data; this is
a simple implementation of the scenario-to-hedge workflow, not VolGAN or a
reproduction of the paper's empirical results. One path does not measure variance
reduction; use the saved [200-path validation](reports/stage2/validation_report.md)
for that discussion. Nothing is retuned for this demonstration.

For the code walkthrough, follow `paper_demo.py`'s `main()` into
`learned_scenarios.py:generate()`, `scenario_hedge.py:__call__()`,
`optimization.py:solve_hedge()`, then `engine.py:run()`.
`first_decision.json` records the generated stock scenarios, targets and actual
first fills; `model_and_decisions.json` records model provenance and each hedge
decision. The normal account, order and posting reports are saved alongside them.

For a separate **accounting and execution demonstration**, use [examples/demo.py](examples/demo.py):

```powershell
python -m pip install -e ".[test,demo]"
python -B examples/demo.py
```

This runs the saved hand-calculated, synthetic and margin-breach cases with both
baseline policies. It writes ordinary account reports and a `run.png` chart for
each case under `reports/demo/`. Charts show net account P&L, actual stock holdings,
and equity/cash backing above the margin-plus-buffer requirement. They stop where
the run stops. The script does not train a model or change engine settings.

Suggested five-minute demonstration:

1. Show `examples/demo.py`'s `main()`: load a case, choose a policy, run a fresh engine, report.
2. Show `policies.py`'s `stock_delta_hedge()`: return absolute stock targets.
3. Show `engine.py`'s `run()` and `ledger.py`'s `fill()`: check risk before trading, then book accepted fills once.
4. Open `reports/demo/synthetic/run.png`: explain P&L, holdings and funding capacity. Delta loses more on this path; the chart is a mechanics demonstration.
5. Open `reports/demo/gap/run.png`: the equity requirement fails before the old order fills; both policies stop with zero fills.

The deterministic example ends at P&L -200 unhedged and -371 delta, with nine
delta hedge fills. The synthetic example ends at approximately -55.38 and -144.60,
with 63 delta fills. The gap example stops at marked P&L -400 for both; it has no
terminal P&L. Open `summary.json`, `orders.csv` or `postings.csv` to inspect a number.

The engine demonstration is optional background for the paper-focused walkthrough;
rerunning training is unnecessary during a short presentation.

Run from this repository. Python 3.11 or newer is supported; this implementation was tested with Python 3.13.3, NumPy 2.3.1 and pytest 9.1.1. Runtime dependencies: NumPy `>=1.26,<3` and, on Windows, `tzdata>=2024.1` for historical market time zones. Tests use pytest `>=8,<10`; setuptools builds the small installable package. No lint or static-type checker is configured.

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python -m basket_hedging make-deterministic --output-dir fixtures
python -m basket_hedging generate --output fixtures/synthetic.json --replay-seed 20260912 --pricing-seed 731 --samples 8192 --constituents 3 --days 20
python -m basket_hedging run --case fixtures/deterministic_complete.json --policy both --output reports/deterministic
python -m basket_hedging run --case fixtures/synthetic.json --policy both --output reports/synthetic
python -m basket_hedging run --case fixtures/deterministic_gap.json --policy both --output reports/gap
python -m basket_hedging run --case fixtures/deterministic_missing_exit.json --policy both --output reports/missing_exit
python -m basket_hedging run --case fixtures/deterministic_rejected_exit.json --policy both --output reports/rejected_exit
git diff --check
```

`basket-hedging` is also installed as a console entry point. The `python -m` commands work even when the user's scripts directory is absent from PATH. Generation rewrites only its requested fixture files; report commands rewrite their requested report directories' named report files. They do not commit, push, upload or connect to a broker. A recorded experimental breach is a valid CLI result (exit code 0); malformed input or unreadable files return code 2. Reconciliation failures raise an error.

To evaluate the same delta rule at every clock update, add `--every-update` to a `run` command. To generate a basket-call demonstration, add `--basket-kind call` to `generate`; the default remains a put. These change the review schedule or liability payoff, not the ledger or execution model.

Focused stage checks:

```powershell
python -m pytest tests/test_accounting.py -q
python -m pytest tests/test_execution.py tests/test_replay.py tests/test_clock.py tests/test_order_lifecycle.py -q
python -m pytest tests/test_risk.py -q
python -m pytest tests/test_policies.py tests/test_end_to_end.py -q
python -m pytest tests/test_valuation.py tests/test_synthetic.py tests/test_pricing_clock_integration.py -q
python -m pytest tests/test_api.py -q
python -m pytest tests/test_reporting.py tests/test_cli.py -q
```

Tests use hand-calculated balances for the important accounting identities. They also cover immutable point-in-time inputs, quote capacity, exact deadlines, atomic settlement, failed previews, causal prefix invariance, separate RNG streams, cross-process pricing reproducibility, one-stock analytic comparisons and a ten-constituent smoke run. No test requires network access.

## Engine, policy and valuation

- `contracts.py` and `ledger.py` define units, opening records and the authoritative cash/position postings. Preview execution clones this ledger and calls the same methods as actual execution.
- `engine.py` owns funding, lifecycle processing, sequential execution and risk checks. Policies receive immutable positions, live pending orders, latest observations/quotes and released history, with no replay iterator or RNG.
- `policies.py` returns absolute signed targets, or `None` to leave orders alone. The engine subtracts held and compatible pending quantities once. Omitted targets and missing policy inputs preserve valid pending orders. Hedge eligibility is configuration; the basket is always excluded.
- `risk.py` contains one replaceable research margin formula. The test-only fixed-margin stubs isolate execution tests; they are not broker rules.
- `valuation.py` supplies in-process valuation packets. `PricingView` exposes released inputs; returned packets record input and calibration cutoffs. Pricing receives no future path. `synthetic.py` creates test observations independently of pricing draws.
- `replay.py` streams JSONL records and merges their availability times with live order activation/deadline times, registered option expiries, start and horizon. Funding/borrow changes are explicit input records. Clock boundaries never manufacture observations, quotes or marks. `reporting.py` exports the account trace and reconciled outcomes.

The engine evaluates the policy at every clock boundary after mandatory payments, risk checks and existing-order fills. `ScheduledDeltaPolicy` returns `None` outside source records flagged for a decision; this preserves the close-to-next-open demonstrations. The callable `stock_delta_hedge` evaluates at every boundary. Both target stock holdings against all currently held option deltas and target zero constituent-option holdings. An option hedge still contributes delta until its closing fill. For a book with only the basket and stocks, this equals negative basket delta. This convention is a documented research choice, not a claim about Polykern's preferred hedge.

For example, a target of 100 shares with 20 held and a compatible buy order for 80 requires no new order. A changed target cancels obsolete orders immediately and submits only the remaining difference. Orders and positions persist while new data arrives; the engine applies account changes in a deterministic sequence.

## Python entry point

The compact origination API follows the pseudocode. This example loads the hand-authored case through the normal input path:

```python
from basket_hedging import BacktestConfig, backtest_engine, stock_delta_hedge
from basket_hedging.replay import load_case

case = load_case("fixtures/deterministic_complete.json")
basket = case.instruments[case.opening.liability_id]
config = BacktestConfig(
    initial_capital=case.opening.initial_capital,
    hedge_instruments=tuple(i for i in case.instruments.values() if not i.is_basket),
    rules=case.rules,
    liability_quantity=case.opening.liability_quantity,
)
result = backtest_engine(basket, case.events(), stock_delta_hedge, config)
print(result.status, result.ledger.components())
```

For an existing book, use `Engine(instruments, opening, rules, margin_model, pricer).run(events, policy)` with `Opening("IMPORT", ...)`. Both entry points use the same ledger and execution code. A policy may be a callable or an object with `targets(view)`; it may return permitted stock/call/put targets without changing the engine.

Supply explicit `ValuationPacket` records for deterministic tests, or set `config.pricer` to an object implementing `value(PricingView) -> ValuationPacket`, such as `BasketValuator`. A packet must match the current clock time. Without a pricer, a new timer that lacks its own packet produces `STOP_DATA_GAP`; old packets are not restamped. A pricer may use previously released observations only within configured freshness limits, preserving their original timestamps. It never substitutes a model value for an executable quote or the actual inception sale price.

Packets contain unit marks and sensitivities, independent of portfolio quantities. The engine reuses the same packet for sequential fill previews at that market update and recomputes portfolio values and margin for each proposed or accepted position change.

## One trade, event by event

The complete deterministic fixture starts with capital 10,000 and originates one short basket put: quantity −1, multiplier 100 and unit premium/mark 2. Cash becomes 10,200, the liability is −200, and equity remains 10,000. The basket coefficients are fixed constituent shares per basket unit: one third of each of three stocks. They are not rebalanced portfolio weights.

1. After the inception close, supplied long-put unit deltas are −0.10 for each constituent. Each short-liability contribution is `−1 × 100 × −0.10 = +10` shares, so each stock target is −10 shares. These orders cannot use that close to execute.
2. At the next open, the engine applies current marks and checks the actual account before any pending order. For the first stock, mark 99 and bid 98.9 are separate inputs.
3. The whole ten-share sale is previewed with locate, borrow, size, cash and the full prospective margin calculation. On acceptance, gross proceeds are 989 and the fee is 1. Cash rises from 10,200 to 11,188; restricted short proceeds rise by 989. The stock liability is −990. Equity falls by exactly 2: spread cost 1 plus fee 1.
4. The engine checks the filled account, allocates its required margin from existing cash, then considers the next instrument. It does not make all three fills atomic or resize rejected orders.

After two stock-trading batches, deterministic cash is 16,098. Official expiry fixings give basket level 96 and a signed put settlement of −400. Cash becomes 15,698 and the option is removed once. The remaining short stocks still carry risk. The expiry decision submits a normal later cover, which fills at the following open's asks; final cash/equity is 9,629. Independent components are liability P&L −200, hedge P&L −150, execution cost 12 and fees 9: net P&L −371. The unhedged book ends at 9,800, or −200. These hand-authored prices exercise the mechanics; they were not fitted to improve the hedge's result.

## One breach, event by event

The gap fixture uses capital 1,000 and the same initial premium 200. At its next open, before any older hedge order fills, the basket's unit mark is 6 and its adverse stress mark is 8.

1. Cash is still 1,200 and the liability is now −600: equity is 600.
2. The proxy requires margin `600 + (800 − 600) = 800`. With the declared safety buffer 50, required equity is 850.
3. The engine records the pretrade equity-requirement shortfall 250: margin-only shortfall 200 and incremental buffer shortfall 50. Cash backing is sufficient in this example, so it is reported separately.
4. The run becomes `STOP_WITH_BREACH`; pending orders are canceled and the short basket remains on the book. There is no model-price closeout, forced-liquidation assumption or capital injection. Both policies remain in the comparison.

Separate deterministic cases with missing or insufficient final quotes produce `INCOMPLETE_EXIT` for the delta policy, preserving the remaining marked short stocks. Missing essential valuations/fixings instead produce `STOP_DATA_GAP`, or a breach with a gap flag when a cash, buffer, borrow or position-limit failure is independently known. Even with margin unknown, the known cash-shortfall lower bound is `max(buffer - (cash - restricted_short_proceeds), 0)`. A gap report distinguishes known balances from time-stamped last-valid equity and identifies unprocessed mandatory obligations.

## Accounting and opening modes

For instrument quantity `q`, multiplier `m` and unit mark `v`, position value is `q*m*v`. Stocks use signed shares and multiplier 1; options use signed contracts and the declared contract multiplier once. Buying an option exchanges cash for an asset; its premium is not charged again as a loss.

```text
equity = total_cash + sum(position_values)
available_cash = total_cash - posted_margin_cash - restricted_short_proceeds
net account P&L = liability P&L + hedge P&L - trading costs - net financing costs
```

ORIGINATE starts with risk capital and no positions; inception books released premium and liability once. The P&L baseline is that capital. IMPORT loads opening cash, positions, marks, declared equity, restrictions and ordered stock-short proceeds lots. Its baseline is opening marked equity, with no repeated historical premium. Historical premium can be denominator metadata only. Inconsistent units, lots or opening equity are malformed input; valid but financially infeasible books receive recorded inception breaches.

An explicit imported `liability_id` must name the held basket. It can be omitted and inferred from the imported basket position; a book without a basket omits it.

Short sales require a locate, limit and borrow-rate input. Gross new-short proceeds are segregated, receive no interest rebate, and cannot finance other trades or margin. FIFO covers release matching proceeds, including partial covers and zero crossings. Available cash and posted margin earn their declared rates; previous holdings, marks and rates accrue over each ACT/365 interval. New rates apply only going forward. Collateral transfers do not create equity.

Interest is credited at each event boundary and enters the next interval's cash balance. This toy convention makes financing slightly sensitive to extra order timers, even without fills: 1,000 cash at annual rate 0.365 grows to 1,002 over one two-day interval, or 1,002.001 with an intermediate one-day boundary. It does not model a separate broker interest-crediting schedule.

The mandatory phase nets same-event interest/charges and contractual settlements atomically, with separate posting IDs. All required inputs are validated before committing it. It books an unfunded obligation in full and stops with the resulting deficit. This is a demonstration netting convention, not an assertion about broker settlement. There is no discretionary overdraft facility or later capital injection.

Accounting reconciliation uses absolute tolerance `1e-8` currency and relative tolerance `1e-10`. Feasibility uses an absolute `1e-8` currency tolerance; quantities within `1e-12` of zero are treated as floating-point remnants. Orders and report values are otherwise not rounded before calculation. These are numerical checks, not exact certificates.

## Inputs, synthetic assumptions and limits

The saved case contains instrument definitions, opening mode/state, trading/funding rules, provenance and the JSONL filename. Each record supplies availability time/sequence and any newly released observations, valuation packet, official fixings, executable quotes or rule changes. Decision flags select the scheduled baseline's review times; they do not limit engine risk checks or other policies. Optional rules specify start time, execution delay, order lifetime and position limits. See the fixtures and `market.py` for concrete fields. JSON nonfinite numbers, conflicting duplicate IDs, invalid units and unsupported instrument kinds are rejected. Official fixing values for the same stock and fixing time must agree exactly across contracts; conflicts are malformed input.

The clock privately reads and validates one following source record to merge due times and preserve all same-time horizon records. A malformed financial payload in that record can therefore raise `InputError` before an earlier timer or a completed horizon returns, even if its availability is beyond the horizon. It is an input-validation failure, not `STOP_DATA_GAP`; unreleased contents never enter strategy, pricing or account updates.

A historical adapter could supply those same point-in-time records. No production market-data connector or historical dataset is included. A decision needs a later contemporaneous executable quote, even with zero configured delay; an old quote arriving later is not a new execution opportunity. Full fills consume the supplied size once, and missing quotes leave orders pending until their deadline. Cancellation is immediate. The engine checks margin at every available market record and clock boundary; the clock cannot establish market moves or breaches between observations.

The default synthetic fixture declares:

- Three stocks initially 100, 80 and 120; fixed coefficients one third each; one short European arithmetic-basket put, strike 100 and multiplier 100.
- Twenty toy business days with 09:00/16:00 UTC observations, weekdays only and no holiday calendar, followed by one normal stock-exit open.
- Zero replay drift; volatilities 20%, 22.5% and 25%; pairwise correlation 0.35. Fixed risk-neutral pricing rate 1%; zero dividends. These are arbitrary test inputs, not estimates or Polykern terms.
- Capital 10,000; buffer 50; half-spread 0.02; fee 0.05 per order; fractional shares; borrow capacity 1,000 shares per stock. Toy cash interest 1%, collateral interest 0.5%, borrow cost 2%; no short-proceeds rebate.
- Replay PCG64 seed 20260912; pricing seed 731; 8,192 antithetic Monte Carlo samples; central spot bumps of `1e-4 * spot`. Pricing derives an event seed from SHA256 of its own seed, UTC event time and model version. Sample count and strategy order cannot change the market path.
- Named stresses: unchanged, spots down 20%, spots up 20%, volatility up 25%. The **Unnetted Liability plus Stress Cash Proxy** is labeled **uncalibrated research margin proxy** in configuration/reports. It reserves gross short-option marked liabilities plus the largest portfolio stress loss, then requires both net equity and cash excluding short proceeds to cover margin plus buffer. Long options can affect stress loss but do not offset the gross liability term.

Basket pricing remains arithmetic, using joint risk-neutral terminal samples. Constituent options use Black–Scholes. Official expiry payoff replaces model valuation exactly. Pricing accepts positive-semidefinite correlations; negative eigenvalues no smaller than −1e-12 are treated as roundoff and that treatment is recorded. Larger violations are rejected. The toy market generator requires a positive-definite correlation matrix and does not repair it.

`expected_stress_scenarios` declares the complete finite margin stress set. Missing declared scenarios produce a data-gap stop, including when an entire scenario disappears. For older cases without this field, the engine freezes the names supplied at inception; that compatibility mode cannot detect a scenario already missing from the first packet. Declare the set explicitly for new cases.

All options are European and cash-settled in this demonstration; stock trade cash settles immediately. There are no dividends, corporate actions, currencies beyond one, settlement lags, exercise/assignment mechanics, order-book matching, market impact, credit lines or broker connectivity. These conventions must not be applied silently to listed equity options.

The synthetic replay and pricing model share fixed dynamics. That can understate model risk; Monte Carlo noise can also affect hedge turnover. Neither the deterministic cases nor one synthetic path establish real-world profitability, tail protection, optimality or broker compliance. The outstanding Polykern payoff, data, collateral/margin, funding/borrow and evaluation requirements remain the five questions in the specification. No separate project/paper bundle was supplied during this work.

## Reports

Each run writes a small policy summary and CSV traces for account checkpoints, order outcomes, postings, settlements and valuation provenance. The case retains full valuation inputs; reports retain their source hashes, parameters and event seeds without repeating the full pricing configuration in every CSV row. Costs and P&L come from actual posting records and marked instrument values, with reconciliation at each financial checkpoint.

The summary's `accepted_hedge_fill_count` counts executed hedge trades, including exits, and excludes liability origination and option settlement.

`comparison.json` preserves every status and identifies the latest common valid **pretrade** checkpoint. A completed run's cash-only terminal P&L is distinct from another run's marked P&L at its stopping time. Stopped or incomplete runs do not receive invented terminal P&L or premium retention. On data gaps, current unknown values remain null alongside a separate last-valid checkpoint.

For completed runs with known positive original premium, premium-retention ratio means net economic P&L divided by that premium. It can be negative or above one; it is not unspent premium cash. Imported books use their opening observation horizon for the numerator. No distributional confidence, RMSE, VaR or CVaR is reported from one path.

With the default saved synthetic fixture, both baselines complete. Unhedged net P&L is approximately −55.381272; delta net P&L is approximately −144.600523, with 63 stock fills. These are reproducible example outcomes, not a ranking supported by a distributional backtest.

## Synthetic scenario hedge: Stage 1

The optional [Cont–Vuletić optimization prototype](docs/cont_vuletic_stage1.md) implements Eq. (20)'s conditional variance objective and weighted L1 penalty on changes of holdings. It uses the existing synthetic GBM and in-process pricer, and returns targets through the same strategy interface. Accounting, margin, execution and the mechanical baseline are unchanged.

```powershell
python -m basket_hedging compare-scenarios --case fixtures/synthetic.json --output reports/cont_vuletic_stage1 --scenarios 256 --scenario-seed 9173 --pricing-samples 1024 --alpha 0.1 --cost-proxy 0.02
```

This writes ordinary engine reports for four policies, `research_summary.json` with defined gross hedge error and optimizer diagnostics, and `optimization_examples.json` with hand-checkable perfect and redundant-instrument cases. The cost proxy is per share/contract; it influences the objective but is never charged to cash. Alpha is a declared demonstration input, not tuned for profitability. The candidate set is the case's permitted hedge universe; no instruments are created automatically.

Scenario timing comes from the current event's declared schedule. The horizon is the next scheduled rebalance, capped by the first relevant option expiry, and must be strictly after earliest execution eligibility. In the synthetic fixture a weekday decision at 16:00 can execute the next morning at 09:00 and models the next close (24 hours); a Friday decision models Monday's close (72 hours), after Monday's 09:00 eligibility. GBM uses the full elapsed calendar time, including weekends. Diagnostics record decision time, earliest execution, next rebalance and scenario horizon. This replaces the old `--step-hours` setting.

Explicit replay records for this policy must supply `next_rebalance_time` with each pre-expiry decision. The engine exposes it and the effective execution eligibility through read-only `view.rules`; the strategy never scans future records. Missing timing or a window ending at/before eligibility produces a recorded skipped decision, with no fixed-duration fallback. Older synthetic fixtures should be regenerated with the `generate` command before running this comparison; baseline policies still accept records without the new field.

This is a decision-to-next-review risk approximation. Existing positions remain held until actual fills, and quotes or account constraints may delay or reject orders. The optimizer does not model separate pre-fill and post-fill P&L, so aligned horizon endpoints do not establish optimality under delayed execution.

For instrument-specific cost weights or a subset of permitted stock/call/put candidates, construct `ScenarioHedge(source, hedge_ids, cost_weights, alpha=...)` from `basket_hedging.scenario_hedge` and pass it to the existing engine. Its source implements `generate(view, fixed_positions, hedge_ids) -> ScenarioChanges`. The scenario/pricing parameters must already be available at the current time. Each run uses a fresh policy instance; optional explicit `gross_scale` supplies a known positive opening marked gross target value.

Stage 1 itself uses no trained generator. The detailed note explains signs, solver tolerances and scope.

## Learned synthetic scenarios: Stage 2

[Stage 2](docs/cont_vuletic_stage2.md) adds a small conditional Gaussian neural generator behind the same scenario-provider interface. It learns joint constituent log returns from independent synthetic training paths. The existing optimizer selects holdings; the unchanged engine executes and funds them. This is not VolGAN, an implied-volatility-surface model, or a historical backtest.

```powershell
python -m pip install -e ".[test,learn]"
python -B -m pytest -p no:cacheprovider -q
python -B -u -m basket_hedging stage2 --output reports/stage2 --test-paths 200 --workers 6
```

The command trains on 1,024 paths, selects an epoch using 256 separate validation paths, checks conditional distributions, then compares four policies on 200 fresh paths: unhedged, delta, true-simulator optimizer and learned-generator optimizer. Both optimizers retain alpha 0.1 and identical costs, candidates, scenario counts and execution rules. Test results do not select the model or financial parameters.

Training requires the optional PyTorch dependency; frozen JSON models use NumPy-only inference. The reports record the architecture, split seeds, model/source hashes, conditional distribution errors, all path outcomes and aggregate hedge metrics. They test whether learning preserves useful variance reduction inside this synthetic model family. They do not establish real-market performance or reliable real-world tail protection.

The [bundled validation report](reports/stage2/validation_report.md) records the completed 200-path comparison and the exact checks run. All 800 policy runs completed; the learned provider retained approximately all of the true-simulator optimizer's variance reduction, with visible conditional-distribution errors documented separately.
