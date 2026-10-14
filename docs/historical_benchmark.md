# Historical basket and joint VolGAN adaptation

**Real historical constituent prices and option surfaces; model-valued non-traded basket option.**

This extension reads the six specified SPY, QQQ and IWM release files, builds
daily implied-volatility surfaces, and trains one joint conditional GAN. The
existing optimizer converts generated market states into constituent share
targets. The existing engine performs accounting, delayed fills, margin checks
and settlement. The synthetic examples and Gaussian checkpoint remain separate.

## Verified corporate-action supplement

All six published hashes match. The shared history contains 3,693 dates from
23 March 2011 through 12 December 2025. IWM has no dividend or split records in
these files: the columns are entirely null. The user authorized a separate
[corporate-action supplement](../data/iwm_corporate_actions/README.md).
The complete official iShares history supplies 59 distributions in the period,
including ex-date, payable date and full cash distribution per share. All 59
ex-dates match Yahoo's optional cross-check. Yahoo's three-decimal amounts differ
by at most USD 0.000487; the six-decimal official values are retained.

[Two public split histories](iwm_split_verification.md) report only the June 2005
2-for-1 event, before this period. Split factor 1.0 is supported by verified
coverage rather than inferred from the original null fields. The normalized
distribution SHA-256 is
`7e07e3a11542a756c4c14d393e437b745e8f1313f65750b328887fe1b97a9e9b`.

The supplement rebuilds a separate `prepared-etf-v1-actions` dataset. Its manifest
records both the original prepared hash and supplement provenance. Before using
the frozen checkpoints, the runner verifies array-for-array that prices, IVs,
calendars, validity masks and SPY/QQQ actions are unchanged. Only IWM dividends
and split factors differ. The original six Parquets and training dataset remain
unchanged, and the Gaussian/VolGAN models are not retrained.

Both historical models have been trained. VolGAN completed the full 25-epoch
calibration and 10,000-epoch main schedule on 1,592 joint examples, seed 1729.
Its frozen 2018–2019 validation diagnostics are unfavorable:

| ETF | Coverage of nominal 90% return band | Mean generated return SD | Observed return SD |
|---|---:|---:|---:|
| SPY | 49.12% | 0.00363 | 0.00932 |
| QQQ | 60.79% | 0.00647 | 0.01220 |
| IWM | 65.86% | 0.00604 | 0.01047 |

These are diagnostics on 454 chronological validation transitions, with 256
generated states each. The coverage indicates under-dispersed return scenarios.
Mean conditional SD and unconditional observed SD are different quantities;
their numerical comparison alone is not an identity test. Joint generation
does not guarantee accurate correlations or tails. No parameters were retuned
after seeing these diagnostics, and they establish no hedge-performance result.

The Gaussian baseline ran its existing 40-epoch training, selected epoch 26
using validation likelihood, and recorded validation NLL 3.0039504264. This is
not directly comparable with a GAN's adversarial loss. Saved VolGAN wall time
includes an observed host pause of roughly 7.3 hours; it is not compute time.

The [data record](etf_data.md) lists schemas, cleaning, hashes and source terms.
The [VolGAN source record](volgan_sources.md) identifies the equations, network
architecture, training schedule and deviations from the public example.

## Fixed experiment

The split was declared before any test-period hedge outcomes:

| Segment | Dates |
|---|---|
| Training | Common start through 2017-12-29 |
| Validation | 2018-01-02 through 2019-12-31 |
| Test | 2020-01-02 through 2025-12-12 |

Twenty-one return observations are needed for conditioning. Windows crossing
missing sessions or missing surfaces are excluded from training, with the rule
shared by both learned models. This leaves 1,592 training and 454 validation
transitions. Asset rows are kept together. VolGAN uses its fixed final epoch;
Gaussian retains its existing validation-likelihood checkpoint selection.
Neither model selects parameters from test hedge results.

The pricing correlation and parametric calendar-day return moments are frozen
from training data. The GAN generates one **trading-session** transition jointly
for all assets, including the next-session surface changes. A Friday sample ends
on Monday's close; it is not multiplied by a Brownian weekend scaling factor.
The existing Gaussian baseline keeps its calendar-day scaling. Gaussian and
parametric scenarios carry the current IV surface forward; their more restricted
information differs from VolGAN's joint return/surface distribution.

The comparison uses unhedged, stock delta, parametric scenarios,
retrained Gaussian scenarios and VolGAN scenarios. Each optimizer uses the
unchanged L1 objective with fixed alpha 0.1, 256 scenarios and inception
half-spread cost weights. No alpha is chosen to improve historical P&L. Existing
synthetic variance and L1 commands remain available.

At each scheduled inception, fixed constituent quantities give USD 100 of each
ETF per basket unit, so the basket starts at 300. Sell one model-valued European
put, strike 300, multiplier 100, with USD 100,000 initial capital. Hold for 20
exchange sessions, then unwind stocks at the next open. Subsequent episodes
begin at that day's close. Contracts are scheduled on the observed underlying
calendar, including dates whose joint option data is missing. Missing chains
cannot shift an earlier contract's maturity or inception premium. A crossing
episode stops visibly when the missing session is reached.
The fixed test calendar defines 71 episodes, from the first inception on
2020-01-02 to the last inception on 2025-11-06, whose unwind is 2025-12-08.

## Pricing, execution and distributions

The basket proxy uses the existing arithmetic-basket Monte Carlo pricer, each
asset's ATM IV at remaining maturity, frozen correlation, 512 antithetic draws
and pricing seed 731. Risk-neutral rates and dividend yields are zero **pricing
assumptions**, not measured market inputs. It is not a model calibrated to every
smile point or a quoted basket price. Generated spot and IV changes both affect
scenario basket repricing. Shared pricing draws are separate from scenario noise.

At an assumed 16:00 New York release, close prices and that day's surfaces become
available. Decisions can first execute at the next session's 09:30 open. Open
marks use the previous close's surface, with its timestamp recorded. Daily bars
do not establish true publication latency; early closes and intraday movements
are not modeled. Prices between observations are unknown.

An order fills at the observed raw opening price plus or minus an **assumed**
5-basis-point half-spread. Size and locate capacity are assumed at 10,000 shares;
fractional holdings are permitted. No observed stock bid/ask or depth is claimed.
No option-chain quote becomes a tradable basket quote. Funding rates are zero,
stock borrow is an assumed 1% per year, the safety buffer is USD 1,000, and the
existing margin component remains an **uncalibrated research margin proxy**.

Known cash distributions are mandatory postings on pretrade stock holdings:
`dividend_cashflow = signed_shares * cash_dividend_per_share`. Longs receive cash;
shorts owe it. An unfunded obligation is booked before the normal breach check.
It enters hedge P&L once, not trading costs or external capital. The research
timing assumption credits/debits at ex-date opening; payment-date receivables
and their financing are not modeled. Unknown distributions block the benchmark.
Any non-unit split also blocks until explicit position and contract adjustments
exist. These limitations remain visible even with verified corporate actions.

For example, four shares marked at 100 have asset value 400. An ex-dividend move
to 98 reduces that value to 392; a USD 2 distribution adds cash 8, so equity is
unchanged before other market moves and costs. With four short shares, both signs
reverse. Buying after that opening event earns no retroactive distribution.

The close-to-close scenario objective still includes overnight changes before
the new target can fill at the open. Aligning the endpoint avoids an expired
scenario horizon but does not make the target executable throughout that interval.
No optimizer or engine modification hides this approximation.

## Results and four figures

`episodes.csv` retains every policy and scheduled episode, including breaches,
missing-data stops, rejected orders and skipped decisions. Terminal statistics
use the same episodes completed by every policy, with the retained count shown.
All-run trading totals and failure counts remain available. Stopped P&L is never
used as maturity P&L.

The fixed solver can also fail to meet its existing convergence tolerance. The
benchmark driver records this separately as `FAILED_OPTIMIZER` and continues the
other runs. It does not change the engine's financial statuses or relax solver
settings. `failed_runs.json` preserves the failed account, postings, pending
orders and last valid checkpoint; terminal P&L and hedge error remain unknown.
That episode is excluded from terminal comparisons for every policy. Risk
statistics are therefore conditional on successful completion, with failures
and completion rates shown alongside them.

Gross hedge error is negative liability-plus-hedge P&L, including actual hedge
dividends, before trading and funding costs. Net P&L includes both costs. CVaR95
is the mean of the worst `ceil(0.05 * matched_count)` observed net P&L losses;
the small tail count is reported. Turnover is executed absolute stock notional
in USD. Historical episodes are not independent Monte Carlo paths; no statistical
confidence or profitable-deployment claim follows from these sample metrics.

The renderer saves four PNG/SVG pairs: error distribution, error by inception
date, risk versus turnover/cost, and the first test episode's basket, signed
liability, VolGAN holdings and all-policy P&L. The first episode is selected
before outcomes, even if it fails. Missing-data failures are not edited out.

### Observed fixed-run results

The [financial report](../reports/historical/benchmark/benchmark.md) records all
355 outcomes (71 contracts times five policies). Five episodes stop for missing
data under every policy. VolGAN additionally fails the unchanged 20,000-iteration
solver convergence check in four episodes, beginning 2022-06-02, 2022-09-01,
2022-12-01 and 2023-02-02. There are no observed margin breaches. The other four
policies complete 66/71 contracts; VolGAN completes 62/71. Thus the common
terminal comparison contains 62 episodes, and empirical CVaR95 uses four losses.
Each scenario policy also records 42 skipped decisions across all runs when
consecutive conditioning history is unavailable around data gaps; this includes
33 skips in otherwise completed episodes. The engine retains existing holdings
at these decisions. Those outcomes are included rather than backfilled.

On that common set, hedge-error SD is USD 1,174.41 unhedged, 395.04 delta,
428.93 parametric, 418.25 Gaussian and 407.87 VolGAN. VolGAN reduces SD 65.27%
against unhedged, but has 3.25% higher SD and 9.54% higher turnover than delta.
Against Gaussian it has 2.48% lower SD but 48.13% higher turnover and costs.
Its mean 22.98 fills is fewer than Gaussian's 35.40, yet those trades move more
notional: fewer fills does not imply lower turnover. VolGAN's CVaR95 is lower
on the retained sample (USD 1,017.33), but this small tail excludes its four
failed contracts. All four solver-failure episodes were losses for the other
four policies. Excluding them makes a broad downside-protection claim especially
unsafe; the retained CVaR is not evidence of reliable tail protection.

The weak return-band coverage remains SPY 49.12%, QQQ 60.79%, IWM 65.86% at a
nominal 90%. These results show useful hedging relative to no hedge, but do not
establish an advantage over mechanical delta. Poor calibration remains
economically relevant: generated scenarios guide real share allocations, and
this run has greater turnover and numerical failures. The comparison cannot
prove calibration caused those outcomes; surface generation, return dependence,
L1 penalties and delayed fills also affect them. Successful-run statistics alone
cannot measure the economic loss from an unavailable hedge decision.

No correction is applied to this benchmark. Before another model comparison,
the smallest justified follow-up is to inspect the generated return covariance
and near-collinearity at the four failed decisions, then assess a simple
validation-only return scale/covariance calibration against the existing 90%
coverage diagnostic. Keep any proposed correction separate from these frozen
results; do not fit it to the test hedge P&L or silently relax solver tolerance.
No network redesign or new hedge strategy is supported by this evidence.

## Reproduction commands (PowerShell)

```powershell
python -m pip install -e ".[test,learn,historical,demo]"
python -B -m pytest -p no:cacheprovider -q

$rawData = "$env:USERPROFILE\.cache\basket-hedging\data-v1"
$preparedData = "$env:USERPROFILE\.cache\basket-hedging\prepared-etf-v1"
$accountingData = "$env:USERPROFILE\.cache\basket-hedging\prepared-etf-v1-actions"
python -B -m basket_hedging historical fetch --raw $rawData
python -B -m basket_hedging historical check --raw $rawData --output reports/historical/data_check.json
# Reuse the verified base preparation if it is already present.
python -B -m basket_hedging historical prepare --raw $rawData --output $preparedData
# Use the bundled supplement and unchanged saved checkpoints; no retraining.
python -B -m basket_hedging historical supplement --prepared $preparedData --actions data/iwm_corporate_actions --output $accountingData
python -B -m basket_hedging historical run --prepared $accountingData --models reports/historical/models --output reports/historical/benchmark
python -B -m basket_hedging historical figures --report reports/historical/benchmark
git diff --check
```

The `run` command also writes paired differences against unhedged, delta and
Gaussian on the identical completed episode set. Corporate-action amounts enter
only their ex-date accounting event, not the generator's features or future view.
The frozen supplement permits offline reproduction. A separate
`python -B -m basket_hedging.corporate_actions --output data/iwm_corporate_actions`
retrieves a new snapshot; this is not needed to reproduce the saved benchmark.

The prior training ran locally on CPU. VolGAN used 25 gradient-calibration epochs followed
by a restart and 10,000 training epochs; the full command can take tens of minutes.
No training command is part of this completion run. No old synthetic reports
or checkpoints are overwritten by these commands. Python 3.11+ is supported;
the historical Parquet reader adds the optional PyArrow 19–23 dependency.
