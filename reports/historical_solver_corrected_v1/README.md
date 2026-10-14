# Historical benchmark v1: solver reliability correction

**Completed: all 71 scheduled episodes were attempted for all five policies.** Each policy completed 66 episodes and stopped on the same five data gaps. There were no remaining optimizer failures, margin breaches, rejected orders or incomplete exits. This is a **revised retrospective benchmark**, not an untouched test or a generator improvement.

The original frozen VolGAN checkpoint is used. The unsuccessful calibration experiment ended with **NO CONFIRMED IMPROVEMENT**; its seed 1729/epoch 5,000 checkpoint is not used here. No model was trained or selected in this task.

## Exact change and attribution

The prior solver correction adds bounded active-sign polishing after the existing 20,000-iteration coordinate-descent budget is exhausted. Successful coordinate-descent solves are unchanged. The objective, free intercept, weighted L1 penalty, alpha **0.1**, gross scale, cost weights and original scaled KKT tolerance **1e-9** remain unchanged. No ridge, clipping, tolerance relaxation or delta fallback is applied.

The same data-v1 SPY/QQQ/IWM inputs, IWM supplement, 71 contracts, fixed quantities, premiums, 256 scenarios, 512 pricing draws, seeds, stock-only execution, funding and research margin proxy are retained. Original historical, solver-validation, calibration and synthetic artifacts are preserved. `plan.json` was saved before the run and before uncertainty calculation. It fixes the populations and bootstrap settings; `frozen_hashes.json` records the pre-task files.

Original checkpoint SHA-256 values:

- VolGAN: `04e3a56793f3a3bf9b7c1ff11c1bf162c8ac497f44d92c80480a093d51d1c12b`.
- Gaussian: `199f0db9f33334600b8d117e18f45e5a49f8b4c0e098bb1864358a19b967c18a`.
- Supplemented prepared data: `f9d091fc11c0ca39277e447523f178cce90928b2799ace9b577d4742f4147778`.

The original frozen results remain in `../historical/benchmark`. All **351 previously non-solver-failure episode/policy records** agree exactly, excluding elapsed runtime. On the original common 62 episodes, all financial results are unchanged. This correction supplies four previously unknown VolGAN terminal outcomes through actual engine execution; it does not improve the other 62 paths.

## Population and failure accounting

| Policy | Scheduled | Completed | Data-gap stops | Optimizer failures | Margin breaches | Incomplete exits | Completion |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unhedged | 71 | 66 | 5 | 0 | 0 | 0 | 92.96% |
| Mechanical delta | 71 | 66 | 5 | 0 | 0 | 0 | 92.96% |
| Parametric | 71 | 66 | 5 | 0 | 0 | 0 | 92.96% |
| Gaussian | 71 | 66 | 5 | 0 | 0 | 0 | 92.96% |
| Original frozen VolGAN, corrected solver | 71 | 66 | 5 | 0 | 0 | 0 | 92.96% |

The five excluded inception dates are **2020-06-03, 2021-04-05, 2021-05-04, 2021-11-01 and 2024-10-04**. Missing valuation packets stop these accounts. Their terminal P&L and hedge error are blank, not zero. Their recorded costs, holdings and stopping details remain in the outputs. Completion denominators always use 71, not only the successful subset.

The main operational comparison attempts every episode and reports risk on the **66 common completions**, alongside all failure counts. Each policy's own completed population happens to be this same 66 in the new run; it is nevertheless recorded separately. The original common 62 and the sufficient-history sensitivity 62 are separately named populations with explicit IDs in `reliability_populations.json`. They must not be interchanged.

## Main comparison: 66 common completions

All financial numbers are USD; fills are accepted stock fills. Means below use the stated completed population. Full totals over all 71 attempts, including stopped runs, remain in `summary.json` and `episodes.csv`.

| Policy | Hedge-error SD | Error RMSE | P&L SD | CVaR95 | Mean net P&L | Mean turnover | Mean trading costs | Mean fills |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unhedged | 1,218.29 | 1,214.49 | 1,218.29 | 3,097.61 | 115.11 | 0.00 | 0.00 | 0.00 |
| Delta | 391.64 | 388.68 | 397.76 | 1,316.63 | -44.79 | 62,231.49 | 31.12 | 60.14 |
| Parametric | 427.24 | 424.70 | 435.30 | 1,345.66 | -13.85 | 57,575.47 | 28.79 | 45.50 |
| Gaussian | 415.51 | 413.01 | 422.66 | 1,298.97 | -9.57 | 46,464.01 | 23.23 | 35.24 |
| Original frozen VolGAN | 399.96 | 401.11 | 405.14 | 1,017.33 | 13.00 | 67,617.58 | 33.81 | 22.91 |

Hedge error is **negative liability-plus-hedge P&L**, including hedge dividends but before trading and financing costs. Net P&L includes both. SD is sample SD (`ddof=1`); RMSE is `sqrt(mean(error**2))`, without demeaning. CVaR95 is the mean of the largest `ceil(0.05*n)` net P&L losses: **four observations** for both n=66 and n=62. Turnover is actual absolute executed stock cash notional; spreads/fees are counted once. Origination and settlement are not hedge fills.

All-run fill totals are 0 / 4,101 / 3,085 / 2,399 / 1,554 in policy order. VolGAN makes fewer fills but trades more notional than delta or Gaussian. A smaller fill count is not evidence of lower trading activity or cost.

## Original and corrected results on identical original 62

All values below are **identical before and after** the correction on this population. Full original/corrected tables, including costs and turnover, are saved separately in `population_metrics.csv`.

| Policy | Original SD | Corrected SD | Original RMSE | Corrected RMSE | Original CVaR95 | Corrected CVaR95 |
|---|---:|---:|---:|---:|---:|---:|
| Unhedged | 1,174.41 | 1,174.41 | 1,184.64 | 1,184.64 | 2,995.45 | 2,995.45 |
| Delta | 395.04 | 395.04 | 392.10 | 392.10 | 1,316.63 | 1,316.63 |
| Parametric | 428.93 | 428.93 | 427.92 | 427.92 | 1,345.66 | 1,345.66 |
| Gaussian | 418.25 | 418.25 | 417.20 | 417.20 | 1,298.97 | 1,298.97 |
| Original frozen VolGAN | 407.87 | 407.87 | 410.48 | 410.48 | 1,017.33 | 1,017.33 |

### Four recovered VolGAN episodes

Each used exactly one polish. The full benchmark recorded 3,981 successful scenario solves and four polishes, with no new optimizer failure. All polished solutions satisfy the unchanged full-problem KKT check. These are floating-point residuals, not exact certificates.

| Episode | Old failed residual | Corrected residual | VolGAN hedge error | VolGAN net P&L | Fills |
|---|---:|---:|---:|---:|---:|
| 2022-06-02 | 1.56046e-8 | 1.07060e-15 | -145.39 | 94.26 | 18 |
| 2022-09-01 | 4.72515e-9 | 4.05945e-16 | 317.92 | -382.49 | 28 |
| 2022-12-01 | 1.94458e-8 | 9.18285e-16 | 166.74 | -213.88 | 21 |
| 2023-02-02 | 1.07574e-9 | 4.17389e-16 | 148.23 | -197.88 | 20 |

Their same-episode net P&Ls for all policies are:

| Episode | Unhedged | Delta | Parametric | Gaussian | Original VolGAN with corrected solver |
|---|---:|---:|---:|---:|---:|
| 2022-06-02 | -1,910.92 | -160.95 | -186.59 | -202.55 | 94.26 |
| 2022-09-01 | -2,113.73 | -394.39 | -396.94 | -415.90 | -382.49 |
| 2022-12-01 | -1,466.12 | -593.91 | -688.61 | -589.24 | -213.88 |
| 2023-02-02 | -266.72 | -194.56 | -164.53 | -172.68 | -197.88 |

The old failed VolGAN runs still have unknown terminal outcomes in the frozen benchmark. These new outcomes belong only to this separately executed correction. Comparing old common-62 statistics directly with new common-66 statistics would conflate population changes with changed financial behavior.

## Missing-history sensitivity

The existing requirement is unchanged: **21 consecutive return transitions from 22 closing prices**, plus current valid joint IV surfaces. The plan's shorthand “21-observation conditioning” refers to this existing code convention. No missing observation is backfilled. At a skipped decision, the existing holdings persist subject to the ordinary engine rules.

All policies have 1,491 scheduled close reviews (71 × 21, including terminal unwind). Actual scheduled-review outcomes:

| Policy | Scheduled | Targets made | Skipped | Not reached because run stopped |
|---|---:|---:|---:|---:|
| Unhedged | 1,491 | 1,435 | 0 | 56 |
| Delta | 1,491 | 1,435 | 0 | 56 |
| Parametric | 1,491 | 1,393 | 42 | 56 |
| Gaussian | 1,491 | 1,393 | 42 | 56 |
| Original VolGAN | 1,491 | 1,393 | 42 | 56 |

Open-event callbacks are recorded but are not counted as missed scheduled reviews. `decisions.json` preserves every scheduled date, actual decision, skip reason and data-availability flag. Each scenario policy has these skips:

| Episode | Skips | Eventual status |
|---|---:|---|
| 2020-07-02 | 15 | COMPLETED |
| 2021-05-04 | 9 | STOP_DATA_GAP |
| 2021-06-03 | 10 | COMPLETED |
| 2021-12-01 | 4 | COMPLETED |
| 2024-11-04 | 4 | COMPLETED |

There are **33 skips in four completed episodes**, plus nine in a stopped episode. Data availability alone defines the sensitivity subset: remove those four completed episodes from the common 66. The resulting **62 sufficient-history completions** include the four recovered solver episodes and exclude four different history-affected episodes. This is not the original frozen common-62 population.

| Policy | Sensitivity error SD | Error RMSE | Mean net P&L | Mean turnover | Mean trading costs |
|---|---:|---:|---:|---:|---:|
| Unhedged | 1,243.74 | 1,235.65 | 70.04 | 0.00 | 0.00 |
| Delta | 397.06 | 394.22 | -58.68 | 62,775.60 | 31.39 |
| Parametric | 403.12 | 400.26 | -57.67 | 59,654.72 | 29.83 |
| Gaussian | 396.42 | 393.52 | -49.84 | 48,186.19 | 24.09 |
| Original VolGAN | 378.69 | 376.09 | -27.87 | 70,368.92 | 35.18 |

The VolGAN-versus-delta SD ordering changes in this sensitivity. It does not replace the 66-completion headline. The exclusion rule measures the consequences of available conditioning history; it was not selected from profits or errors.

## Paired exploratory uncertainty

Predeclared settings: **5,000** noncircular moving-block resamples, **three scheduled episodes per block**, NumPy PCG64 seed **20261004**, percentile **2.5%/97.5%** limits. Blocks are sampled on the full chronological 71-slot schedule and concatenated/truncated to 71. Every policy uses the same episode-vector draws; each named population is filtered afterward, preserving missing slots and pairing. No trade or generated scenario is treated as an independent historical outcome. Valid replicate counts and population memberships are saved in `uncertainty.csv` and `reliability_populations.json`.

Selected paired differences below are **VolGAN minus reference** on the main common 66, in USD:

| Reference | Statistic | Estimate | Exploratory 95% interval |
|---|---|---:|---|
| Unhedged | Error SD | -818.33 | [-1,203.36, -457.02] |
| Delta | Error SD | 8.33 | [-54.86, 99.51] |
| Gaussian | Error SD | -15.55 | [-72.97, 56.64] |
| Delta | Error RMSE | 12.43 | [-54.50, 107.58] |
| Delta | Mean absolute error | 30.27 | [-27.88, 94.44] |
| Gaussian | Mean absolute error | 1.40 | [-46.26, 46.18] |
| Delta | CVaR95 | -299.31 | [-503.26, 217.12] |
| Gaussian | CVaR95 | -281.64 | [-598.36, 228.81] |
| Delta | Mean trading costs | 2.69 | [0.26, 5.12] |
| Gaussian | Mean trading costs | 10.58 | [8.33, 13.00] |

On the separate sufficient-history 62, VolGAN minus delta SD is -18.37, interval [-75.86, 59.84]; minus Gaussian SD is -17.74, interval [-76.60, 56.79]. All declared paired statistics/references/populations, including old/new matched 62, are retained rather than selecting only favorable intervals.

These intervals are exploratory, population-specific and unadjusted for multiple comparisons. A fixed block length cannot guarantee adequate dependence modelling for every regime. Zero inclusion does not prove equivalence. A result for absolute error is not a result about equal SD or CVaR. Four observed tail losses cannot establish reliable tail protection, regardless of a bootstrap interval.

## Calibration and modelling limits

The model actually used is the original final-epoch VolGAN. Its saved 2018–2019 nominal 90% return coverage remains **SPY 49.12%, QQQ 60.79%, IWM 65.86%**, over 454 validation targets and 256 scenarios per target. These are developmental diagnostics, not financial-test statistics. The separate bounded checkpoint-selection experiment did not establish consistent improvement across two seeds, so no replacement was accepted and its 2019 confirmation was not evaluated. See `../historical_calibration/README.md`; its unconfirmed intermediate checkpoint is not a financial strategy in this report.

- The nontraded basket liability and premiums are model-valued. Shared pricing improves comparative consistency but does not remove pricing-model bias.
- Execution uses assumed 5bp half-spreads around historical opens, not observed intraday executable bid/ask quotes. Fills occur after decisions; the scenario objective still includes exposure before the new hedge can execute.
- The margin rule is an uncalibrated research proxy. Locates, depth, rates and ex-date cash posting are the same declared research assumptions.
- VolGAN generates joint trading-session return/surface transitions. Parametric and Gaussian use calendar-day scaling and carry IV forward. Differences are not attributable to generator quality alone.
- Generated near-collinearity and marginal undercoverage are separate findings. Near-collinearity slowed coordinate descent in the audited regressions; poor coverage is not a proven cause of every numerical failure or trading pattern. Conditional covariance is not forced to match unconditional realized covariance.
- This is a free-data ETF basket adaptation, not the papers' OptionMetrics/SPX experiment. The existing fixed alpha differs from the paper's alpha-selection procedure; AIC-alpha work is outside this correction.
- The financial 2020–2025 period was already inspected. This revision is retrospective, with small dependent samples and conditional-on-completion risk estimates. Unknown outcomes for the five failed episodes remain unknown.

## Figures and saved evidence

The four existing figure types are regenerated from saved tables, labelled with the original frozen model and solver correction:

1. [Hedge-error distribution](figures/01_error_distribution.png).
2. [Hedge error through time](figures/02_error_through_time.png): missing episodes have explicit gaps and markers, not connecting lines.
3. [Risk versus turnover and costs](figures/03_risk_cost.png).
4. [Representative episode](figures/04_representative_episode.png): the unchanged predetermined first episode, **2020-01-02**, with basket value, signed short-put value, three holdings and all-policy net P&L.

Each also has an SVG. Captions identify the 66/71 common population, five exclusions and 25/355 stopped policy runs. `figures/figure_data.json` retains exact plotted data, memberships and input hashes. All four PNGs were visually inspected.

`episodes.csv` contains all 355 outcomes; `fills.csv` and `order_outcomes.json` retain actual trading records; `accounting_components.csv` supplies independently reconcilable completed P&L legs and posting-derived costs; `failed_runs.json` preserves stopping details; `solver_diagnostics.json` records all convergence checks. Stopped-book valuation-dependent components are blank and the last valid valuation time is separate from known posted costs. A reporting-only normalization applied this clarification to the completed run's export; it did not change any financial outcome. `report_normalization.json` and `runner_source_at_start.json` document it. The final runner emits this schema directly.

## Files changed and checks

Only reporting additions were made to:

- `src/basket_hedging/historical.py`: scheduled/actual decisions, fill/component/order and compact solver exports; no financial rule change.
- `src/basket_hedging/historical_report.py`: explicit population tables and paired block-bootstrap summaries using existing reporting functions.
- `src/basket_hedging/historical_plots.py`: version labels, visible missing dates, exact figure data and readable date labels.
- `tests/test_historical.py`, `tests/test_historical_report.py`, and new `tests/test_historical_plots.py`: independent expected numbers, missing-history/stopped-value handling and plot/uncertainty checks.
- New artifacts under `reports/historical_solver_corrected_v1/` only. All other pre-existing work is preserved.

Actual validation:

- Focused historical/solver tests: **25 passed in 13.31 s**.
- Focused population/uncertainty tests: **14 passed in 1.16 s**.
- Full suite after stopped-value reporting correction: **624 passed in 42.11 s**.
- Final plot-format check: **2 passed in 8.19 s**.
- All 355 historical outcomes run through the real engine.
- Independent checks passed: 334 headline/paired assertions, 48,031 raw checks covering 11,139 fills, and 2,079 population/figure/interval checks. Twelve selected bootstrap intervals were independently reproduced with scalar formulas. Fresh January/March accounting, dividend and future-mutation probes also passed; see [independent/AUDIT.md](independent/AUDIT.md).
- Preservation checked 360 pre-task files: 357 unchanged and exactly the three declared reporting-source changes, with no unexpected difference. Original datasets, checkpoints, reports and the existing solver fix are unchanged; see `preservation.json`.
- Ten legacy policy runs reproduced completed, breach and missing/rejected-exit fixtures. Their **50 financial CSV files are byte-identical** to the saved legacy reports. No training was run.

No lint/type checker is configured. `git diff --check` completes with exit 0; existing LF/CRLF warnings are not whitespace errors.

## Reproduce into a separate directory

Use the existing installed environment and frozen data/checkpoints. From the repository root:

```powershell
$prepared = "$env:USERPROFILE/.cache/basket-hedging/prepared-etf-v1-actions"
$out = "reports/historical_solver_corrected_v1_reproduction"
if (Test-Path -LiteralPath $out) { throw "Choose a new output directory." }
New-Item -ItemType Directory -Path $out | Out-Null
Copy-Item reports/historical_solver_corrected_v1/plan.json $out
Copy-Item reports/historical_solver_corrected_v1/benchmark_version.json $out

python -B -m basket_hedging historical run --prepared $prepared --models reports/historical/models --output $out
python -B -c "from pathlib import Path; import sys; from basket_hedging.historical_report import reliability_report; reliability_report(Path(sys.argv[1]), Path('reports/historical/benchmark'), seed=20261004, repetitions=5000, block_length=3)" $out
python -B -m basket_hedging historical figures --report $out
python -B -m pytest -q
git diff --check
```

The existing install command, if required, is `python -m pip install -e ".[test,learn,historical,demo]"`. This run used Python 3.13.3, NumPy 2.3.1 and PyTorch 2.12.1 CPU. Hashes and complete settings are saved in the plan, summary and preservation records. Neither downloading/preparing data nor training is needed to reproduce the financial run.

Legacy checks, also into the new directory:

```powershell
$cases = @{ deterministic='deterministic_complete.json'; synthetic='synthetic.json'; gap='deterministic_gap.json'; missing_exit='deterministic_missing_exit.json'; rejected_exit='deterministic_rejected_exit.json' }
foreach ($name in $cases.Keys) {
    python -B -m basket_hedging run --case "fixtures/$($cases[$name])" --policy both --output "$out/legacy/$name"
}
```

## Plain-English conclusion

The solver correction removes four numerical stoppages without changing the model or the outcomes that already completed. We can now compare all five hedges on the same 66 completed contracts. VolGAN reduces risk substantially versus no hedge, but its full-population error SD is slightly above delta and its turnover is higher. The apparent ordering changes when history-affected episodes are excluded, and paired uncertainty does not establish a clear advantage over delta or Gaussian. Weak calibration and five unknown data-gap outcomes remain visible.
