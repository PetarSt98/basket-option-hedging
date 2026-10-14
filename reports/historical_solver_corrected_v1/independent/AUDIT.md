# Independent audit of solver-reliability v1

This audit used the frozen original checkpoints, prepared dataset and settings. It did not retrain, tune alpha, alter engine inputs or modify either benchmark. Probes wrote to a new temporary directory; these compact scripts and evidence are copied here for reproducibility. The historical benchmark was run by the parent task; the calculations below independently inspect its actual outputs and also rerun small cases.

## Results

- 355 unique episode-policy outcomes, representing 71 scheduled episodes and five policies.
- Every policy: 66 COMPLETED and 5 shared STOP_DATA_GAP. No optimizer or margin failures in this corrected run.
- All 351 rows that were not original solver failures remain exactly equal to the frozen results after excluding runtime. The four recovered VolGAN outcomes are separately identified; no terminal value is assigned retroactively to the original failed rows.
- Main comparison uses the same 66 completed episodes for all five policies. CVaR95 is the mean of the worst 4 losses, not a fitted tail estimate.
- Common sufficient-history sensitivity uses 62 episodes. This is a different set from the original frozen common 62: the sensitivity excludes four completed episodes with unavailable conditioning history; the original common set excluded the four solver failures.
- Each optimizer has 42 skipped scheduled decisions over all 71 episodes: 33 across four completed episodes and 9 in one stopped episode. Each policy has 56 scheduled reviews not reached due to stopping. Open callbacks are not counted as scheduled review failures.
- 3,981 successful numerical regression calls; 4 used polishing. All original tolerances remain 1e-9 and iteration caps 20,000; every recorded final optimality residual satisfies the unchanged tolerance.

## Independent calculations

334 headline metric/paired-table checks passed using Python math/statistics rather than production reporting helpers. Another 48,031 checks verify 11,139 raw accepted fills, turnover and one-time spread/fees, reported P&L component reconciliation, calendar timing, available raw opens, declared 5bp execution convention, quote capacity, and 355 decision records against independently reconstructed histories and schedules.

The stopped account-components export originally risked labelling stale-mark/current-cash P&L with a last-valid timestamp. This reporting ambiguity was raised during review and corrected without rerunning or changing financial execution: valuation-dependent components are null for stopped rows, while posting-derived costs remain available. `report_normalization.json` records this adjustment. Main terminal metrics were already unavailable for stopped runs.

2,079 additional checks passed for all population metric rows, all four figure input datasets and 12 selected moving-block bootstrap intervals. Bootstrap checking independently used the same PCG64 seed 20261004, 5,000 replications and block length 3 on the full 71-slot calendar; missing slots were filtered only after common index draws. Scalar sample SD, means, four-or-resampled-tail CVaR and explicit empirical percentile interpolation reproduced the saved intervals.

On the main 66 population, VolGAN minus delta 95% exploratory block intervals are:

| Metric difference | Lower | Upper |
|---|---:|---:|
| Mean absolute hedge error | -27.876454 | 94.438192 |
| Hedge-error SD | -54.855877 | 99.510662 |
| CVaR95 | -503.256512 | 217.115720 |

These intervals include zero. They do not establish equivalence or a VolGAN improvement. The original frozen models' weak calibration remains unchanged.

## Fresh end-to-end checks

The first historical episode reran all five policies, reproducing original financial results and 25 CSV files exactly. Independent reconstruction passed 3,360 event-level accounting checks and 181 eligible fill checks. Future mutation after January 9 left prior delta/VolGAN financial records exactly unchanged. Generated IV sensitivity changes option repricing and optimizer targets while preserving stock-scenario changes. An independent scalar scenario price matched within8.88e-14.

March 2020 delta and VolGAN replays also reproduce frozen results. IWM's official March 25 ex-date cash 0.41988/share is posted once on pre-opening-fill holdings; no March 31 payable-date recredit occurs.

Legacy deterministic, synthetic, gap, missing-final-exit and rejected-final-exit cases retain all financial outputs: 50 CSV files are byte-identical, and 10 summaries preserve all original fields. Some old summaries gain only previously absent execution/rebalance timing metadata.

## Reproduce

From the repository root, with the prepared cache available:

```
python -B reports/historical_solver_corrected_v1/independent/metrics_probe.py
python -B reports/historical_solver_corrected_v1/independent/raw_probe.py
python -B reports/historical_solver_corrected_v1/independent/population_bootstrap_figures.py
python -B reports/historical_solver_corrected_v1/independent/legacy_probe.py
python -B reports/historical_solver_corrected_v1/independent/fresh/probe.py
python -B reports/historical_solver_corrected_v1/independent/fresh/independent_financial.py
python -B reports/historical_solver_corrected_v1/independent/fresh/ex_dividend.py
```

The scripts record the local repository path used by this audit. They write only beside themselves. Saved JSON contains exact populations, recovered outcomes, checks, source hashes and interval results. The 58 original historical checkpoint/report hashes captured before the audit remain unchanged. Main raw-data/checkpoint/config hashes also match the frozen benchmark manifest. No calibration conclusion or production-readiness claim follows from these checks.
