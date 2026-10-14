# Historical basket hedge benchmark

Real historical constituent prices and option surfaces; model-valued non-traded basket option.

Financial comparisons use **62 of 71 episodes** completed by all five policies. Failures remain in the table and raw episode records. Completion rates use all scheduled episodes. Risk statistics are conditional on successful completion; they do not measure losses in failed runs.

| Policy | Error SD | Error RMSE | P&L SD | CVaR95 | Mean turnover | Mean trading costs | Mean fills | Margin breaches | Optimizer failures | Completion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unhedged | 1,174.41 | 1,184.64 | 1,174.41 | 2,995.45 | 0.00 | 0.00 | 0.00 | 0 | 0 | 92.96% |
| Mechanical delta | 395.04 | 392.10 | 400.87 | 1,316.63 | 61,722.18 | 30.86 | 59.95 | 0 | 0 | 92.96% |
| Parametric | 428.93 | 427.92 | 436.66 | 1,345.66 | 56,861.60 | 28.43 | 45.66 | 0 | 0 | 92.96% |
| Gaussian | 418.25 | 417.20 | 424.99 | 1,298.97 | 45,643.18 | 22.82 | 35.40 | 0 | 0 | 92.96% |
| VolGAN | 407.87 | 410.48 | 412.93 | 1,017.33 | 67,611.52 | 33.81 | 22.98 | 0 | 4 | 87.32% |

Financial columns are USD; fills are counts. Error is negative liability-plus-hedge P&L, including hedge dividends, before trading and funding costs. CVaR95 is the mean of the largest ceil(5% × matched episode count) net P&L losses. These are empirical results, not confidence bounds.

## Paired differences

Every value below is policy minus reference on the identical completed episode set. Negative error SD/RMSE or cost differences mean less risk or cost; positive P&L differences mean higher net P&L. The CSV/JSON also includes signed error, absolute/squared error, P&L SD, CVaR95, turnover and fill differences.

| Policy | Reference | Error SD difference | Error RMSE difference | Mean P&L difference | Mean cost difference |
|---|---|---:|---:|---:|---:|
| Mechanical delta | Unhedged | -779.36 | -792.54 | -241.40 | 30.86 |
| Parametric | Unhedged | -745.48 | -756.73 | -206.96 | 28.43 |
| Gaussian | Unhedged | -756.15 | -767.44 | -203.32 | 22.82 |
| VolGAN | Unhedged | -766.53 | -774.16 | -190.27 | 33.81 |
| Unhedged | Mechanical delta | 779.36 | 792.54 | 241.40 | -30.86 |
| Parametric | Mechanical delta | 33.88 | 35.82 | 34.44 | -2.43 |
| Gaussian | Mechanical delta | 23.21 | 25.10 | 38.08 | -8.04 |
| VolGAN | Mechanical delta | 12.83 | 18.38 | 51.13 | 2.94 |
| Unhedged | Gaussian | 756.15 | 767.44 | 203.32 | -22.82 |
| Mechanical delta | Gaussian | -23.21 | -25.10 | -38.08 | 8.04 |
| Parametric | Gaussian | 10.68 | 10.71 | -3.64 | 5.61 |
| VolGAN | Gaussian | -10.38 | -6.72 | 13.05 | 10.98 |

## Failures

- Unhedged: COMPLETED=66, STOP_DATA_GAP=5
- Mechanical delta: COMPLETED=66, STOP_DATA_GAP=5
- Parametric: COMPLETED=66, STOP_DATA_GAP=5
- Gaussian: COMPLETED=66, STOP_DATA_GAP=5
- VolGAN: COMPLETED=62, FAILED_OPTIMIZER=4, STOP_DATA_GAP=5

`FAILED_OPTIMIZER` records a numerical convergence failure in the benchmark driver, separately from an engine breach or data gap. Its account, pending orders and last valid checkpoint are retained in `failed_runs.json`; its terminal P&L and hedge error remain unavailable. The common comparison set excludes that episode for every policy. The optimizer settings were not relaxed to turn these failures into successful results.

## VolGAN calibration and interpretation

The existing nominal 90% return-band coverage is **SPY 49.12%, QQQ 60.79%, IWM 65.86%**. The weak calibration remains unchanged for this financial benchmark. The model is a multi-asset VolGAN adaptation, not a reproduction of the paper's original SPX experiment.

Paired risk and cost differences show the economic results of the existing model. They cannot by themselves attribute performance to calibration: VolGAN also generates IV changes, whereas Gaussian and parametric baselines carry current IV forward. Delayed stock execution and model-valued basket prices remain shared approximations. No model, alpha or trading parameter was tuned from these results.

## Figures

The four figures are in `figures/` as PNG and SVG. The representative episode is the first test episode, selected before outcomes. Distribution, time-series and risk/cost figures use the same matched completed episode set.
