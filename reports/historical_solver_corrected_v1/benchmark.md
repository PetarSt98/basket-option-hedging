# Historical basket hedge benchmark

Real historical constituent prices and option surfaces; model-valued non-traded basket option.

Financial comparisons use **66 of 71 episodes** completed by all five policies. Failures remain in the table and raw episode records. Completion rates use all scheduled episodes. Risk statistics are conditional on successful completion; they do not measure losses in failed runs.

| Policy | Error SD | Error RMSE | P&L SD | CVaR95 | Mean turnover | Mean trading costs | Mean fills | Margin breaches | Optimizer failures | Completion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unhedged | 1,218.29 | 1,214.49 | 1,218.29 | 3,097.61 | 0.00 | 0.00 | 0.00 | 0 | 0 | 92.96% |
| Mechanical delta | 391.64 | 388.68 | 397.76 | 1,316.63 | 62,231.49 | 31.12 | 60.14 | 0 | 0 | 92.96% |
| Parametric | 427.24 | 424.70 | 435.30 | 1,345.66 | 57,575.47 | 28.79 | 45.50 | 0 | 0 | 92.96% |
| Gaussian | 415.51 | 413.01 | 422.66 | 1,298.97 | 46,464.01 | 23.23 | 35.24 | 0 | 0 | 92.96% |
| VolGAN | 399.96 | 401.11 | 405.14 | 1,017.33 | 67,617.58 | 33.81 | 22.91 | 0 | 0 | 92.96% |

Financial columns are USD; fills are counts. Error is negative liability-plus-hedge P&L, including hedge dividends, before trading and funding costs. CVaR95 is the mean of the largest ceil(5% × matched episode count) net P&L losses. These are empirical results, not confidence bounds.

## Paired differences

Every value below is policy minus reference on the identical completed episode set. Negative error SD/RMSE or cost differences mean less risk or cost; positive P&L differences mean higher net P&L. The CSV/JSON also includes signed error, absolute/squared error, P&L SD, CVaR95, turnover and fill differences.

| Policy | Reference | Error SD difference | Error RMSE difference | Mean P&L difference | Mean cost difference |
|---|---|---:|---:|---:|---:|
| Mechanical delta | Unhedged | -826.65 | -825.82 | -159.90 | 31.12 |
| Parametric | Unhedged | -791.05 | -789.79 | -128.95 | 28.79 |
| Gaussian | Unhedged | -802.78 | -801.48 | -124.68 | 23.23 |
| VolGAN | Unhedged | -818.33 | -813.39 | -102.11 | 33.81 |
| Unhedged | Mechanical delta | 826.65 | 825.82 | 159.90 | -31.12 |
| Parametric | Mechanical delta | 35.61 | 36.02 | 30.95 | -2.33 |
| Gaussian | Mechanical delta | 23.87 | 24.34 | 35.22 | -7.88 |
| VolGAN | Mechanical delta | 8.33 | 12.43 | 57.79 | 2.69 |
| Unhedged | Gaussian | 802.78 | 801.48 | 124.68 | -23.23 |
| Mechanical delta | Gaussian | -23.87 | -24.34 | -35.22 | 7.88 |
| Parametric | Gaussian | 11.73 | 11.69 | -4.28 | 5.56 |
| VolGAN | Gaussian | -15.55 | -11.91 | 22.57 | 10.58 |

## Failures

- Unhedged: COMPLETED=66, STOP_DATA_GAP=5
- Mechanical delta: COMPLETED=66, STOP_DATA_GAP=5
- Parametric: COMPLETED=66, STOP_DATA_GAP=5
- Gaussian: COMPLETED=66, STOP_DATA_GAP=5
- VolGAN: COMPLETED=66, STOP_DATA_GAP=5

## VolGAN calibration and interpretation

The existing nominal 90% return-band coverage is **SPY 49.12%, QQQ 60.79%, IWM 65.86%**. The weak calibration remains unchanged for this financial benchmark. The model is a multi-asset VolGAN adaptation, not a reproduction of the paper's original SPX experiment.

Paired risk and cost differences show the economic results of the existing model. They cannot by themselves attribute performance to calibration: VolGAN also generates IV changes, whereas Gaussian and parametric baselines carry current IV forward. Delayed stock execution and model-valued basket prices remain shared approximations. No model, alpha or trading parameter was tuned from these results.

## Figures

The four figures are in `figures/` as PNG and SVG. The representative episode is the first test episode, selected before outcomes. Distribution, time-series and risk/cost figures use the same matched completed episode set.
