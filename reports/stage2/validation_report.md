# Stage 2 implementation and validation

Implemented and validated as a synthetic research prototype. The frozen learned generator retained approximately all of the true-simulator optimizer's variance reduction on 200 independent test paths. This does not establish exact conditional distribution recovery, historical performance, or a VolGAN reproduction.

## A–E. Model, method, data, conditioning and split

- Model: 19 inputs → Dense(24) → tanh → Dense(24) → tanh → Dense(9), 1,305 network parameters. Outputs are three mean rates and six full lower-triangular Cholesky entries; positive diagonal uses softplus plus 0.0001.
- Joint log returns: `0.01 * days * mean_rate + 0.01 * sqrt(days) * L @ z`, with independent standard normal noise. Horizon scaling and the Gaussian family are supplied structure, not learned discoveries.
- Training: CPU float64 PyTorch, Adam learning rate 0.001, weight decay 0.0001, batch 512, 40 epochs, seed 271828. Checkpoint 19 was selected by minimum validation NLL (4.5744868611); no financial test outcomes selected parameters.
- Faithful structure: causal conditional scenarios, joint instrument repricing, existing Stage 1 variance/L1 objective, absolute target holdings and independent accounting. Simplification: likelihood-trained conditional Gaussian network, with no GAN/discriminator or implied-volatility surface.
- Data: existing zero-drift, three-stock GBM, fixed volatilities 20%/22.5%/25%, pairwise correlation 0.35. Twenty business-day decision intervals per path, with 09:00/16:00 observations and actual 24/72-hour next-close labels. True dynamics generate data but are not passed to model fitting.
- Inputs: three current log prices, three recent joint return bars normalized by square-root elapsed days, presence masks, log lag durations and log scenario horizon. Only released prefix observations enter features. Train-only mean/scale normalization.
- TRAIN: 1,024 independent paths / 20,480 rows, 2024-01-08 through 2024-02-05 targets. VALIDATION: 256 paths / 5,120 rows, 2025-01-06 through 2025-02-03 targets. TEST: 200 fresh paths starting 2026-01-05, never used for fitting or selection. Model available 2025-12-31.
- Master seed 20260920; namespaced SHA256-derived path/scenario seeds. Split path seeds are disjoint. Frozen model SHA256: `2994d85f3ab13638b268ca08d31891720e4a6bbc44d106d52b7fc9f283c649f8`.

See [the implementation note](../../docs/cont_vuletic_stage2.md) for the exact architecture, source references, feature layout, loss, timing, interfaces and limitations.

## F. Conditional distribution validation

Each of eight states uses 20,000 independent simulator draws and 20,000 generated draws. Values below are worst constituent errors within each state; one basis point is 0.0001 of simple return. Basket W1 is empirical mean absolute sorted-sample distance.

| History | Horizon | Max mean error (bp) | Max variance error | Basket W1 (bp) |
|---|---:|---:|---:|---:|
| normal | 24h | 1.95 | 10.47% | 4.10 |
| normal | 72h | 4.54 | 13.92% | 7.64 |
| recent_positive | 24h | 2.43 | 6.91% | 2.96 |
| recent_positive | 72h | 12.94 | 9.25% | 8.94 |
| recent_negative | 24h | 1.96 | 1.66% | 0.87 |
| recent_negative | 72h | 2.98 | 6.95% | 4.76 |
| cold_start | 24h | 5.13 | 5.70% | 1.11 |
| cold_start | 72h | 26.74 | 7.26% | 6.96 |

Maximum covariance-entry error was 7.03e-5 squared-return units; maximum correlation error was 0.0648. Across the 1%, 5%, 95% and 99% quantiles, the largest constituent error was 47.9 bp and largest basket error was 29.8 bp. Basket variance differences ranged from −0.95% to +12.0%. Full arrays, signed errors and quantiles are in [generator_validation.json](generator_validation.json).

Cold-start/72h was absent from training: every path begins on Monday with a 24h first horizon. Its 26.7 bp maximum mean error is an extrapolation limitation. The oracle GBM has independent future increments; differences across positive/negative histories reflect learned estimation error and sampling noise, not discovered regimes. No synthetic volatility-state variable exists. The model was not retuned after these diagnostics.

## G. Exact file inventory

Added source files:

- `src/basket_hedging/learning_data.py`
- `src/basket_hedging/learned_model.py`
- `src/basket_hedging/learned_scenarios.py`
- `src/basket_hedging/generator_validation.py`
- `src/basket_hedging/stage2.py`

Added tests:

- `tests/test_learning_data.py`
- `tests/test_learned_model.py`
- `tests/test_learned_scenarios.py`
- `tests/test_generator_validation.py`
- `tests/test_stage2.py`

Changed existing files:

- `src/basket_hedging/scenarios.py`: shared input/horizon validation, seed derivation and repricing; original GBM results preserved.
- `src/basket_hedging/cli.py`: Stage 2 command.
- `pyproject.toml`: optional `learn` PyTorch dependency.
- `README.md`: installation/run commands and scope.
- `docs/backtest_spec.md`: explicit subsequently authorized Stage 2 scope; accounting contract unchanged.
- `docs/cont_vuletic_stage1.md`: link to the implemented Stage 2 extension.

Added documentation: `docs/cont_vuletic_stage2.md` and this report.

Generated reports in `reports/stage2/`: `model.json`, `experiment.json`, `training_history.csv`, `generator_validation.json`, `path_results.csv`, `hedging_summary.json`, `test_manifest.json`, `validation_report.md`. Installation also regenerated ignored `src/basket_hedging.egg-info/` metadata. Existing fixtures and reports were preserved.

Byte comparison against the pre-change snapshot confirmed unchanged `engine.py`, `ledger.py`, `orders.py`, `risk.py`, `optimization.py`, `scenario_hedge.py`, `policies.py`, `contracts.py`, `market.py`, `replay.py`, `synthetic.py` and `valuation.py`.

## H. Actual checks

```powershell
python -m pip install -e ".[test,learn]"
python -B -m pytest -p no:cacheprovider -q
python -B -u -m basket_hedging stage2 --output reports/stage2 --test-paths 200 --workers 6
git diff --check
```

- Installation succeeded. Python 3.13.3, NumPy 2.3.1, PyTorch 2.12.1+cpu, pytest 9.1.1.
- Full suite: **465 passed in 22.04 seconds**. This includes deterministic training, disjoint splits, future-data mutation, scaling, dependence, independent payoff/sign/multiplier checks, provider compatibility, unchanged portfolio, old provider behavior and repeatable small end-to-end artifacts.
- Full command completed **200 paths × 4 policies = 800 runs**. Both optimized policies made 4,000 decisions. Zero skipped decisions, rejected orders or margin breaches.
- Independent comparison against the original provider on three decisions was bit-for-bit equal. A constructed oracle-equivalent learned Gaussian matched true-provider repricing to 3.13e-13 (liability) and 1.42e-14 (hedges).
- An independent standard-library `statistics` calculation recomputed all 36 reported aggregate financial/activity metrics from the 800 CSV rows; each matched within absolute 1e-10 / relative 1e-12 tolerances.
- Source/model hashes match the bundled experiment manifest; test seeds match path rows and remain disjoint from train/validation.
- `git diff --check` returned exit 0. The repository files were already untracked, so added/changed source and documentation were also checked directly for trailing whitespace against the pre-change snapshot.
- No separate lint/type tool is configured. No historical-market validation was run.

## I. Held-out hedging comparison

All 200 paths completed for every policy. Terminal metrics therefore cover the same 200 paths. Both optimizers used alpha 0.1, stock-only candidates, cost proxy 0.02/share, 256 scenarios, 1,024 inner pricing samples, identical paired base normals, and the unchanged engine/execution/cost configuration.

| Policy | Error SD | Error RMSE | Mean terminal P&L | P&L SD | CVaR95 loss |
|---|---:|---:|---:|---:|---:|
| unhedged | 252.867 | 252.516 | 19.188 | 252.924 | 700.979 |
| delta | 80.837 | 80.714 | -4.084 | 80.964 | 178.068 |
| true_simulator | 81.420 | 81.299 | -3.856 | 81.534 | 181.200 |
| learned_generator | 81.286 | 81.168 | -3.824 | 81.407 | 181.294 |

| Policy | Mean turnover | Mean trading costs | Mean fills | Breach rate | Completion |
|---|---:|---:|---:|---:|---:|
| unhedged | 0.00 | 0.0000 | 0.000 | 0% | 100% |
| delta | 23048.35 | 7.7534 | 62.235 | 0% | 100% |
| true_simulator | 23038.87 | 7.6190 | 59.655 | 0% | 100% |
| learned_generator | 23029.21 | 7.6274 | 59.850 | 0% | 100% |

Error is gross `-(liability P&L + hedge P&L)`, excluding trading/funding costs; net P&L includes them. Currency amounts use the demonstration account currency. Turnover counts both sides of executed trades. Empirical CVaR95 averages the ten worst terminal losses; 200 paths provide limited tail information.

The true-simulator optimizer reduced hedge-error variance by **89.632%** relative to unhedged; the learned generator reduced it by **89.666%**. Retained variance reduction was **100.038%**, without clamping. The learned result's tiny advantage over the oracle scenario estimate does not establish superiority. Mechanical delta had slightly lower error SD than either optimizer. No objective, alpha, engine setting or model parameter was altered to improve these outcomes.

## J–K. Limits and next step

The software integration and synthetic risk-reduction objective are demonstrated. The model does not exactly recover conditional distributions, particularly on unseen history/horizon combinations. It assumes Gaussian innovations and diffusion time scaling, has no jumps/heavy tails/stochastic volatility, and can learn spurious history dependence. Fixed risk-neutral pricing and Monte Carlo approximation remain. The unchanged optimizer still approximates delayed execution with a decision-to-next-review objective. The research margin proxy is not broker margin.

Remaining differences from the paper include no VolGAN adversarial architecture, no generated implied-volatility surfaces, no surface validity penalties, no historical calibration or replication of empirical results, and no data-selected alpha.

The next real-data/surface stage would require point-in-time constituent spot/option records, explicit option conventions and a joint surface representation, a validated surface generator and consistent repricing, independent chronological evaluation, and Polykern's actual payoff, hedge-universe, margin, funding and borrow requirements. This implementation does not supply or assume those missing inputs.
