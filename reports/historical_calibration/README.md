# Bounded VolGAN calibration review

**Result: NO CONFIRMED IMPROVEMENT.** One seed produced a better developmental checkpoint; the second did not. The preregistered consistency rule therefore rejected promotion before evaluating 2019. The frozen VolGAN remains the benchmark model. No historical financial comparison was run.

This is the existing SPY/QQQ/IWM multi-asset adaptation, not a reproduction of the original SPX experiment. The original model, Gaussian reference, financial conventions, solver reliability fix, dataset and corporate-action supplement remain unchanged.

## What was established

**No implementation defect was demonstrated** in asset/date/grid alignment, scaling, inverse transformations, conditioning/label separation, output slicing or gradient updates. Independent formulas reproduced all 1,592 training rows and retained 2018 rows. The training-array hash matches the frozen original. Noise affects every asset's returns and surfaces; distinct output/Jacobian rows rule out accidental duplicated outputs in the inspected path. One isolated update on discarded network copies changed every G and D parameter tensor with finite gradients. This was a diagnostic, not another trained candidate.

**Demonstrated weaknesses:** inadequate predictive dispersion, poor QQQ-minus-IWM spread forecasts, almost perfectly correlated conditional QQQ/IWM draws, and substantial seed instability. Scheduled diagnostics show underdispersion already at epoch 100 in both runs. Seed 1729 partially recovers and later deteriorates; seed 1731 is unstable. This is not simply a late-training collapse.

**Unproven explanations:** small network width and the adversarial/smoothness balance. For one fixed training batch, weighted moneyness/maturity gradient norms divided by the adversarial gradient norm were 4.34/4.27 at initialization and 0.364/0.0915 at the frozen final checkpoint. These local measurements do not establish causation throughout training. No width or loss change was justified.

See [diagnosis.md](diagnosis.md) and [diagnosis.json](diagnosis.json) for alignment, Jacobian, gradient and covariance checks. Their initial diagnostic used 512 draws with seed 4402; the selection experiment below consistently uses 256 draws with seed 1730.

## Minimal change and fixed experiment

The only modelling-procedure change was **validation-based checkpoint selection instead of always retaining epoch 10,000**. Architecture, transformations, source-defined objectives, calibration procedure and optimizer settings are unchanged. The production trainer has an optional isolated checkpoint callback; its default final-epoch behavior remains unchanged. A small empirical multivariate energy-score function and additional diagnostics reuse the existing validation path.

The [plan](plan.md) and [machine-readable rules](plan.json) were saved before training. Exactly **one configuration and two seeds** were attempted:

- Shared latent dimension 32; original generator widths 16/32 and discriminator width 16.
- 25 calibration epochs and 10,000 main epochs, batch 100, RMSprop learning rate 0.0001.
- Training seeds **1729** (original) and **1731**. Checkpoints at **100, 250, 500, 1,000, 2,500, 5,000, 7,500, 10,000**.
- Weights/preprocessing use the existing training period ending 2017-12-29. Development uses 227 target states from 2018-01-03 through 2018-12-31. All model comparisons share those states and reproducible noise draws.
- Primary criterion: lower joint energy score in raw log-return units. Eligibility also requires improved coverage errors and QQQ/IWM spread interval score, with limits on deterioration of per-asset interval scores, return-mean RMSE and full-grid log-IV mean RMSE. Exact predeclared thresholds are in the plan; they are research screens, not statistical significance tests.
- Both seeds must have an eligible checkpoint before freezing a choice and evaluating 2019 once. That condition failed. **2019 confirmation was not evaluated.** Both validation years had previously been inspected together and are developmental checks, not a new untouched holdout.

The archive loader verifies/unpacks the prepared dataset before slicing its arrays; only the through-2018 slice enters development and training-pair construction. No 2020–2025 hedge outcomes enter this procedure. This distinction matters: absence of confirmation evaluation does not mean that the storage reader never opened an archive containing later dates.

The original seed's final generator and discriminator weight hashes exactly match the frozen original, demonstrating that scheduled observation did not change the training updates.

## Every attempted checkpoint

These are 16 snapshots from two training runs, not 16 separate training configurations. Failed safeguards and weight hashes are retained in [candidates.csv](candidates.csv); complete diagnostics are in each seed's `checkpoints.json`.

| Epoch | Seed 1729 energy score | Eligible | Seed 1731 energy score | Eligible |
|---:|---:|:---:|---:|:---:|
| 100 | 0.016039 | No | 0.017313 | No |
| 250 | 0.015279 | No | 0.094085 | No |
| 500 | 0.015578 | No | 0.018484 | No |
| 1,000 | 0.021226 | No | 0.021138 | No |
| 2,500 | 0.013072 | Yes | 3.947647 | No |
| 5,000 | **0.011764** | **Yes** | 0.047252 | No |
| 7,500 | 0.012629 | No | 0.024957 | No |
| 10,000 | 0.014050 | No | 0.015778 | No |

Seed 1729/epoch 7,500 achieves apparently attractive marginal coverages of 88.11%/88.99%/95.59%, but fails the spread interval-score safeguard. It was rejected. Seed 1731/epoch 2,500's very poor finite score is retained rather than hidden.

The best eligible within-seed checkpoint, **1729/5,000**, is saved as `seed_1729/best_eligible.json` for auditability. It is **unconfirmed and not accepted**. Its file hash is `a1bdb3aa45d04ce7678dc88fe9d6b6361cff7e997968845b9712d9018d51eb79`. There is no promoted `accepted.json`. [selection.json](selection.json) and [result.json](result.json) record the rejection.

## Developmental comparison

Below, “candidate” means the best eligible **within seed 1729**, not an accepted replacement. Asset order is SPY / QQQ / IWM. All return numbers are log-return units unless percentages are explicitly shown. Lower scores/RMSE are better.

The existing Gaussian reference uses calendar-day scaling and carries current IV surfaces forward; VolGAN generates joint next-trading-session returns and surface changes. The Gaussian checkpoint was previously selected using combined 2018–2019 validation. It is a descriptive reference, not a new independently selected competitor.

| Diagnostic | Frozen VolGAN | Unconfirmed 1729/5,000 | Frozen Gaussian reference |
|---|---|---|---|
| Joint energy score | 0.014050 | 0.011764 | 0.011694 |
| Central 90% return coverage | 45.37% / 56.83% / 63.88% | 65.20% / 67.84% / 80.18% | 69.16% / 68.28% / 77.97% |
| Mean 90% interval width | .01177 / .01959 / .01824 | .01769 / .02772 / .03485 | .01689 / .02494 / .02359 |
| Mean 90% interval score | .08918 / .10516 / .07043 | .05512 / .06723 / .05107 | .05897 / .08007 / .05701 |
| Mean forecast return | -.00323 / .00179 / .00137 | -.00052 / -.00076 / -.00182 | .00036 / .00073 / .00028 |
| Mean within-state return SD | .00375 / .00646 / .00603 | .00589 / .00913 / .01167 | .00519 / .00769 / .00726 |
| Across-date SD of forecast means | .00376 / .00577 / .00374 | .00349 / .00568 / .00517 | .00080 / .00100 / .00081 |
| Return-mean forecast RMSE | .01229 / .01544 / .01226 | .01011 / .01354 / .01137 | .01053 / .01393 / .01102 |

Observed across-date return means are -.00017 / .00004 / -.00031; observed unconditional SDs are .01047 / .01383 / .01087. These are **not** the same objects as conditional scenario variation or variation of conditional means. JSON diagnostics retain all three covariance objects separately, including the finite-ensemble adjustment in the generated-mixture covariance identity. No covariance matching is imposed.

The candidate improves energy score by 16.27% and improves interval scores as well as coverage; wider intervals alone are not the acceptance criterion. It nevertheless remains undercalibrated. The previously reported 49.12%/60.79%/65.86% frozen coverage used combined 2018–2019. Its difference from the 2018-only values above is not a transform correction or an overwritten baseline.

### Joint behaviour

Each entry is the 5th percentile / median / 95th percentile of **within-state generated correlations across conditioning dates**.

| Pair | Frozen VolGAN | Unconfirmed candidate | Gaussian reference |
|---|---|---|---|
| SPY–QQQ | .94866 / .98562 / .99151 | .99803 / .99902 / .99936 | .33585 / .62581 / .86053 |
| SPY–IWM | .95044 / .98673 / .99184 | .99841 / .99926 / .99959 | .33711 / .55891 / .75491 |
| QQQ–IWM | .99880 / .99936 / .99961 | .99717 / .99930 / .99954 | .05049 / .42795 / .68159 |

The candidate does **not** fix extreme conditional dependence; its SPY correlations become more concentrated. Correlation cannot be judged by forcing conditional covariance to equal realized unconditional covariance. The observed spread outcomes provide a separate predictive check:

| QQQ-minus-IWM diagnostic | Frozen VolGAN | Unconfirmed candidate | Gaussian reference |
|---|---:|---:|---:|
| Mean forecast spread return | .000416 | .001060 | .000448 |
| Mean within-state SD | .000503 | .002603 | .008013 |
| Across-date SD of forecast means | .004348 | .000952 | .001138 |
| Central 90% coverage | 6.17% | 38.33% | 88.11% |
| Mean interval width | .001607 | .007230 | .026020 |
| Mean interval score | .115325 | .068138 | .034435 |

Observed unconditional spread mean is .000346 and SD .007447. Although the candidate improves spread forecasts, 38.33% coverage remains poor. A lower joint energy score is insufficient evidence that dependence is calibrated.

### Surface changes and downstream use

| Diagnostic, SPY / QQQ / IWM | Frozen VolGAN | Unconfirmed candidate | Gaussian reference |
|---|---|---|---|
| Full-grid log-IV mean-forecast RMSE | .08364 / .07291 / .07412 | .07835 / .07054 / .07263 | .07686 / .06910 / .07169 |
| Mean grid-average log-IV change | .00475 / .00222 / -.00225 | .00987 / .01060 / .00736 | 0 / 0 / 0 |
| Mean conditional SD of grid-average change | .02929 / .03108 / .02751 | .02876 / .02797 / .02145 | 0 / 0 / 0 |
| Across-date SD of conditional mean change | .01208 / .00873 / .01128 | .00797 / .00588 / .00504 | 0 / 0 / 0 |
| Median conditional return/change correlation | -.90863 / -.94484 / -.94944 | -.99277 / -.99213 / -.99440 | Undefined: no IV variation |

Observed grid-average log-IV change means are .00247 / .00236 / .00169 and unconditional SDs .06612 / .05795 / .05281. Candidate surface mean forecasts improve modestly in RMSE but exhibit positive mean-change bias and even stronger negative return/surface dependence. These basic diagnostics do not certify surface calibration or absence of static arbitrage.

An independent 2018 dependency probe used the actual frozen generator and existing basket repricer. Keeping generated spots fixed while replacing generated IV with carried IV changed the liability response by up to **USD 75.00864323**; stock regressors were identical. Generated surfaces therefore genuinely enter optimizer inputs. This is a dependency check, not a trade, model-selection criterion or revised financial benchmark. Details are in [valuation_dependency_probe.json](valuation_dependency_probe.json).

## Files and verification

Production changes are limited to `src/basket_hedging/volgan.py` (optional isolated checkpoint observation and existing-validator diagnostics) and `src/basket_hedging/generator_validation.py` (small energy score). Added `examples/calibrate_volgan.py`, `tests/test_volgan_calibration.py`; extended `tests/test_generator_validation.py`. New experiment artifacts are confined to this directory. Other pre-existing repository changes were preserved.

Focused tests cover independent energy-score values and input rejection, callback RNG/weight isolation, identical final training weights, checkpoint schedules, future-data mutation, year/noise alignment and acceptance safeguards. Existing joint-GAN and surface-repricing tests remain in use. Independent scalar calculations reproduce the extended metrics; see [extended_metrics_review.md](extended_metrics_review.md).

Actual commands/results:

| Command | Result |
|---|---|
| `python -B -m pytest tests/test_generator_validation.py -q` | 33 passed in 0.74 s |
| `python -B -m pytest tests/test_volgan_calibration.py tests/test_volgan.py tests/test_historical_hedge.py -q` | 47 passed in 8.15 s |
| `python -B -m pytest -q` | **613 passed in 26.65 s** |
| Both declared `develop` runs | Completed; 8 checkpoints each, approximately 20 minutes each while running concurrently |
| `select-confirm` | Exit 0; development consistency gate failed, no confirmation evaluation |
| `git diff --check` | Exit 0; existing line-ending warnings only |

[preservation.json](preservation.json) checks **331 pre-task files**: 329 unchanged, with exactly the two intended production-source changes and no unexpected difference. It includes original data/checkpoint/report hashes and financial source files. Original VolGAN file SHA-256 is `04e3a56793f3a3bf9b7c1ff11c1bf162c8ac497f44d92c80480a093d51d1c12b`; Gaussian is `199f0db9f33334600b8d117e18f45e5a49f8b4c0e098bb1864358a19b967c18a`. Plans and per-seed attempts record input, training-array and source hashes. Hashes establish preservation, not market-data accuracy.

## Reproduce separately

Run from the repository root with the existing historical prepared dataset and environment (this run: Python 3.13, NumPy 2.3.1, PyTorch 2.12.1 CPU). Use a new output directory; the runner rejects repeated seed/confirmation attempts and protects the frozen historical directory.

```powershell
$prepared = "$env:USERPROFILE/.cache/basket-hedging/prepared-etf-v1-actions"
$out = "reports/historical_calibration_reproduction"
python -B examples/calibrate_volgan.py develop --seed 1729 --prepared $prepared --output $out
python -B examples/calibrate_volgan.py develop --seed 1731 --prepared $prepared --output $out
python -B examples/calibrate_volgan.py select-confirm --prepared $prepared --output $out
python -B -m pytest tests/test_generator_validation.py tests/test_volgan_calibration.py tests/test_volgan.py tests/test_historical_hedge.py -q
python -B -m pytest -q
git diff --check
```

The saved diagnosis and independent probes can be rerun with `python -B reports/historical_calibration/diagnosis_probe.py`, `metrics_probe.py` or `valuation_dependency_probe.py` using their full path in this directory; they overwrite their own diagnostic JSON only. They do not train candidates or run the financial benchmark.

## Decision and limits

**No replacement is accepted.** Seed instability, severe spread undercoverage and extreme generated dependence remain. The intermediate improvement demonstrates a limitation of final-epoch selection, not a solved calibration problem. Two seeds, one development year and finite scenario ensembles do not establish robust generalization. Neither validation year is a newly untouched holdout. No conclusion about hedge profitability or comparative economic performance is drawn, and no additional search was performed.
