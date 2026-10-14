# Bounded VolGAN calibration correction

This plan was written before candidate training. `plan.json` is the machine-readable contract; its hash is stored with every attempt and the frozen selection.

Diagnosis found no demonstrated alignment, inversion, broadcasting or update defect. The original model uses latent noise in every asset, but its generated return dispersion and QQQ/IWM spread variation are poor. Saved final-epoch diagnostics cannot establish when this weakness emerged. A wider network is therefore not justified by the evidence.

The only change is **checkpoint selection instead of always keeping epoch 10,000**. Train one unchanged configuration with the two predetermined seeds **1729 and 1731**. Keep the original 16/32-unit generator, 16-unit discriminator, shared 32-dimensional noise, 25 calibration epochs, 10,000 training epochs, batch size 100, RMSprop learning rate 0.0001, transformations and source-defined losses.

Evaluate only epochs **100, 250, 500, 1,000, 2,500, 5,000, 7,500 and 10,000**. Reuse identical 256-scenario noise draws, seed 1730, and identical conditioning states for the frozen original model and candidates. Fit weights only through 2017. Use **2018** for development; do not inspect **2019** until one choice is saved with its checkpoint hash. Both years were inspected together previously, so these are developmental checks, not a new untouched holdout.

The primary criterion is mean **joint energy score** in raw log-return units. It scores the equally weighted empirical scenario distribution using Euclidean distances, includes diagonal pairs and uses the same ensemble size throughout. Lower is better. It distinguishes joint distributions, but may be less sensitive to dependence than marginal dispersion; spread and correlation diagnostics remain necessary. [Definition: Gneiting and Raftery (2007)](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

A checkpoint is eligible only if all these predeclared safeguards pass against the frozen original on the same segment:

- Joint energy score improves by at least 2%.
- Mean absolute error of nominal 90% coverage improves by at least 5 percentage points; no asset's absolute coverage error worsens by more than 3 points.
- Each asset's 90% interval score, return-mean forecast RMSE and mean-log-IV forecast RMSE worsens by no more than 10%.
- The QQQ-minus-IWM spread's 90% interval score improves by at least 5%.
- Generated outputs and all required metrics remain finite.

The central 90% interval score is its width plus 20 times any distance outside the interval. This penalizes excessively wide forecasts as well as missed observations. Thresholds are declared research safeguards, not calibrated hypothesis tests or guarantees.

Choose the eligible checkpoint with the lowest 2018 energy score within each seed; ties favor the earlier epoch. **Both seeds must produce an eligible checkpoint.** Then select the lower-scoring seed's checkpoint, with ties resolved by earlier epoch and smaller seed. Save its ID/hash before loading 2019. Apply exactly the same safeguards on 2019 once. If development is inconsistent across seeds or confirmation fails, report **NO CONFIRMED IMPROVEMENT** and stop. Do not try a second choice.

Report coverage with widths, mean and dispersion, energy score, pairwise-correlation distributions, QQQ/IWM spread returns, surface-change variation and return/surface dependence. Distinguish generated conditional variation, variation of conditional means and realized unconditional variation. The frozen Gaussian reference retains its calendar-day scaling and current IV surfaces; its checkpoint was previously selected using both validation years.

Every attempted checkpoint receives a compact diagnostic record and candidate-table row. Only the best eligible checkpoint per seed is retained; nonselected checkpoints can be regenerated from the recorded seed and epoch. Preserve original data, action supplement, checkpoints, results and financial conventions. Keep the solver reliability fix. Do not run a financial benchmark in this task.

Run from the repository root:

```powershell
python -B examples/calibrate_volgan.py develop --seed 1729 --prepared "$env:USERPROFILE/.cache/basket-hedging/prepared-etf-v1-actions"
python -B examples/calibrate_volgan.py develop --seed 1731 --prepared "$env:USERPROFILE/.cache/basket-hedging/prepared-etf-v1-actions"
python -B examples/calibrate_volgan.py select-confirm --prepared "$env:USERPROFILE/.cache/basket-hedging/prepared-etf-v1-actions"
```

Use `--output reports/historical_calibration_reproduction` for a separate reproduction. Repeating an existing seed attempt or confirmation is rejected.
