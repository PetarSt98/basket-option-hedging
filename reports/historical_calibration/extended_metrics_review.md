# Independent review of extended developmental metrics

No implementation issue found in the inspected formulas. Production source,
the preregistered plan and training processes were not modified.

`metrics_probe.py` uses a manufactured two-state, four-scenario, three-asset
example. Independent scalar/pairwise formulas reproduce:

- empirical Euclidean energy score, including the diagonal pairs;
- linearly interpolated central-90% bounds, widths, coverage and interval
  score `width + 20 * distance outside the interval`;
- sample within-state covariance separately from the population covariance
  of conditional means and the pooled generated mixture covariance;
- conditional pairwise correlations and their quantiles;
- per-asset grid-average log-IV changes, while mean-forecast surface RMSE
  remains a full-grid calculation;
- within-state return versus grid-average log-IV-change correlations;
- the selected target year and exactly identical draw sequences between
  separate-year evaluation and the combined evaluation.

The covariance conversion multiplies the averaged sample covariance by
`(N-1)/N` before adding population covariance of conditional means. This
correctly describes the empirical mixture and does not equate it with the
realized unconditional covariance.

The illustrative 2019 rows in the test are fabricated test data only. No
historical 2019 confirmation outcomes were inspected. The two candidate
directories contain identical original-2018 diagnostics: 227 states from
2018-01-03 through 2018-12-31. Reported 90% coverage corresponds to counts
103/227, 129/227 and 145/227; both directories use the same baseline.

`valuation_dependency_probe.py` separately follows the actual frozen VolGAN
checkpoint through existing basket repricing on the first 2018 conditioning
state. It uses a declared diagnostic basket, no engine or financial run.
With generated spots fixed, replacing generated surfaces by carried-current
surfaces leaves every stock hedge regressor identical but changes the
liability response: maximum absolute difference USD 75.00864323, mean
absolute difference USD 7.63438451. This establishes a real path from the
generator's surfaces into the unchanged optimizer input; it is not a model
selection diagnostic or a changed historical benchmark.

Commands (from the repository root):

```text
python -B C:/Users/petar/AppData/Local/Temp/volgan-calibration-diagnosis-98atsm1h/metrics_probe.py
python -B C:/Users/petar/AppData/Local/Temp/volgan-calibration-diagnosis-98atsm1h/valuation_dependency_probe.py
```

Both completed successfully. The detailed JSON artifacts accompany the
probes. All written files are in the separate temporary diagnosis directory.
