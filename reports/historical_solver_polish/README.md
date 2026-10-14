# Solver-polish validation

This is a separate numerical-reliability validation, not a replacement for
`reports/historical/benchmark`. No checkpoint was trained, and the original
71-episode benchmark and its four recorded failures remain unchanged.

The unchanged frozen experiment was replayed for the four affected VolGAN
episodes and the previously successful first episode, 2020-01-02. All five
completed. Each formerly failed episode used one active-set polish after
20,000 coordinate-descent iterations; the control used none.

| Episode | Original scaled KKT residual | Corrected scaled KKT residual |
|---|---:|---:|
| 2022-06-02 | 1.5604606628e-8 | 1.0706030248e-15 |
| 2022-09-01 | 4.7251503011e-9 | 4.0594454318e-16 |
| 2022-12-01 | 1.9445781939e-8 | 9.1828484496e-16 |
| 2023-02-02 | 1.0757443896e-9 | 4.1738919525e-16 |

The original scaled KKT tolerance remains **1e-9**. Objective, alpha, gross
scale, penalty weights, scenarios, pricing and accounting are unchanged.
These are floating-point checks, not exact optimality certificates.

The exact objective remains

```text
mean((delta_v - intercept - delta_h @ holdings)**2)
    + alpha * gross_scale * sum(cost_weights * abs(holdings - previous_holdings))
```

The intercept is free and unpenalized. With `beta = holdings - previous_holdings`,
`lambda = alpha * gross_scale * cost_weights`, centered design `Xc` and centered
response `yc = center(delta_v - delta_h @ previous_holdings)`, each KKT violation
is divided by `max(1, abs(2 * Xc.T @ yc / N), lambda)`. The maximum must remain
at most `1e-9`, including inactive-coordinate inequalities.

[original_failures.json](original_failures.json) is the saved pre-edit
reproduction of all four failures, including original iteration counts,
residuals and exact input-file hashes.

| Episode | Original outcome | Corrected outcome | Accepted hedge fills |
|---|---|---|---:|
| 2020-01-02 | COMPLETED | COMPLETED | 23 |
| 2022-06-02 | FAILED_OPTIMIZER | COMPLETED | 18 |
| 2022-09-01 | FAILED_OPTIMIZER | COMPLETED | 28 |
| 2022-12-01 | FAILED_OPTIMIZER | COMPLETED | 21 |
| 2023-02-02 | FAILED_OPTIMIZER | COMPLETED | 20 |

Verification recorded in [validation.json](validation.json):

- All four regenerated failed regressions, including previous holdings and
  solver settings, exactly match the captured audit fixtures.
- All earlier postings, order outcomes and decisions match the original failed
  account prefixes.
- All 20 control solves exactly match every original solution field from a
  pre-edit solver snapshot. All control targets are unchanged, and the five
  financial CSVs are byte-identical to the frozen control reports.
- There were 100 successful decision solves, four polished solutions and no
  optimizer failures in these five episodes.
- All **239** frozen input/output hashes are unchanged and match the snapshot
  taken before editing the solver. See [frozen_hashes.json](frozen_hashes.json).

[episodes.csv](episodes.csv) records the separate corrected terminal outcomes.
Each episode directory contains ordinary financial reports, replay events and
the policy's solver diagnostics. The full 71-episode benchmark was not rerun;
these results do not establish improved financial performance.

## Tests run

| Command | Actual outcome |
|---|---|
| `python -B -m pytest tests/test_optimization.py tests/test_scenario_hedge.py tests/test_learned_scenarios.py -q` | 57 passed in 0.83s |
| `python -B -m pytest -q tests/test_optimization_polish.py` | 15 passed in 4.45s |
| `python -B -m pytest -q` | 581 passed in 29.08s |
| `git diff --check` | Exit 0; LF/CRLF conversion warnings only |

No lint or type checker is configured. No model training was run.

## Reproduce

From the repository root, using the existing frozen data and checkpoints:

```powershell
python -B examples/validate_solver_polish.py `
  --prepared "$env:USERPROFILE/.cache/basket-hedging/prepared-etf-v1-actions" `
  --raw "$env:USERPROFILE/.cache/basket-hedging/data-v1" `
  --output reports/historical_solver_polish_reproduction
```

The output directory must not already exist. The runner refuses to write into
the frozen historical reports. This command checks the control against saved
financial CSVs and targets, and the four failed regressions against their exact
fixtures. The recorded validation also used the optional `--original-solver`
argument with the pre-edit source snapshot, to compare every control solution
field directly. That temporary snapshot is not required to reproduce the
corrected runs or the saved-report comparisons.
