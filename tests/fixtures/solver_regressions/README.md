# Frozen historical solver regressions

These four small NPZ files are byte-for-byte copies of the regressions captured
by the independent historical VolGAN audit. They contain the actual 256 scenarios,
previous holdings, cost weights and final coordinate-descent changes from the
four original `FAILED_OPTIMIZER` episodes. They are not newly generated scenarios.

`manifest.json` records each file's SHA-256, decision time, scenario seed,
instrument order, unchanged solver arguments and original failure residual.
Reference objectives and changes came from independently enumerating all 27
negative/zero/positive active-sign patterns, corroborated using 60-digit
arithmetic on the captured binary64 inputs. The weighted L1 objective and
binary64 penalty weights were unchanged; no ridge or position bounds were added.

The original captures remain in the separate audit directory
`historical-volgan-audit-lnphdcqq/solver`. The original historical benchmark and
its failures remain under `reports/historical/benchmark`; these fixtures do not
replace them. The test suite needs only NumPy and pytest, not the audit's
high-precision library, models, historical data or a training run.
