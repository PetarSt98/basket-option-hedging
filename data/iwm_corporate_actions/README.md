# IWM corporate-action supplement

`distributions.csv` contains 59 official iShares cash distributions with ex-dates
from 2011-03-24 to 2025-09-16. The verified coverage interval is 2011-03-23 through
2025-12-12. `distributions_manifest.json` records the publisher URL, retrieval
timestamp, raw and normalized SHA-256 hashes, parsing assumptions and every
Yahoo cross-check discrepancy.

The complete embedded iShares table contains 106 distributions from 2000-06-21
through 2026-09-15. It spans both benchmark boundaries. There are four published
events in each year 2011–2024 and three before the cutoff in 2025. Yahoo independently
matches all 59 ex-dates. Its amounts are rounded to three decimal places: 58 differ
from iShares by at most $0.000487 per share. The official amounts are retained.
Their sum is $28.779858 per share; this total is a data check, not an investment return.

The field is **total distribution**, including any cash capital-gain or return-of-capital
components. It is not just dividend income. The CSV preserves the published decimal
precision and uses ex-dates rather than record or payable dates. The prepared data
uses zero on other dates only within the verified coverage interval.

The existing benchmark books cash on the ex-date, before that day's hedge fills.
Actual payable dates are retained here but do not change that declared timing proxy.
These are accounting inputs; the strategy does not receive future action records.
The six original price/option Parquets are unchanged.

To retrieve and normalize a fresh official snapshot:

```powershell
python -B -m basket_hedging.corporate_actions --output data/iwm_corporate_actions
```

This writes an updated retrieval manifest. The original raw publisher and optional
Yahoo responses are retained by hash in
`$env:USERPROFILE\.cache\basket-hedging\corporate-actions`.
Use the bundled CSV/manifest to reproduce this benchmark without refreshing it.
Split verification is recorded separately in `split_verification.json`.
