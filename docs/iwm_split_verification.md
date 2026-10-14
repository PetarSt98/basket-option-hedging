# IWM split verification

For **2011-03-23 through 2025-12-12**, two separately operated public
split-history sources report no IWM split. Both list the same sole historical
event: **2005-06-09, two new shares for each old share**, before this benchmark.

| Source | Evidence checked on 2026-10-04 UTC |
| --- | --- |
| [SplitHistory / BNK Invest](https://www.splithistory.com/iwm/) | Explicit lifetime count of one split; table lists June 9, 2005, 2:1. The page contains current 2026 data. Its older fund-assets paragraph does not date the split-history table. |
| [Twelve Data](https://twelvedata.com/markets/141255/etf/nyse/iwm/historical-data/splits) | Complete public splits table has one row: June 9, 2005, 2:1; no additional page or date filter is exposed. The surrounding page contains October market information and 2026 copyright. |

The [SEC's July 12, 2007 notice, Release 34-56051](https://www.sec.gov/rules/sro/bse/2007/34-56051.pdf),
pages 2–3, independently corroborates the June 2005 2:1 event. This primary
historical filing cannot establish the absence of later splits; that conclusion
uses the two current historical tables above. The SEC text was available through
web extraction, but a direct download returned HTTP 403, so no SEC file hash is
claimed.

The separate supplement
[`split_verification.json`](../data/iwm_corporate_actions/split_verification.json)
records each URL, retrieval time, response-body SHA-256, normalized event facts
and parsing assumptions. Response hashes identify the inspected HTML bytes;
live market fields and advertising can change them on later retrievals.

The integration rule is to set IWM's split factor to **1.0 only within the
verified closed interval**. The normalized in-period event list is empty.
There are no fabricated null split events, and the 2005 split is not applied
again. Prices, option quotes, fixed basket quantities and SPY/QQQ records remain
unchanged.

This is documented research-data verification, not an exchange certification of
completeness. The providers are separate organizations; whether they share any
upstream data vendor has not been established. Coverage after 2025-12-12 is not
inferred. These records are accounting inputs and do not enter strategy features.
