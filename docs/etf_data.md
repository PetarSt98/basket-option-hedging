# Historical ETF data and surface preparation

The historical input is the **data-v1 release of
[lambdaclass/options_portfolio_backtester](https://github.com/lambdaclass/options_portfolio_backtester/releases/tag/data-v1)**.
It is not OptionMetrics. Download exactly these six files:

| File | SHA-256 |
|---|---|
| SPY_options.parquet | `a7152991b45b81f090f970e945bf88def8093b8ecb9b250e9891cb6d88041f0a` |
| SPY_underlying.parquet | `847e60a441eb10969d87cd4a6da604257b782d9076168f68d5730b84096c79db` |
| QQQ_options.parquet | `1f831556cd87ec9d7af43d3b47a69a829ffb2fb199cfcef4ff55d289badf8734` |
| QQQ_underlying.parquet | `78c015fd7fa9d493cd798d5ae81b3c2ae6dc3b8176e25e7bb90cd4114b103351` |
| IWM_options.parquet | `d16a728116f8095c97c39ed66e75000bf66a0ab422b678002eb9f6086fd10ee5` |
| IWM_underlying.parquet | `e980f50473a8c346ba9ac8b05dcf3da691581dae742f649cb3608d0efe2ef940` |

Hashes are pinned to `scripts/fetch_data.py` at commit
`e53ef86928777de6ee0721424762ea3dc133f993`. A mismatch stops processing; there
is no alternative-source or synthetic fallback. Files remain outside the
repository in the user's cache. No licensed market records are bundled here.

The upstream software is MIT licensed. Its separate
[data notice](https://github.com/lambdaclass/options_portfolio_backtester/blob/e53ef86928777de6ee0721424762ea3dc133f993/data/DATA_NOTICE.md)
describes research/educational redistribution, disclaims ownership of price
facts and accuracy, and says the original vendor was not documented. This is
not evidence of an audited vendor extraction or unrestricted commercial rights.

## Actual coverage and schema

The intersection of underlying and chain dates across all three ETFs is
**23 March 2011 through 12 December 2025: 3,693 dates**. Having these endpoints
does not mean every exchange session is present. Twelve transitions in this
intersection cross a missing constituent/chain session. They are recorded in
the prepared manifest and the `consecutive` mask, rather than treated as a
single ordinary model transition. `common_date_gaps` lists each preceding
date, resumption date and the exact missing dates from the underlying union.
It can therefore drive a visible data-gap stop inside an existing episode,
rather than silently omitting that episode from the comparison.

Underlying files contain `date`, `symbol`, raw `open/high/low/close`,
`adjusted_close`, `volume`, `dividend_amount` and `split_coefficient`.
IWM's adjusted close, dividends and split coefficients are entirely null.
The loader preserves missing corporate actions as NaN and explicitly reports
the resulting funding/accounting data blocker in the original preparation.
A separately verified [IWM supplement](../data/iwm_corporate_actions/README.md)
now supports a second prepared copy for accounting. Its merge changes only IWM
dividends and split factors; the original six Parquets and training arrays are
preserved. Zero distributions and factor one are used only within verified
complete coverage, never inferred from null values.

Option files contain `contract_id`, `symbol`, `expiration`, `strike`, `type`,
`last`, `mark`, `bid/ask`, displayed sizes, `volume`, `open_interest`, `date`,
`implied_volatility`, `delta/gamma/theta/vega/rho` and `in_the_money`.
The preparation uses raw closes, raw strikes, annual-decimal IVs and relative
vega weights. A common unit scaling of all same-day vegas cancels from these
weights. The source does not document the Greek/IV estimation method.
Vendor ITM flags and `mark` are not used; moneyness is recomputed as strike/spot.

These are American ETF options. Their supplied IVs are used as market-state
features. This does not turn them into European option quotes. The basket is
a separate European, model-valued, non-traded instrument. The headline label is:

> Real historical constituent prices and option surfaces; model-valued non-traded basket option.

## Deterministic cleaning and smoothing

For each asset and date, duplicate contract IDs and conflicting symbols cause
an error. Filters then remove, in the following order: malformed contract
terms; nonfinite inputs; negative bid, nonpositive ask or crossed quotes; nonpositive traded volume;
nonpositive IV/vega; expired observations; and in-the-money observations.
Calls with K/S >= 1 and puts with K/S < 1 form the retained OTM sample.
The manifest records disjoint exclusion counts and retained counts. Volume
is a surface-quality filter, not assumed fill capacity for stock hedges.

Actual quote-to-expiration date differences use calendar days divided by 365.
Expiry dates are not moved to manufacture fixed maturity observations.
Same-day options are excluded; historical Saturday expiry labels remain as
supplied for surface construction. No options are executed or settled in this
stock-only experiment, so no guessed AM/PM settlement convention is introduced.

Surfaces use the published VolGAN grid, independently for each ETF:

- Moneyness: 0.6, 0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3, 1.4.
- Maturity in years: 1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1.

The one-day grid node is the paper's 1/252; individual observed maturities
still use ACT/365. Arrays are moneyness by maturity; flattening places every
moneyness node of the first maturity first. Joint model features place each
asset's complete flattened surface in SPY, QQQ, IWM order.

The estimator is vega-weighted Nadaraya-Watson smoothing, with Gaussian weights
proportional to `vega * exp(-dm**2/(2*h_m) - dtau**2/(2*h_tau))`.
The ETF adaptation predeclares `h_m = h_tau = 0.01`. These numbers were not
selected on final hedging outcomes and are not asserted to match the paper's
calibrated bandwidths. Extreme grid points may have distant supporting quotes.
The output is not certified arbitrage-free. Interpolation is linear first in
moneyness, then maturity, with linear extrapolation. Nonpositive extrapolated
IVs are rejected instead of silently clipped.

The method and ordering follow the [VolGAN paper](https://doi.org/10.1080/1350486X.2025.2471317)
Section 4.1 and the [hedging paper](https://link.springer.com/article/10.1007/s10479-025-06867-3)
Section 4.1. The public code's older 10-by-8 example differs from the
published 11-by-9 grid; this implementation chooses the latter. No unlicensed
source code was copied from the VolGAN repository.

## Prepared data and chronology

`prepare_etf_data(raw_directory, output_directory)` writes `surfaces.npz`
and `manifest.json`. `load_prepared(output_directory)` validates its hash and
returns `(arrays, manifest)`. No strategy receives this complete dataset.

| Array | Shape and meaning |
|---|---|
| `dates`, `symbols` | ISO dates; fixed order SPY/QQQ/IWM |
| `open`, `close` | N by 3, unadjusted observed prices |
| `dividends`, `splits` | N by 3; missing fields remain NaN |
| `iv` | N by 3 by 11 by 9, smoothed vendor IVs |
| `valid_surface` | N by 3, finite positive surface availability |
| `consecutive` | N booleans; row i follows row i-1 in the observed union calendar |
| `moneyness`, `maturities` | Grid axes |

All common dates remain present even when a surface cannot be constructed.
Training must mask incomplete observations and nonconsecutive transitions,
without compressing a missing session into a one-step label. The chronological
split was declared before benchmark results: training through 29 December
2017; validation 2 January 2018 through 31 December 2019; testing from
2 January 2020 through the common end date. Weekend intervals remain one
observed session transition; no GBM square-root-time rescaling is applied to GAN outputs.

Only same-day observations enter surface smoothing. No test-set normalizer is
fitted and no future value is used to fill a missing surface. The NPZ and each
raw input are SHA-256 recorded in the manifest. Dates are EOD labels, not
verified publication times or executable intraday stock quotes. The historical
execution adapter must declare its separate release and fill assumptions.
