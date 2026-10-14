"""Pinned public ETF data and deterministic daily surface preparation.

No OptionMetrics, substitute download, executable option quote or synthetic
replacement is constructed here. Vendor IVs/Greeks describe American ETF
options; their undisclosed estimation conventions remain a dataset limitation.
"""

from collections import Counter
from pathlib import Path
import hashlib
import json
from urllib.request import Request, urlopen

import numpy as np

from .contracts import InputError
from .surfaces import PAPER_GRID, smooth_surface


SYMBOLS = ("SPY", "QQQ", "IWM")
REPOSITORY = "lambdaclass/options_portfolio_backtester"
RELEASE = "data-v1"
SOURCE_COMMIT = "e53ef86928777de6ee0721424762ea3dc133f993"
RELEASE_URL = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE}"
HASHES = {
    "SPY_options.parquet": "a7152991b45b81f090f970e945bf88def8093b8ecb9b250e9891cb6d88041f0a",
    "SPY_underlying.parquet": "847e60a441eb10969d87cd4a6da604257b782d9076168f68d5730b84096c79db",
    "QQQ_options.parquet": "1f831556cd87ec9d7af43d3b47a69a829ffb2fb199cfcef4ff55d289badf8734",
    "QQQ_underlying.parquet": "78c015fd7fa9d493cd798d5ae81b3c2ae6dc3b8176e25e7bb90cd4114b103351",
    "IWM_options.parquet": "d16a728116f8095c97c39ed66e75000bf66a0ab422b678002eb9f6086fd10ee5",
    "IWM_underlying.parquet": "e980f50473a8c346ba9ac8b05dcf3da691581dae742f649cb3608d0efe2ef940",
}
OPTION_COLUMNS = ("date", "expiration", "contract_id", "symbol", "type", "strike",
                  "bid", "ask", "volume", "open_interest", "implied_volatility", "vega")
UNDERLYING_COLUMNS = ("date", "symbol", "open", "close", "dividend_amount", "split_coefficient")
SPLIT = {"train_end": "2017-12-29", "validation_start": "2018-01-02",
         "validation_end": "2019-12-31", "test_start": "2020-01-02"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InputError("historical data requires the 'historical' dependency extra") from error
    return pq


def download_release(directory: str | Path) -> dict:
    """Fetch exactly the six pinned files; reject differing existing/downloaded bytes."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, expected in HASHES.items():
        path = root / filename
        if not path.exists():
            temporary = root / (filename + ".partial")
            request = Request(f"{RELEASE_URL}/{filename}", headers={"User-Agent": "basket-hedging-research"})
            try:
                with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            except Exception as error:
                raise InputError(f"DATA_BLOCKED: download failed for {filename}; no fallback used") from error
            if sha256(temporary) != expected:
                raise InputError(f"DATA_BLOCKED: SHA-256 mismatch for {filename}; partial file retained")
            temporary.replace(path)
        actual = sha256(path)
        if actual != expected:
            raise InputError(f"DATA_BLOCKED: SHA-256 mismatch for existing {filename}")
        records.append({"file": filename, "sha256": actual, "bytes": path.stat().st_size,
                        "url": f"{RELEASE_URL}/{filename}"})
    report = {"repository": REPOSITORY, "release": RELEASE, "hash_source_commit": SOURCE_COMMIT,
              "hash_source_file": "scripts/fetch_data.py", "files": records}
    (root / "verified_downloads.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _underlying(path: Path, symbol: str) -> dict[str, np.ndarray]:
    table = _parquet().read_table(path, columns=list(UNDERLYING_COLUMNS))
    data = {name: table[name].to_numpy(zero_copy_only=False) for name in UNDERLYING_COLUMNS}
    dates = np.asarray(data["date"], dtype="datetime64[D]")
    if np.isnat(dates).any() or len(np.unique(dates)) != len(dates):
        raise InputError(f"{symbol}: missing or duplicate underlying dates")
    order = np.argsort(dates)
    result = {name: value[order] for name, value in data.items()}
    result["date"] = dates[order]
    if not np.all(result["symbol"] == symbol):
        raise InputError(f"{symbol}: conflicting underlying symbols")
    for key in ("open", "close", "dividend_amount", "split_coefficient"):
        result[key] = np.asarray(result[key], dtype=float)
    for key in ("open", "close"):
        if not np.isfinite(result[key]).all() or np.any(result[key] <= 0):
            raise InputError(f"{symbol}: nonpositive/nonfinite {key}")
    for key in ("dividend_amount", "split_coefficient"):
        finite = np.isfinite(result[key])
        if np.isinf(result[key]).any() or np.any(result[key][finite] < 0):
            raise InputError(f"{symbol}: invalid {key}")
    if np.any(result["split_coefficient"] == 0):
        raise InputError(f"{symbol}: zero split coefficient")
    return result


def _option_dates(path: Path) -> np.ndarray:
    dates: set[str] = set()
    for batch in _parquet().ParquetFile(path).iter_batches(batch_size=262144, columns=["date"]):
        values = batch.column(0).to_numpy(zero_copy_only=False).astype("datetime64[D]")
        if np.isnat(values).any():
            raise InputError("option records have missing quote dates")
        dates.update(np.unique(values).astype(str))
    return np.array(sorted(dates), dtype="datetime64[D]")


def check_data_readiness(directory: str | Path) -> dict:
    """Readiness for preparation is separate from corporate-action completeness."""
    root = Path(directory)
    report = {"status": "DATA_BLOCKED", "repository": REPOSITORY, "release": RELEASE,
              "hash_source_commit": SOURCE_COMMIT, "files": {}, "issues": []}
    for filename, expected in HASHES.items():
        path = root / filename
        if not path.is_file():
            report["issues"].append(f"missing {filename}")
            continue
        actual = sha256(path)
        report["files"][filename] = {"sha256": actual, "matches": actual == expected}
        if actual != expected:
            report["issues"].append(f"SHA-256 mismatch: {filename}")
    if report["issues"]:
        return report
    joint = None
    action_gaps = []
    coverage = {}
    try:
        for symbol in SYMBOLS:
            underlying = _underlying(root / f"{symbol}_underlying.parquet", symbol)
            available = _option_dates(root / f"{symbol}_options.parquet")
            common = np.intersect1d(underlying["date"], available)
            if len(common) == 0:
                raise InputError(f"{symbol}: no joint underlying and option observations")
            coverage[symbol] = {"start": str(common[0]), "end": str(common[-1]), "dates": len(common)}
            joint = common if joint is None else np.intersect1d(joint, common)
            for key in ("dividend_amount", "split_coefficient"):
                if np.any(np.isnan(underlying[key])):
                    action_gaps.append(f"{symbol}: missing {key}")
        if joint is None or len(joint) == 0:
            raise InputError("no common dates across SPY, QQQ and IWM")
    except (InputError, ValueError, KeyError) as error:
        report["issues"].append(str(error))
        return report
    report.update(status="READY_FOR_PREPARATION", coverage=coverage,
                  common={"start": str(joint[0]), "end": str(joint[-1]), "dates": len(joint)},
                  benchmark_ready=not action_gaps, benchmark_blockers=action_gaps)
    return report


def _daily_options(path: Path, *, batch_size: int = 65536):
    """Stream a date-sorted chain, including days crossing Arrow batch boundaries."""
    pq = _parquet()
    pending = None
    previous = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(OPTION_COLUMNS)):
        data = {name: batch.column(name).to_numpy(zero_copy_only=False) for name in OPTION_COLUMNS}
        data["date"] = data["date"].astype("datetime64[D]")
        if np.isnat(data["date"]).any() or np.any(data["date"][1:] < data["date"][:-1]):
            raise InputError("option parquet must be ordered by quote date")
        if previous is not None and data["date"][0] < previous:
            raise InputError("option parquet is not chronological across batches")
        previous = data["date"][-1]
        if pending is not None:
            data = {name: np.concatenate((pending[name], values)) for name, values in data.items()}
        boundaries = np.r_[0, np.flatnonzero(data["date"][1:] != data["date"][:-1]) + 1, len(data["date"])]
        for begin, end in zip(boundaries[:-2], boundaries[1:-1]):
            yield data["date"][begin], {name: values[begin:end] for name, values in data.items()}
        pending = {name: values[boundaries[-2]:] for name, values in data.items()}
    if pending is not None:
        yield pending["date"][0], pending


def clean_option_day(records: dict[str, np.ndarray], spot: float, symbol: str) -> tuple[dict, dict]:
    """Explicit disjoint filter counts; ignore vendor ITM flags and recompute K/S."""
    count = len(records["date"])
    keep = np.ones(count, dtype=bool)
    counts = {"input_rows": count}
    if len(np.unique(records["contract_id"])) != count:
        raise InputError(f"{symbol}: duplicate contract/date observations")
    if not np.all(records["symbol"] == symbol):
        raise InputError(f"{symbol}: conflicting option symbols")
    if not np.isfinite(spot) or spot <= 0:
        raise InputError("moneyness needs a positive finite unadjusted spot")
    numeric = {key: np.asarray(records[key], dtype=float)
               for key in ("strike", "bid", "ask", "volume", "implied_volatility", "vega")}
    expiry = np.asarray(records["expiration"], dtype="datetime64[D]")
    quoted = np.asarray(records["date"], dtype="datetime64[D]")
    maturity = (expiry - quoted).astype(float) / 365
    m = numeric["strike"] / spot
    filters = {
        "invalid_contract_terms": (~np.isnat(expiry) & (m > 0)
                                   & np.isin(records["type"], ("call", "put"))),
        "nonfinite": np.logical_and.reduce([np.isfinite(values) for values in numeric.values()]),
        "invalid_bid_ask": ((numeric["bid"] >= 0) & (numeric["ask"] > 0)
                            & (numeric["ask"] >= numeric["bid"])),
        "no_traded_volume": numeric["volume"] > 0,
        "nonpositive_iv_or_vega": (numeric["implied_volatility"] > 0) & (numeric["vega"] > 0),
        "expired": maturity > 0,
        "in_the_money": (((records["type"] == "call") & (m >= 1))
                         | ((records["type"] == "put") & (m < 1))),
    }
    for reason, valid in filters.items():
        counts[reason] = int(np.count_nonzero(keep & ~valid))
        keep &= valid
    counts["retained"] = int(keep.sum())
    return {"moneyness": m[keep], "maturities": maturity[keep],
            "values": numeric["implied_volatility"][keep], "vegas": numeric["vega"][keep]}, counts


def prepare_etf_data(raw_directory: str | Path, output_directory: str | Path, *,
                     bandwidth_m: float = 0.01, bandwidth_tau: float = 0.01) -> dict:
    """Smooth each asset/day independently, then join dates without future filling."""
    raw, output = Path(raw_directory), Path(output_directory)
    readiness = check_data_readiness(raw)
    if readiness["status"] != "READY_FOR_PREPARATION":
        raise InputError("DATA_BLOCKED: " + "; ".join(readiness["issues"]))
    sources = {symbol: _underlying(raw / f"{symbol}_underlying.parquet", symbol) for symbol in SYMBOLS}
    dates = None
    full_dates = np.array([], dtype="datetime64[D]")
    for symbol, source in sources.items():
        own_dates = np.intersect1d(source["date"], _option_dates(raw / f"{symbol}_options.parquet"))
        dates = own_dates if dates is None else np.intersect1d(dates, own_dates)
        full_dates = np.union1d(full_dates, source["date"])
    n = len(dates)
    iv = np.full((n, len(SYMBOLS), *PAPER_GRID.shape), np.nan)
    arrays = {"dates": dates.astype(str), "symbols": np.array(SYMBOLS), "iv": iv,
              "moneyness": np.array(PAPER_GRID.moneyness), "maturities": np.array(PAPER_GRID.maturities)}
    for name, source_key in (("open", "open"), ("close", "close"),
                             ("dividends", "dividend_amount"), ("splits", "split_coefficient")):
        arrays[name] = np.column_stack([sources[symbol][source_key][np.searchsorted(sources[symbol]["date"], dates)]
                                       for symbol in SYMBOLS])
    locations = {date: index for index, date in enumerate(dates)}
    cleaning = {}
    missing = {}
    for asset, symbol in enumerate(SYMBOLS):
        totals = Counter()
        missing[symbol] = []
        print(f"Preparing {symbol} implied-volatility surfaces ...", flush=True)
        for date, records in _daily_options(raw / f"{symbol}_options.parquet"):
            if date not in locations:
                continue
            index = locations[date]
            clean, counts = clean_option_day(records, float(arrays["close"][index, asset]), symbol)
            totals.update(counts)
            if not counts["retained"]:
                missing[symbol].append(str(date))
                continue
            iv[index, asset] = smooth_surface(**clean, bandwidth_m=bandwidth_m, bandwidth_tau=bandwidth_tau)
        cleaning[symbol] = dict(totals)
    arrays["valid_surface"] = np.isfinite(iv).all(axis=(-2, -1)) & (iv > 0).all(axis=(-2, -1))
    indices = np.searchsorted(full_dates, dates)
    arrays["consecutive"] = np.r_[False, np.diff(indices) == 1]
    calendar_gaps = [{"previous": str(dates[i - 1]), "resumed": str(dates[i]),
                      "missing_dates": full_dates[indices[i - 1] + 1:indices[i]].astype(str).tolist()}
                     for i in range(1, n) if not arrays["consecutive"][i]]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "surfaces.npz"
    np.savez_compressed(path, **arrays)
    manifest = {
        "schema_version": 1, "data_kind": "historical_etf_options_data_v1", "symbols": list(SYMBOLS),
        "label": "Real historical constituent prices and option surfaces; model-valued non-traded basket option.",
        "repository": REPOSITORY, "release": RELEASE, "hash_source_commit": SOURCE_COMMIT,
        "source_file_hashes": {name: entry["sha256"] for name, entry in readiness["files"].items()},
        "prepared_sha256": sha256(path), "common": readiness["common"], "split": SPLIT,
        "grid": {"moneyness": list(PAPER_GRID.moneyness), "maturities_years": list(PAPER_GRID.maturities),
                 "flattening": "asset-major then maturity-major (all moneyness points per maturity)"},
        "smoothing": {"method": "vega-weighted Nadaraya-Watson Gaussian", "bandwidth_m": bandwidth_m,
                      "bandwidth_tau": bandwidth_tau, "kernel": "exp(-dm^2/(2*h_m)-dtau^2/(2*h_tau))",
                      "selection": "predeclared ETF adaptation; no calibration on validation or test data"},
        "cleaning": cleaning, "missing_surface_dates": missing,
        "nonconsecutive_common_dates": arrays["dates"][~arrays["consecutive"]][1:].tolist(),
        "common_date_gaps": calendar_gaps,
        "benchmark_ready": readiness["benchmark_ready"], "benchmark_blockers": readiness["benchmark_blockers"],
        "limitations": ["Original market-data vendor and IV/Greek estimation conventions are undocumented.",
                        "ETF option IVs are vendor estimates for American contracts; surfaces are model inputs, not executable quotes.",
                        "Dates are EOD labels, not verified historical publication or order-fill timestamps.",
                        "Underlying closes are unadjusted; missing dividends/splits remain NaN, never assumed zero/one.",
                        "No interest-rate curve or dividend forecast is supplied by these six files.",
                        "Extreme grid points can rely on distant observations; smoothing is not an arbitrage certificate."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def load_prepared(directory: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("data_kind") != "historical_etf_options_data_v1":
        raise InputError("prepared data must be the declared historical ETF dataset")
    if sha256(root / "surfaces.npz") != manifest.get("prepared_sha256"):
        raise InputError("prepared data checksum does not match its manifest")
    with np.load(root / "surfaces.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    required = {"dates", "symbols", "open", "close", "dividends", "splits", "iv",
                "valid_surface", "consecutive", "moneyness", "maturities"}
    if not required <= arrays.keys() or manifest.get("schema_version") != 1:
        raise InputError("prepared data schema is incomplete or unsupported")
    n = len(arrays["dates"])
    dates = np.asarray(arrays["dates"], dtype="datetime64[D]")
    if (n < 2 or tuple(arrays["symbols"]) != SYMBOLS or np.isnat(dates).any()
            or np.any(dates[1:] <= dates[:-1])
            or arrays["iv"].shape != (n, 3, *PAPER_GRID.shape)
            or any(arrays[key].shape != (n, 3) for key in ("open", "close", "dividends", "splits", "valid_surface"))
            or arrays["consecutive"].shape != (n,) or arrays["consecutive"].dtype != bool
            or arrays["valid_surface"].dtype != bool or arrays["consecutive"][0]
            or not np.array_equal(arrays["moneyness"], PAPER_GRID.moneyness)
            or not np.array_equal(arrays["maturities"], PAPER_GRID.maturities)):
        raise InputError("prepared arrays have inconsistent shapes or chronology")
    for key in ("open", "close"):
        if not np.isfinite(arrays[key]).all() or np.any(arrays[key] <= 0):
            raise InputError("prepared spot prices must be positive and finite")
    for key in ("dividends", "splits"):
        if np.isinf(arrays[key]).any() or np.any(arrays[key] < 0):
            raise InputError("prepared corporate actions must be nonnegative or explicitly missing")
    if np.any(arrays["splits"] == 0):
        raise InputError("prepared split coefficients cannot be zero")
    observed_valid = np.isfinite(arrays["iv"]).all(axis=(-2, -1)) & (arrays["iv"] > 0).all(axis=(-2, -1))
    if not np.array_equal(observed_valid, arrays["valid_surface"]):
        raise InputError("prepared surface validity flags disagree with the surface values")
    return arrays, manifest
