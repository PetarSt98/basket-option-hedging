"""Tiny artificial rows test data mechanics; they are not historical results."""

import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from basket_hedging.contracts import InputError
from basket_hedging import etf_data


def records(symbol="SPY", date="2017-12-28"):
    return {
        "date": np.array([date] * 2, dtype="datetime64[D]"),
        "expiration": np.array(["2018-01-28"] * 2, dtype="datetime64[D]"),
        "contract_id": np.array([f"{symbol}-P90", f"{symbol}-C110"]),
        "symbol": np.array([symbol] * 2), "type": np.array(["put", "call"]),
        "strike": np.array([90., 110.]), "bid": np.array([1., 1.]), "ask": np.array([2., 2.]),
        "volume": np.array([10., 5.]), "open_interest": np.array([20, 10]),
        "implied_volatility": np.array([.2, .2]), "vega": np.array([1., 2.]),
    }


def write_raw(root, monkeypatch, *, missing_day=False):
    """Tests explicitly replace expected hashes with their generated fixture hashes."""
    root.mkdir()
    dates = ["2017-12-28", "2017-12-29", "2018-01-02", "2018-01-03"]
    hashes = {}
    for symbol in etf_data.SYMBOLS:
        # Reverse order tests that underlying dates are sorted deterministically.
        data = {"date": dates[::-1], "symbol": [symbol] * 4, "open": [100.] * 4, "close": [100.] * 4,
                "dividend_amount": [None] * 4 if symbol == "IWM" else [0., .1, 0., 0.],
                "split_coefficient": [None] * 4 if symbol == "IWM" else [1.] * 4}
        path = root / f"{symbol}_underlying.parquet"
        pq.write_table(pa.table(data), path)
        hashes[path.name] = etf_data.sha256(path)
        days = [records(symbol, date) for date in dates if not (missing_day and symbol == "QQQ" and date == dates[1])]
        data = {key: np.concatenate([row[key] for row in days]) for key in days[0]}
        data["date"] = data["date"].astype("datetime64[ns]")
        data["expiration"] = data["expiration"].astype("datetime64[ns]")
        path = root / f"{symbol}_options.parquet"
        pq.write_table(pa.table(data), path)
        hashes[path.name] = etf_data.sha256(path)
    monkeypatch.setattr(etf_data, "HASHES", hashes)
    return root


def test_filter_reasons_disjoint_and_moneyness_uses_raw_spot():
    data = records()
    cleaned, counts = etf_data.clean_option_day(data, 100., "SPY")
    np.testing.assert_allclose(cleaned["moneyness"], [.9, 1.1])
    np.testing.assert_allclose(cleaned["maturities"], [31 / 365, 31 / 365])
    assert counts["retained"] == 2
    data["volume"][0] = 0
    data["bid"][1] = 3
    _, counts = etf_data.clean_option_day(data, 100., "SPY")
    assert counts["retained"] == 0
    assert counts["no_traded_volume"] == 1
    assert counts["invalid_bid_ask"] == 1
    assert sum(v for k, v in counts.items() if k not in {"input_rows", "retained"}) == 2


@pytest.mark.parametrize("field,value,reason", [
    ("implied_volatility", np.nan, "nonfinite"),
    ("vega", 0., "nonpositive_iv_or_vega"),
    ("strike", 120., "in_the_money"),
])
def test_bad_observations_are_visible_exclusions(field, value, reason):
    data = records()
    data[field][0] = value
    _, counts = etf_data.clean_option_day(data, 100., "SPY")
    assert counts[reason] == 1
    assert counts["retained"] == 1


def test_conflicting_contract_duplicate_rejected():
    data = records()
    data["contract_id"][1] = data["contract_id"][0]
    with pytest.raises(InputError, match="duplicate"):
        etf_data.clean_option_day(data, 100., "SPY")


def test_arrow_batch_boundaries_do_not_split_daily_surfaces(tmp_path, monkeypatch):
    raw = write_raw(tmp_path / "raw", monkeypatch)
    days = list(etf_data._daily_options(raw / "SPY_options.parquet", batch_size=1))
    assert [str(date) for date, _ in days] == ["2017-12-28", "2017-12-29", "2018-01-02", "2018-01-03"]
    assert [len(day["contract_id"]) for _, day in days] == [2, 2, 2, 2]


def test_missing_files_and_hash_mismatches_stop_without_download(tmp_path):
    report = etf_data.check_data_readiness(tmp_path)
    assert report["status"] == "DATA_BLOCKED"
    assert len(report["issues"]) == 6
    (tmp_path / "SPY_options.parquet").write_bytes(b"wrong bytes")
    report = etf_data.check_data_readiness(tmp_path)
    assert "SHA-256 mismatch: SPY_options.parquet" in report["issues"]


def test_prepare_preserves_joint_dates_missing_actions_and_session_gaps(tmp_path, monkeypatch):
    raw = write_raw(tmp_path / "raw", monkeypatch, missing_day=True)
    report = etf_data.check_data_readiness(raw)
    assert report["common"] == {"start": "2017-12-28", "end": "2018-01-03", "dates": 3}
    assert not report["benchmark_ready"]
    output = tmp_path / "prepared"
    manifest = etf_data.prepare_etf_data(raw, output)
    data, loaded = etf_data.load_prepared(output)
    assert loaded == manifest
    assert data["dates"].tolist() == ["2017-12-28", "2018-01-02", "2018-01-03"]
    assert data["consecutive"].tolist() == [False, False, True]
    assert np.isnan(data["dividends"][:, 2]).all()
    assert np.isnan(data["splits"][:, 2]).all()
    assert data["valid_surface"].all()
    np.testing.assert_allclose(data["iv"], .2, atol=1e-14)
    assert manifest["benchmark_blockers"] == ["IWM: missing dividend_amount", "IWM: missing split_coefficient"]
    assert manifest["nonconsecutive_common_dates"] == ["2018-01-02"]
    assert manifest["common_date_gaps"] == [{"previous": "2017-12-28", "resumed": "2018-01-02",
                                             "missing_dates": ["2017-12-29"]}]


def test_preprocessing_future_mutation_cannot_change_previous_surfaces(tmp_path, monkeypatch):
    raw = write_raw(tmp_path / "raw", monkeypatch)
    first = tmp_path / "first"
    etf_data.prepare_etf_data(raw, first)
    earlier, _ = etf_data.load_prepared(first)
    path = raw / "SPY_options.parquet"
    table = pq.read_table(path).to_pydict()
    table["implied_volatility"][-2:] = [.9, .9]
    pq.write_table(pa.table(table), path)
    etf_data.HASHES[path.name] = etf_data.sha256(path)
    second = tmp_path / "second"
    etf_data.prepare_etf_data(raw, second)
    later, _ = etf_data.load_prepared(second)
    np.testing.assert_array_equal(earlier["iv"][:-1], later["iv"][:-1])
    np.testing.assert_allclose(later["iv"][-1, 0], .9)


def test_prepared_checksum_and_kind_rejected(tmp_path, monkeypatch):
    raw = write_raw(tmp_path / "raw", monkeypatch)
    output = tmp_path / "prepared"
    etf_data.prepare_etf_data(raw, output)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["prepared_sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(InputError, match="checksum"):
        etf_data.load_prepared(output)
    manifest["data_kind"] = "synthetic"
    path.write_text(json.dumps(manifest))
    with pytest.raises(InputError, match="historical"):
        etf_data.load_prepared(output)


def test_expired_contract_is_filtered_without_replacing_fixed_strike():
    data = records()
    data["expiration"][0] = data["date"][0]
    _, counts = etf_data.clean_option_day(data, 100., "SPY")
    assert counts["expired"] == 1
    assert data["strike"][0] == 90


def test_zero_ask_is_not_a_usable_surface_quote_even_with_positive_volume():
    data = records()
    data["bid"][0] = data["ask"][0] = 0
    _, counts = etf_data.clean_option_day(data, 100., "SPY")
    assert counts["invalid_bid_ask"] == 1
    assert counts["retained"] == 1
