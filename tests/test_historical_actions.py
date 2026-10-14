"""Synthetic action fixtures test a data-only rebuild and unchanged engine timing."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import numpy as np
import pytest

from basket_hedging.contracts import InputError, Instrument, Opening
from basket_hedging.engine import Engine
from basket_hedging.etf_data import load_prepared, sha256
from basket_hedging.historical_actions import supplement_prepared, verified_training_parent
from basket_hedging.market import Event, Quote, Rules, ValuationPacket
from basket_hedging.replay import write_json
from basket_hedging.risk import UnnettedLiabilityStressMargin
from basket_hedging.surfaces import PAPER_GRID


def original(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    prices = np.array([[100., 90., 80.], [101., 91., 81.], [102., 92., 82.]])
    arrays = dict(dates=np.array(["2020-03-20", "2020-03-23", "2020-03-24"]),
        symbols=np.array(["SPY", "QQQ", "IWM"]), open=prices, close=prices.copy(),
        dividends=np.array([[0., .2, np.nan], [.5, 0., np.nan], [0., 0., np.nan]]),
        splits=np.array([[1., 1., np.nan]] * 3),
        iv=np.full((3, 3, *PAPER_GRID.shape), .2), valid_surface=np.ones((3, 3), dtype=bool),
        consecutive=np.array([False, True, True]),
        moneyness=np.array(PAPER_GRID.moneyness), maturities=np.array(PAPER_GRID.maturities))
    np.savez_compressed(root / "surfaces.npz", **arrays)
    manifest = dict(schema_version=1, data_kind="historical_etf_options_data_v1",
                    prepared_sha256=sha256(root / "surfaces.npz"), split={"train_end": "2017-12-29"},
                    common_date_gaps=[], source_file_hashes={"option_fixture": "unchanged"})
    write_json(root / "manifest.json", manifest)
    return root, arrays, manifest


def actions(tmp_path, monkeypatch):
    from basket_hedging import corporate_actions
    root = tmp_path / "actions"
    root.mkdir()
    rows = [dict(ex_date="2020-03-23", payable_date="2020-03-26", cash_per_share="0.500000")]
    manifest = dict(coverage_start="2011-03-23", coverage_end="2025-12-12")
    write_json(root / "distributions_manifest.json", manifest)
    write_json(root / "split_verification.json", dict(manifest, verified_no_splits=True,
        split_factor=1., in_period_events=[], sources=[{"url": "https://one.example/history"},
                                                     {"url": "https://two.example/history"}]))
    monkeypatch.setattr(corporate_actions, "load_iwm_distributions", lambda path: (rows, manifest))
    return root, rows, manifest


def test_rebuild_changes_only_two_iwm_columns_and_preserves_training_parent(tmp_path, monkeypatch):
    source, before, parent = original(tmp_path)
    supplement, _, _ = actions(tmp_path, monkeypatch)
    output = tmp_path / "supplemented"
    original_bytes = (source / "surfaces.npz").read_bytes()
    # Stand-in raw quote file is never rewritten by the prepared-data integration.
    quote_file = tmp_path / "original_option_quotes"
    quote_file.write_bytes(b"contract,date,bid,ask\nIWM-C,2020-03-20,1.0,1.2\n")
    quote_hash = sha256(quote_file)
    supplement_prepared(source, supplement, output)
    after, manifest = load_prepared(output)
    assert (source / "surfaces.npz").read_bytes() == original_bytes
    assert sha256(quote_file) == quote_hash
    for key in before.keys() - {"dividends", "splits"}:
        np.testing.assert_array_equal(before[key], after[key])
    np.testing.assert_array_equal(after["dividends"][:, :2], [[0., .2], [.5, 0.], [0., 0.]])
    np.testing.assert_array_equal(after["dividends"][:, 2], [0., .5, 0.])
    np.testing.assert_array_equal(after["splits"], np.ones((3, 3)))
    assert verified_training_parent(output, after, manifest) == parent["prepared_sha256"]
    # An identical rebuild is idempotent; distributions are not accumulated.
    supplement_prepared(source, supplement, output)
    repeated, _ = load_prepared(output)
    np.testing.assert_array_equal(repeated["dividends"], after["dividends"])


@pytest.mark.parametrize("tamper", ["price", "calendar", "actions", "source"])
def test_checkpoint_ancestry_rejects_changed_inputs(tmp_path, monkeypatch, tamper):
    source, _, _ = original(tmp_path)
    supplement, _, _ = actions(tmp_path, monkeypatch)
    output = tmp_path / "supplemented"
    supplement_prepared(source, supplement, output)
    arrays, manifest = load_prepared(output)
    if tamper == "price":
        arrays["close"][0, 0] += 1
    elif tamper == "calendar":
        manifest["common_date_gaps"] = [{"missing_dates": ["2020-03-21"]}]
    elif tamper == "actions":
        arrays["dividends"][1, 2] += 1
    else:
        manifest["source_file_hashes"]["option_fixture"] = "changed"
    with pytest.raises(InputError):
        verified_training_parent(output, arrays, manifest)


def test_incomplete_verification_cannot_fill_unknowns_with_zeros(tmp_path, monkeypatch):
    source, _, _ = original(tmp_path)
    supplement, _, manifest = actions(tmp_path, monkeypatch)
    manifest["coverage_start"] = "2020-03-21"
    with pytest.raises(InputError, match="coverage"):
        supplement_prepared(source, supplement, tmp_path / "new")
    manifest["coverage_start"] = "2011-03-23"
    path = supplement / "split_verification.json"
    splits = json.loads(path.read_text())
    splits["sources"] = splits["sources"][:1]
    write_json(path, splits)
    with pytest.raises(InputError, match="two-source"):
        supplement_prepared(source, supplement, tmp_path / "new")
    with pytest.raises(InputError, match="overwrite"):
        supplement_prepared(source, supplement, source)


def test_buy_at_ex_date_open_does_not_earn_pretrade_dividend_or_pay_again():
    start = datetime(2020, 3, 20, 20, tzinfo=timezone.utc)
    ex_date = datetime(2020, 3, 23, 13, 30, tzinfo=timezone.utc)
    exit_time = ex_date + timedelta(hours=1)
    end = datetime(2020, 3, 26, 13, 30, tzinfo=timezone.utc)  # reported payable date
    instruments = {"IWM": Instrument("IWM", "stock")}

    def packet(time):
        return ValuationPacket(time, {"IWM": 100.}, {"unchanged": {"IWM": 100.}},
                               observed_at={"IWM": time})

    def quote(time):
        return (Quote(str(time), "IWM", time, time, 100., 100., 10., 10.),)

    events = [Event("decision", start, 0, packet(start), decision=True,
                    earliest_execution=ex_date, order_deadline=exit_time),
              Event("ex", ex_date, 0, packet(ex_date), quotes=quote(ex_date), dividends={"IWM": .5},
                    decision=True, earliest_execution=exit_time, order_deadline=end),
              Event("exit", exit_time, 0, packet(exit_time), quotes=quote(exit_time)),
              Event("payable", end, 0, packet(end))]
    result = Engine(instruments, Opening("IMPORT", cash=1000., opening_equity=1000.),
                    Rules(end, ("IWM",), 0), UnnettedLiabilityStressMargin()).run(
        events, lambda view: {"IWM": 4. if view.time == start else 0.})
    assert result.status == "COMPLETED"
    assert result.ledger.cash == 1000.
    distributions = [post for post in result.ledger.postings if post.kind == "dividend"]
    assert len(distributions) == 1
    assert distributions[0].time == ex_date
    assert distributions[0].quantity == distributions[0].cash_change == 0
