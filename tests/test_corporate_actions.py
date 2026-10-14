import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import pytest

from basket_hedging.contracts import InputError
from basket_hedging.corporate_actions import (
    crosscheck_yahoo, load_iwm_distributions, parse_ishares_distributions,
    write_iwm_distributions,
)


def _html(ex_dates=(20101222, 20110324, 20250916, 20251216),
          pay_dates=(20101229, 20110330, 20250919, 20251219),
          amounts=("0.25", "0.168933", "0.676797", "0.842454")):
    columns = [{"name": name, "value": list(values)} for name, values in (
        ("exDate", ex_dates), ("payableDate", pay_dates),
        ("totalDistribution", amounts), ("incomeAmount", ("0.1",) * len(ex_dates)))]
    payload = escape(json.dumps({"distributionTableData": columns}), quote=True)
    return f'<walrus-render-on-client componentprops="{payload}"></walrus-render-on-client>'


def test_complete_embedded_table_uses_ex_date_total_cash_and_preserves_precision():
    rows = parse_ishares_distributions(_html())
    # The total, not the distinct income component, is the actual cash action.
    assert rows[1] == {"ex_date": "2011-03-24", "payable_date": "2011-03-30",
                       "cash_per_share": "0.168933"}
    assert len(rows) == 4
    assert parse_ishares_distributions(_html() + _html()) == rows


@pytest.mark.parametrize("html,match", [
    (_html(ex_dates=(20110324, 20110324), pay_dates=(20110330, 20110330),
           amounts=("0.168933", "0.168933")), "duplicate"),
    (_html(amounts=("0.25", "NaN", "0.676797", "0.842454")), "finite"),
    (_html(pay_dates=(20101229, 20110323, 20250919, 20251219)), "precedes"),
    (_html(amounts=("0.25",)), "unequal"),
    ("<html>No data</html>", "missing"),
    (_html() + _html(amounts=("0.25", "0.2", "0.676797", "0.842454")), "conflicting"),
])
def test_malformed_or_duplicate_source_records_are_rejected(html, match):
    with pytest.raises(InputError, match=match):
        parse_ishares_distributions(html)


def test_normalization_limits_ex_dates_and_loader_checks_hash_and_coverage(tmp_path):
    report = write_iwm_distributions(tmp_path, _html().encode(), "2026-10-04T10:00:00+00:00")
    rows, reloaded = load_iwm_distributions(tmp_path)
    assert reloaded == report
    assert [row["ex_date"] for row in rows] == ["2011-03-24", "2025-09-16"]
    assert report["coverage_start"] == "2011-03-23"
    assert report["coverage_end"] == "2025-12-12"
    with (tmp_path / "distributions.csv").open("a") as stream:
        stream.write("2011-03-24,2011-03-30,0.168933\n")
    with pytest.raises(InputError, match="SHA-256"):
        load_iwm_distributions(tmp_path)


def test_incomplete_source_history_cannot_imply_no_distribution(tmp_path):
    html = _html(ex_dates=(20110324, 20250916), pay_dates=(20110330, 20250919),
                 amounts=("0.168933", "0.676797"))
    with pytest.raises(InputError, match="boundaries"):
        write_iwm_distributions(tmp_path, html.encode(), "2026-10-04T10:00:00+00:00")


def test_crosscheck_uses_event_date_not_monthly_bar_key_and_never_overrides():
    rows = [{"ex_date": "2011-03-24", "payable_date": "2011-03-30", "cash_per_share": "0.168933"}]
    ex_time = int(datetime(2011, 3, 24, 13, 30, tzinfo=timezone.utc).timestamp())
    raw = json.dumps({"chart": {"result": [{"meta": {"symbol": "IWM"}, "events": {
        "dividends": {"monthly-bar-time-not-ex-date": {"date": ex_time, "amount": 0.169}}}}]}}).encode()
    result = crosscheck_yahoo(rows, raw)
    assert result["matching_date_count"] == 1
    assert result["missing_from_yahoo"] == result["missing_from_ishares"] == []
    assert result["amount_differences"][0]["yahoo_minus_ishares"] == "0.000067"
    assert rows[0]["cash_per_share"] == "0.168933"


def test_bundled_verified_history_contains_all_59_events_and_no_duplicates():
    directory = Path(__file__).resolve().parents[1] / "data/iwm_corporate_actions"
    rows, report = load_iwm_distributions(directory)
    assert len(rows) == len({row["ex_date"] for row in rows}) == 59
    assert rows[0] == {"ex_date": "2011-03-24", "payable_date": "2011-03-30",
                       "cash_per_share": "0.168933"}
    assert rows[-1] == {"ex_date": "2025-09-16", "payable_date": "2025-09-19",
                        "cash_per_share": "0.676797"}
    assert report["source_first_ex_date"] == "2000-06-21"
    assert report["source_last_ex_date"] > "2025-12-12"
    assert report["yahoo_crosscheck"]["matching_date_count"] == 59
    assert report["yahoo_crosscheck"]["missing_from_yahoo"] == []
    assert report["yahoo_crosscheck"]["missing_from_ishares"] == []
    assert report["yahoo_crosscheck"]["maximum_absolute_difference"] == 0.000487
    # Total independently summed from the publisher's raw Decimal table columns.
    with (directory / "distributions.csv").open(newline="", encoding="utf-8") as stream:
        assert sum(float(row["cash_per_share"]) for row in csv.DictReader(stream)) == pytest.approx(28.779858)
