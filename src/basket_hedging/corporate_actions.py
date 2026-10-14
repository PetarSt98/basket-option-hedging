"""Reproducible IWM distribution supplement; never a source of market prices."""

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

from .contracts import InputError


ISHARES_URL = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
COVERAGE_START = "2011-03-23"
COVERAGE_END = "2025-12-12"
FIELDS = ("ex_date", "payable_date", "cash_per_share")
YAHOO_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/IWM"
             "?period1=1300838400&period2=1765584000&interval=1mo&events=div%2Csplits")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cash(value) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise InputError("invalid distribution amount") from error
    if not number.is_finite() or number <= 0:
        raise InputError("distribution amount must be finite and positive")
    return number


def _validate_rows(rows: list[dict[str, str]]) -> None:
    seen = set()
    for row in rows:
        try:
            ex_date = date.fromisoformat(row["ex_date"])
            payable = date.fromisoformat(row["payable_date"])
        except (KeyError, TypeError, ValueError) as error:
            raise InputError("invalid ex-date or payable date") from error
        if ex_date in seen:
            raise InputError(f"duplicate IWM distribution ex-date: {ex_date}")
        seen.add(ex_date)
        if payable < ex_date:
            raise InputError("distribution payable date precedes ex-date")
        _cash(row["cash_per_share"])
    if not rows:
        raise InputError("empty IWM distribution history")


def parse_ishares_distributions(html: str) -> list[dict[str, str]]:
    """Read the complete embedded table, including rows hidden by pagination."""
    tables = []

    class TableParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag != "walrus-render-on-client":
                return
            value = dict(attrs).get("componentprops", "")
            if '"distributionTableData"' in value:
                try:
                    payload = json.loads(value)
                except ValueError as error:
                    raise InputError("invalid embedded distribution JSON") from error
                if "distributionTableData" in payload:
                    tables.append(payload["distributionTableData"])

    TableParser().feed(html)
    if not tables or any(table != tables[0] for table in tables[1:]):
        raise InputError("missing or conflicting official distribution tables")
    columns = {column["name"]: column["value"] for column in tables[0]}
    if len(columns) != len(tables[0]):
        raise InputError("duplicate official distribution columns")
    names = ("exDate", "payableDate", "totalDistribution")
    if any(name not in columns for name in names):
        raise InputError("missing official distribution columns")
    if len({len(columns[name]) for name in names}) != 1:
        raise InputError("official distribution columns have unequal lengths")
    rows = []
    for ex_date, payable, total in zip(*(columns[name] for name in names)):
        try:
            row = {"ex_date": datetime.strptime(str(ex_date), "%Y%m%d").date().isoformat(),
                   "payable_date": datetime.strptime(str(payable), "%Y%m%d").date().isoformat()}
        except (TypeError, ValueError) as error:
            raise InputError("invalid official distribution date") from error
        row["cash_per_share"] = format(_cash(total), "f")
        rows.append(row)
    _validate_rows(rows)
    return sorted(rows, key=lambda row: row["ex_date"])


def crosscheck_yahoo(rows: list[dict[str, str]], raw: bytes) -> dict:
    """Compare actions only; official amounts remain authoritative."""
    result = json.loads(raw)["chart"]["result"][0]
    if result["meta"]["symbol"] != "IWM":
        raise InputError("Yahoo cross-check returned the wrong symbol")
    actions = result.get("events", {}).get("dividends", {})
    yahoo = {}
    for event in actions.values():
        # Monthly chart keys identify bars, not ex-dates; use event.date itself.
        ex_date = datetime.fromtimestamp(event["date"], timezone.utc).date().isoformat()
        if COVERAGE_START <= ex_date <= COVERAGE_END:
            if ex_date in yahoo:
                raise InputError("duplicate Yahoo dividend ex-date")
            yahoo[ex_date] = _cash(event["amount"])
    official = {row["ex_date"]: _cash(row["cash_per_share"]) for row in rows}
    missing_yahoo = sorted(official.keys() - yahoo.keys())
    missing_official = sorted(yahoo.keys() - official.keys())
    differences = [{"ex_date": day, "ishares_cash_per_share": str(official[day]),
                    "yahoo_cash_per_share": str(yahoo[day]),
                    "yahoo_minus_ishares": str(yahoo[day] - official[day])}
                   for day in sorted(official.keys() & yahoo.keys())
                   if yahoo[day] != official[day]]
    return {"source_url": YAHOO_URL, "raw_sha256": _hash(raw),
            "matching_date_count": len(official.keys() & yahoo.keys()),
            "missing_from_yahoo": missing_yahoo, "missing_from_ishares": missing_official,
            "amount_differences": differences,
            "maximum_absolute_difference": float(max(
                (abs(Decimal(row["yahoo_minus_ishares"])) for row in differences), default=Decimal(0))),
            "interpretation": "Yahoo values are rounded to 0.001 USD; retain official iShares values."}


def write_iwm_distributions(output: str | Path, raw_html: bytes, retrieved_at: str,
                            *, yahoo_raw: bytes | None = None) -> dict:
    """Normalize and document a full-history snapshot spanning both boundaries."""
    all_rows = parse_ishares_distributions(raw_html.decode("utf-8"))
    if not all_rows[0]["ex_date"] < COVERAGE_START or not all_rows[-1]["ex_date"] > COVERAGE_END:
        raise InputError("official history does not span both benchmark boundaries")
    rows = [row for row in all_rows if COVERAGE_START <= row["ex_date"] <= COVERAGE_END]
    _validate_rows(rows)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    normalized = text.getvalue().encode("utf-8")
    report = {"symbol": "IWM", "currency": "USD", "source_url": ISHARES_URL,
              "retrieved_at": retrieved_at, "raw_sha256": _hash(raw_html),
              "distributions_sha256": _hash(normalized),
              "coverage_start": COVERAGE_START, "coverage_end": COVERAGE_END,
              "distribution_count": len(rows), "source_distribution_count": len(all_rows),
              "source_first_ex_date": all_rows[0]["ex_date"],
              "source_last_ex_date": all_rows[-1]["ex_date"],
              "first_ex_date": rows[0]["ex_date"], "last_ex_date": rows[-1]["ex_date"],
              "count_by_year": dict(sorted(Counter(row["ex_date"][:4] for row in rows).items())),
              "parsing_assumptions": [
                  "Parse complete distributionTableData from HTML-decoded componentprops JSON, not the first visible page.",
                  "Raw exDate/payableDate integers are calendar YYYYMMDD; filter inclusively on ex-date.",
                  "Use totalDistribution (all cash components), not incomeAmount; USD per ETF share as published.",
                  "Preserve decimal strings without rounding or price/split adjustment; reject duplicate ex-dates.",
                  "The full publisher history spans both boundaries; omitted days have no published distribution.",
                  "Payable dates are retained; the fixed benchmark books cash on ex-date using its pre-existing cash-timing proxy.",
                  "Future rows are excluded. Distribution amounts are accounting inputs released on ex-date, never strategy features."
              ]}
    if yahoo_raw is not None:
        report["yahoo_crosscheck"] = crosscheck_yahoo(rows, yahoo_raw)
        if report["yahoo_crosscheck"]["missing_from_ishares"]:
            raise InputError("Yahoo reports ex-dates absent from official history; investigate before merging")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "distributions.csv").write_bytes(normalized)
    (output / "distributions_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def load_iwm_distributions(directory: str | Path) -> tuple[list[dict[str, str]], dict]:
    directory = Path(directory)
    raw = (directory / "distributions.csv").read_bytes()
    report = json.loads((directory / "distributions_manifest.json").read_text(encoding="utf-8"))
    if report.get("distributions_sha256") != _hash(raw):
        raise InputError("IWM supplement SHA-256 mismatch")
    if (report.get("symbol"), report.get("currency"), report.get("source_url")) != ("IWM", "USD", ISHARES_URL):
        raise InputError("unexpected IWM supplement provenance")
    if (report.get("coverage_start"), report.get("coverage_end")) != (COVERAGE_START, COVERAGE_END):
        raise InputError("IWM supplement coverage does not match the verified benchmark period")
    if not (report.get("source_first_ex_date", "9999") < COVERAGE_START
            and report.get("source_last_ex_date", "0000") > COVERAGE_END):
        raise InputError("IWM source history does not span the benchmark")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise InputError("unexpected IWM supplement columns")
    rows = list(reader)
    _validate_rows(rows)
    if len(rows) != report.get("distribution_count") or rows != sorted(rows, key=lambda row: row["ex_date"]):
        raise InputError("IWM supplement count/order mismatch")
    if any(not COVERAGE_START <= row["ex_date"] <= COVERAGE_END for row in rows):
        raise InputError("IWM supplement contains ex-dates outside verified coverage")
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/iwm_corporate_actions"))
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/basket-hedging/corporate-actions")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.now(timezone.utc).isoformat()
    with urlopen(Request(ISHARES_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
        raw = response.read()
    (args.cache / f"iwm-ishares-{_hash(raw)}.html").write_bytes(raw)
    yahoo_raw = None
    try:
        with urlopen(Request(YAHOO_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
            yahoo_raw = response.read()
        (args.cache / f"iwm-yahoo-{_hash(yahoo_raw)}.json").write_bytes(yahoo_raw)
    except OSError as error:
        print(f"Optional Yahoo cross-check unavailable: {error}")
    report = write_iwm_distributions(args.output, raw, retrieved, yahoo_raw=yahoo_raw)
    print(json.dumps({key: report[key] for key in ("distribution_count", "first_ex_date", "last_ex_date", "distributions_sha256")}, indent=2))


if __name__ == "__main__":
    main()
