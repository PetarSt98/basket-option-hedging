"""Reports preserve independently checked balances, failures and observation times."""

import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from basket_hedging.contracts import Instrument, Opening, ReconciliationError
from basket_hedging.deterministic import deterministic_case, option_accounting_case
from basket_hedging.engine import Engine
from basket_hedging.market import Event, Quote, Rules, ValuationPacket
from basket_hedging.policies import ScheduledDeltaPolicy, UnhedgedPolicy
from basket_hedging.replay import load_case, write_case
from basket_hedging.reporting import write_reports
from basket_hedging.risk import UnnettedLiabilityStressMargin


def run_case(components, policy):
    instruments, opening, rules, events = components
    return Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()).run(events, policy)


def read_summary(comparison, policy):
    outcome = next(item for item in comparison["outcomes"] if item["policy"] == policy)
    with open(outcome["summary_path"], encoding="utf-8") as source:
        return json.load(source)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def test_completed_reports_reconcile_hand_derived_balances_and_csvs(tmp_path):
    components = deterministic_case()
    path = write_case(tmp_path / "case.json", *components, metadata={"source": "hand-derived"})
    results = [run_case(components, UnhedgedPolicy()), run_case(components, ScheduledDeltaPolicy())]
    comparison = write_reports(results, tmp_path / "reports", case_path=path, case=load_case(path))
    assert [outcome["status"] for outcome in comparison["outcomes"]] == ["COMPLETED", "COMPLETED"]
    assert [outcome["terminal_pnl"] for outcome in comparison["outcomes"]] == pytest.approx([-200, -371])
    assert [outcome["premium_retention_ratio"] for outcome in comparison["outcomes"]] == pytest.approx([-1, -1.855])
    summary = read_summary(comparison, "delta")
    assert summary["accepted_hedge_fill_count"] == 9
    assert read_summary(comparison, "unhedged")["accepted_hedge_fill_count"] == 0
    # Both runs originate and settle once; neither operation is a hedge fill.
    for result in results:
        assert sum(post.kind == "origination" for post in result.ledger.postings) == 1
        assert sum(post.kind == "settlement" for post in result.ledger.postings) == 1
    assert summary["pnl_baseline"] == 10000
    assert summary["current"]["equity"] == pytest.approx(9629)
    assert summary["current"]["pnl_components"]["liability_pnl"] == -200
    assert summary["current"]["pnl_components"]["hedge_pnl"] == pytest.approx(-150)
    assert summary["costs"]["execution_cost"] == pytest.approx(12)
    assert summary["costs"]["fees"] == 9
    assert summary["turnover"]["stock_currency_traded"] == pytest.approx(11970)
    assert summary["remaining_obligations"] == summary["residual_inventory"] == []
    assert summary["current"]["posted_margin_cash"] == summary["current"]["restricted_stock_short_proceeds"] == 0
    assert comparison["matched_checkpoint"]["event_id"] == "hand/5"
    assert comparison["matched_checkpoint"]["policies"]["delta"]["net_pnl"] == pytest.approx(-362)
    assert comparison["matched_checkpoint"]["phase"] == "pretrade"
    postings = read_csv(tmp_path / "reports" / "delta" / "postings.csv")
    assert 10000 + sum(float(row["cash_change"]) for row in postings) == pytest.approx(9629)
    assert -sum(float(row["cash_change"]) for row in postings if row["kind"] == "fee") == 9
    accounts = read_csv(tmp_path / "reports" / "delta" / "account.csv")
    assert float(accounts[-1]["equity"]) == pytest.approx(9629)
    assert json.loads(accounts[-1]["positions"]) == {}
    settlements = read_csv(tmp_path / "reports" / "delta" / "settlements.csv")
    assert float(settlements[0]["cashflow"]) == -400
    assert settlements[0]["funded"] == "True"
    assert settlements[0]["official_fixing_id"] == "official/BASKET"
    assert json.loads(settlements[0]["official_spots"]) == {"S1": 94, "S2": 76, "S3": 118}
    assert summary["provenance"]["files"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    audit = read_csv(tmp_path / "reports" / "delta" / "valuation_audit.csv")
    assert len(audit) == 6
    assert len({row["parameters_sha256"] for row in audit}) == 1
    assert len(summary["valuation_configurations"]) == 1
    assert json.loads((tmp_path / "reports" / "comparison.json").read_text()) == comparison


def test_stopped_strategies_remain_and_have_no_invented_terminal_metrics(tmp_path):
    results = [run_case(deterministic_case("gap"), policy) for policy in (UnhedgedPolicy(), ScheduledDeltaPolicy())]
    comparison = write_reports(results, tmp_path)
    assert len(comparison["outcomes"]) == 2
    for outcome in comparison["outcomes"]:
        assert outcome["status"] == "STOP_WITH_BREACH"
        assert outcome["terminal_pnl"] is None
        assert outcome["premium_retention_ratio"] is None
        assert outcome["marked_pnl"] == -400
        summary = read_summary(comparison, outcome["policy"])
        assert summary["terminal"] is None
        assert summary["max_observed_shortfall"] == 250
        assert summary["breaches"][0]["risk"]["margin_equity_shortfall"] == 200
        assert summary["breaches"][0]["risk"]["buffer_equity_shortfall"] == 50
        assert summary["remaining_obligations"][0]["instrument_id"] == "BASKET"
        assert summary["settlements"] == []


def test_incomplete_exit_is_kept_beside_completed_unhedged_outcome(tmp_path):
    results = [run_case(deterministic_case("missing_exit"), policy)
               for policy in (UnhedgedPolicy(), ScheduledDeltaPolicy())]
    comparison = write_reports(results, tmp_path)
    unhedged, delta = comparison["outcomes"]
    assert unhedged["status"] == "COMPLETED"
    assert delta["status"] == "INCOMPLETE_EXIT"
    assert delta["terminal_pnl"] is None
    assert delta["marked_pnl"] == pytest.approx(-362)
    summary = read_summary(comparison, "delta")
    assert summary["current"]["total_cash"] == 15698
    assert summary["current"]["restricted_stock_short_proceeds"] == pytest.approx(5904)
    assert [row["quantity"] for row in summary["residual_inventory"]] == [-20, -20, -20]
    assert sum(row["current_position_value"] for row in summary["residual_inventory"]) == -6060
    assert comparison["matched_checkpoint"]["policies"]["delta"]["net_pnl"] == pytest.approx(-362)


@pytest.mark.parametrize("exit_price,original_premium,expected_ratio", [(120, 10, 2), (80, 10, -2),
                                                                      (120, 0, None), (120, None, None)])
def test_import_premium_is_metadata_only_and_retention_is_not_clamped(tmp_path, exit_price, original_premium, expected_ratio):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    instruments = {"S": Instrument("S", "stock")}
    opening = Opening("IMPORT", cash=0, positions={"S": 1}, marks={"S": 100},
                      opening_equity=100, original_premium=original_premium)

    def valuation(time, spot):
        return ValuationPacket(time, {"S": spot}, {"unchanged": {"S": spot}}, observed_at={"S": time})

    events = [Event("opening", start, 0, valuation(start, 100), decision=True, terminal_decision=True,
                    earliest_execution=end, order_deadline=end),
              Event("exit", end, 0, valuation(end, exit_price), execution=True,
                    quotes=(Quote("exit", "S", end, end, exit_price, exit_price, 1, 1),))]
    result = run_case((instruments, opening, Rules(end, ("S",), 0), events), UnhedgedPolicy())
    comparison = write_reports([result], tmp_path)
    summary = read_summary(comparison, "unhedged")
    assert summary["opening_mode"] == "IMPORT" and summary["pnl_baseline"] == 100
    assert summary["terminal"]["cash"] == exit_price
    assert summary["terminal"]["net_pnl"] == exit_price - 100
    assert summary["terminal"]["premium_retention_ratio"] == expected_ratio
    assert not any(post.kind == "origination" for post in result.ledger.postings)


def test_data_gap_current_values_are_unknown_and_last_valid_checkpoint_is_timestamped(tmp_path):
    instruments, opening, rules, events = deterministic_case()
    events[2] = replace(events[2], valuation=None)
    result = run_case((instruments, opening, rules, events), ScheduledDeltaPolicy())
    comparison = write_reports([result], tmp_path)
    summary = read_summary(comparison, "delta")
    assert summary["status"] == "STOP_DATA_GAP"
    assert summary["current"]["equity"] is None
    assert summary["current"]["marked_pnl"] is None
    assert summary["current"]["pnl_components"] is None
    assert summary["last_valid_checkpoint"]["event_id"] == "hand/1"
    assert summary["last_valid_checkpoint"]["equity"] == pytest.approx(9994)
    assert summary["last_valid_checkpoint"]["time"] != summary["current"]["as_of"]
    assert summary["current"]["total_cash"] == 13194
    assert summary["costs"]["fees"] == 3
    assert summary["terminal"] is None
    assert all(row["current_unit_mark"] is None for row in summary["residual_inventory"])
    account = read_csv(tmp_path / "delta" / "account.csv")[-1]
    assert account["equity"] == account["net_pnl"] == ""


def test_committed_settlement_and_costs_are_reported_even_when_later_valuation_is_missing(tmp_path):
    class ScriptedOptions:
        name = "option-test"

        def targets(self, view):
            return {"OPTION": 1}

    instruments, opening, rules, events = option_accounting_case()
    events[2] = replace(events[2], valuation=None)
    result = run_case((instruments, opening, rules, events), ScriptedOptions())
    comparison = write_reports([result], tmp_path)
    summary = read_summary(comparison, "option-test")
    assert summary["status"] == "STOP_DATA_GAP"
    assert summary["current"]["total_cash"] == 1028
    assert summary["current"]["equity"] is None
    assert summary["terminal"] is None
    assert summary["settlements"][0]["record"]["cashflow"] == 50
    assert summary["settlements"][0]["funded"]
    assert summary["data_gaps"][0]["mandatory_phase"] == "committed"
    assert summary["remaining_obligations"] == []
    assert summary["costs"]["fees"] == 1


def test_completed_with_unreleased_restrictions_is_rejected(tmp_path):
    result = run_case(deterministic_case(), UnhedgedPolicy())
    result.ledger.posted_margin_cash = 1
    with pytest.raises(ReconciliationError, match="released restrictions"):
        write_reports([result], tmp_path)
