"""Small auditable reports from the authoritative ledger and event snapshots.

Valuation-dependent P&L is unavailable after a data-gap snapshot. Posting-derived
cost totals remain observable; they use Ledger.components without publishing its
valuation-dependent legs at an unavailable current mark. No distributional
statistics or comparison of unequal observation horizons is inferred here.
"""

import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .contracts import InputError, ReconciliationError, close
from .engine import OrderOutcome, Result, Snapshot
from .ledger import Posting
from .replay import Case, jsonable, load_case, write_json
from .risk import Risk


def _compact(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], headers: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=headers, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            plain = jsonable(row)
            writer.writerow({key: _compact(value) if isinstance(value, (dict, list)) else value
                             for key, value in plain.items()})


def _checkpoint(snapshot: Snapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return dict(event_id=snapshot.event_id, time=snapshot.time, sequence=snapshot.sequence,
                phase=snapshot.phase, equity=snapshot.risk.equity,
                marked_pnl=snapshot.pnl["net_pnl"] if snapshot.pnl is not None else None,
                pnl_components=snapshot.pnl, risk=snapshot.risk,
                positions=snapshot.positions, residual_delta=snapshot.residual_delta)


def _provenance(case_path: str | Path | None, case: Case | None) -> dict[str, Any]:
    records = []
    if case_path is not None:
        path = Path(case_path).resolve()
        records.append(dict(role="case_configuration", path=str(path),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if case is not None:
        path = case.events_path.resolve()
        records.append(dict(role="replay_events", path=str(path),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(files=records, metadata=case.metadata if case is not None else {},
                configuration=(dict(instruments=case.instruments, opening=case.opening, rules=case.rules)
                               if case is not None else None),
                hash_timing="File hashes are computed after the run for provenance; the engine never receives them.")


def _valuation_audit(result: Result) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    configurations = {}
    for record in result.valuations:
        metadata = record.get("metadata", {})
        # Dynamic seed/source IDs belong to an audit row, not repeated full configurations.
        configuration = {key: value for key, value in metadata.items()
                         if key not in {"pricing_seed_derived", "input_ids", "source"}}
        digest = hashlib.sha256(_compact(configuration).encode("utf-8")).hexdigest()
        configurations[digest] = configuration
        parameters = metadata.get("parameters", {})
        rows.append(dict(event_id=record["event_id"], as_of=record["as_of"],
                         model_version=metadata.get("model_version", metadata.get("model")),
                         parameters_sha256=digest, pricing_seed=parameters.get("pricing_seed"),
                         pricing_seed_derived=metadata.get("pricing_seed_derived"),
                         input_ids=record.get("input_ids", ()), source=metadata.get("source")))
    return rows, configurations


def _usage(result: Result) -> dict[str, Any]:
    risks = [snapshot.risk for snapshot in result.snapshots]
    ledger = result.ledger
    margin_requirements = [risk.required_margin for risk in risks if risk.required_margin is not None]
    # Every feasible CHECK allocates its computed margin. A pre-allocation risk
    # snapshot therefore also identifies that actual following cash allocation.
    allocated = [risk for risk in risks if risk.feasible and risk.required_margin is not None]
    buffer_headrooms = [risk.cash - risk.restricted_short_proceeds - risk.required_margin - risk.safety_buffer
                        for risk in allocated]
    return dict(
        minimum_total_cash=min([ledger.cash] + [risk.cash for risk in risks]),
        minimum_available_cash=min([ledger.available_cash] + [risk.available_cash for risk in risks]
                                  + [risk.cash - risk.restricted_short_proceeds - risk.required_margin for risk in allocated]),
        peak_posted_margin_cash=max([ledger.posted_margin_cash] + [risk.posted_margin_cash for risk in risks]
                                   + [risk.required_margin for risk in allocated]),
        peak_restricted_stock_short_proceeds=max([ledger.restricted_short_proceeds]
                                                 + [risk.restricted_short_proceeds for risk in risks]),
        peak_locked_cash=max([ledger.posted_margin_cash + ledger.restricted_short_proceeds]
                             + [risk.posted_margin_cash + risk.restricted_short_proceeds for risk in risks]
                             + [risk.required_margin + risk.restricted_short_proceeds for risk in allocated]),
        peak_required_margin=max(margin_requirements) if margin_requirements else None,
        minimum_buffer_headroom_after_allocation=min(buffer_headrooms) if buffer_headrooms else None,
        observation_scope="Released observations through this result's stopping time; mandatory-phase netting is respected.",
    )


def _summary(result: Result, provenance: Mapping[str, Any], valuation_configurations: Mapping[str, Any]) -> dict[str, Any]:
    if result.status not in {"COMPLETED", "STOP_WITH_BREACH", "STOP_DATA_GAP", "INCOMPLETE_EXIT"}:
        raise InputError("reports require a finished result")
    ledger = result.ledger
    final = result.snapshots[-1] if result.snapshots else None
    valid = final is not None and final.pnl is not None
    if result.status == "COMPLETED" and (not valid or ledger.positions
            or not close(ledger.posted_margin_cash, 0) or not close(ledger.restricted_short_proceeds, 0)):
        raise ReconciliationError("COMPLETED requires valued cash-only equity and released restrictions")
    # Accumulated fees/execution/financing/turnover remain known even when the
    # last market packet is unavailable. Publish no stale valuation-dependent leg.
    accumulated = ledger.components()
    current_components = final.pnl if valid else None
    current = dict(as_of=final.time if final else None, event_id=final.event_id if final else None,
                   valuation_valid=valid, equity=final.risk.equity if valid else None,
                   marked_pnl=current_components["net_pnl"] if valid else None,
                   pnl_components=current_components, total_cash=ledger.cash,
                   posted_margin_cash=ledger.posted_margin_cash,
                   restricted_stock_short_proceeds=ledger.restricted_short_proceeds,
                   available_cash=ledger.available_cash,
                   required_margin=final.risk.required_margin if final else None,
                   risk=final.risk if final else None,
                   residual_delta=final.residual_delta if valid else None)
    original_premium = ledger.original_premium
    terminal = None
    if result.status == "COMPLETED":
        terminal = dict(time=result.horizon, equity=ledger.cash, cash=ledger.cash,
                        net_pnl=current_components["net_pnl"], net_premium_retained=current_components["net_pnl"],
                        premium_retention_ratio=(current_components["net_pnl"] / original_premium
                                                 if original_premium is not None and original_premium > 0 else None))
    inventory = []
    obligations = []
    for key, quantity in sorted(ledger.positions.items()):
        instrument = ledger.instruments[key]
        mark = ledger.marks[key] if valid else None
        inventory.append(dict(instrument_id=key, quantity=quantity,
                              quantity_unit="shares" if instrument.kind == "stock" else "contracts",
                              multiplier=instrument.multiplier, current_unit_mark=mark,
                              current_position_value=quantity * instrument.multiplier * mark if mark is not None else None))
        if instrument.is_option:
            obligations.append(dict(instrument_id=key, quantity=quantity, multiplier=instrument.multiplier,
                                    expiry=instrument.expiry, settlement_status="outstanding"))
    known_shortfalls = [snapshot.risk.maximum_shortfall for snapshot in result.snapshots
                        if snapshot.risk.maximum_shortfall is not None]
    return dict(
        schema_version=1, policy=result.policy, status=result.status, reason=result.reason,
        currency=next(iter(ledger.instruments.values())).currency,
        opening_mode=result.opening_mode, pnl_baseline=result.baseline, horizon=result.horizon,
        margin_label=result.margin_label, current=current,
        last_valid_checkpoint=_checkpoint(result.last_valid_snapshot), terminal=terminal,
        original_gross_premium=original_premium,
        premium_normalization="Completed economic P&L / known positive original gross premium; IMPORT premium is metadata only; ratio is not clamped.",
        costs={key: accumulated[key] for key in ("execution_cost", "fees", "trading_costs", "interest",
                                                "borrow_charges", "financing_costs")},
        accepted_hedge_fill_count=len(result.fills),
        turnover=dict(stock_currency_traded=accumulated["stock_turnover"],
                      option_premium_currency_traded=accumulated["option_premium_turnover"],
                      units="Account currency; both trade sides counted; fees and liability origination/settlement excluded."),
        usage=_usage(result), breaches=result.breaches,
        max_observed_shortfall=max(known_shortfalls) if known_shortfalls else None,
        current_shortfall_unknown=final is None or final.risk.maximum_shortfall is None,
        max_known_unencumbered_cash_deficit=max([0] + [s.risk.unencumbered_cash_deficit for s in result.snapshots]),
        settlements=result.settlement_outcomes, remaining_obligations=obligations,
        residual_inventory=inventory, rejections=[outcome for outcome in result.order_outcomes if outcome.outcome == "rejected"],
        decisions=result.decisions, data_gaps=result.data_gaps, provenance=provenance,
        valuation_configurations=valuation_configurations,
        interpretation="A single hand-authored or synthetic path validates mechanics; it does not establish profitability, tail protection or broker compliance.",
    )


def _matched_checkpoint(results: Sequence[Result]) -> dict[str, Any] | None:
    checkpoints = [{(s.time, s.sequence, s.event_id): s for s in result.snapshots
                    if s.phase == "pretrade" and s.pnl is not None} for result in results]
    common = set(checkpoints[0])
    for available in checkpoints[1:]:
        common.intersection_update(available)
    if not common:
        return None
    key = max(common)
    return dict(time=key[0], sequence=key[1], event_id=key[2], phase="pretrade",
                policies={result.policy: dict(equity=available[key].risk.equity,
                                               net_pnl=available[key].pnl["net_pnl"],
                                               pnl_components=available[key].pnl)
                          for result, available in zip(results, checkpoints)})


def write_reports(results: Sequence[Result], output_directory: str | Path, *,
                  case_path: str | Path | None = None, case: Case | None = None) -> dict[str, Any]:
    """Write five CSV audit files plus a summary per policy and a common comparison."""
    if not results or len({result.policy for result in results}) != len(results):
        raise InputError("reports need nonempty results with unique policy names")
    if case is None and case_path is not None:
        case = load_case(case_path)
    provenance = _provenance(case_path, case)
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    used_names = set()
    outcomes = []
    for result in results:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", result.policy).strip("._") or "policy"
        if name in used_names:
            raise InputError("policy names collide as report-directory names")
        used_names.add(name)
        target = directory / name
        target.mkdir(exist_ok=True)
        valuation_rows, configurations = _valuation_audit(result)
        summary = _summary(result, provenance, configurations)
        write_json(target / "summary.json", summary)
        pnl_keys = tuple(result.ledger.components())
        account_rows = [dict(event_id=s.event_id, time=s.time, sequence=s.sequence, phase=s.phase,
                             **jsonable(s.risk),
                             **{key: s.pnl.get(key) if s.pnl is not None else None for key in pnl_keys},
                             positions=s.positions, residual_delta=s.residual_delta) for s in result.snapshots]
        _csv(target / "account.csv", account_rows,
             ("event_id", "time", "sequence", "phase", *(field.name for field in fields(Risk)), *pnl_keys,
              "positions", "residual_delta"))
        _csv(target / "orders.csv", [jsonable(outcome) for outcome in result.order_outcomes],
             tuple(field.name for field in fields(OrderOutcome)))
        _csv(target / "postings.csv", [jsonable(posting) for posting in result.ledger.postings],
             tuple(field.name for field in fields(Posting)))
        settlement_rows = []
        for outcome in result.settlement_outcomes:
            fixing = jsonable(outcome.get("fixing")) or {}
            settlement_rows.append(dict(**jsonable(outcome["record"]), funded=outcome["funded"],
                                        funding_convention=outcome["funding_convention"],
                                        official_fixing_id=fixing.get("id"),
                                        fixing_observed_at=fixing.get("observed_at"),
                                        fixing_available_at=fixing.get("available_at"),
                                        official_spots=fixing.get("spots")))
        _csv(target / "settlements.csv", settlement_rows,
             ("id", "time", "instrument_id", "quantity", "payoff", "cashflow", "funded", "funding_convention",
              "official_fixing_id", "fixing_observed_at", "fixing_available_at", "official_spots"))
        _csv(target / "valuation_audit.csv", valuation_rows,
             ("event_id", "as_of", "model_version", "parameters_sha256", "pricing_seed",
              "pricing_seed_derived", "input_ids", "source"))
        terminal = summary["terminal"] or {}
        outcomes.append(dict(policy=result.policy, status=result.status, reason=result.reason,
                             opening_mode=result.opening_mode, pnl_baseline=result.baseline,
                             observed_at=summary["current"]["as_of"], equity=summary["current"]["equity"],
                             marked_pnl=summary["current"]["marked_pnl"], terminal_pnl=terminal.get("net_pnl"),
                             premium_retention_ratio=terminal.get("premium_retention_ratio"),
                             summary_path=str(target / "summary.json")))
    comparison = dict(schema_version=1, outcomes=outcomes, matched_checkpoint=_matched_checkpoint(results),
                      provenance=provenance,
                      comparison_rule="Compare the latest common valid pretrade checkpoint. Outcome marked P&L and completed horizon P&L are separate; unequal stopping times are not ranked.")
    write_json(directory / "comparison.json", comparison)
    return jsonable(comparison)
