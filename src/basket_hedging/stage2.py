"""Frozen-design synthetic Stage 2 experiment; no financial-parameter selection.

Train, check conditional distributions, freeze the artifact, then evaluate fresh
held-out paths. Saved replay files go through the ordinary case loader. Only
compact metrics and reproducibility artifacts remain in the output directory.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import tempfile
from typing import Callable

import numpy as np

from .contracts import InputError
from .engine import Engine
from .generator_validation import validate_generator
from .learned_model import ConditionalGaussianModel, fit_generator
from .learned_scenarios import LearnedScenarios
from .learning_data import experiment_seed, generate_learning_data
from .policies import ScheduledDeltaPolicy, UnhedgedPolicy
from .replay import load_case, write_json
from .risk import UnnettedLiabilityStressMargin
from .scenario_comparison import _metrics
from .scenario_hedge import ScenarioHedge
from .scenarios import ConditionalGBMScenarios
from .synthetic import generate_synthetic_case
from .valuation import PricingParameters


POLICIES = ("unhedged", "delta", "true_simulator", "learned_generator")
STOCKS = ("S1", "S2", "S3")
SPOTS = (100., 80., 120.)
VOLATILITIES = (.20, .225, .25)
CORRELATION = ((1., .35, .35), (.35, 1., .35), (.35, .35, 1.))
MASTER_SEED = 20260920


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_path(index: int, model_path: str, settings: dict) -> dict:
    """One fresh saved replay, shared by all four policies. No training here."""
    replay_seed = experiment_seed("test-path", index, settings["master_seed"])
    scenario_seed = experiment_seed("test-scenarios", index, settings["master_seed"])
    model = ConditionalGaussianModel.load(model_path)
    with tempfile.TemporaryDirectory(prefix="basket-stage2-path-") as temporary:
        path = generate_synthetic_case(
            Path(temporary) / "case.json", replay_seed=replay_seed, pricing_seed=731,
            samples=settings["accounting_samples"], days=settings["days"])
        case = load_case(path)
        raw = dict(case.metadata["pricing_parameters"])
        raw["available_at"] = datetime.fromisoformat(raw["available_at"])
        raw["samples"] = settings["pricing_samples"]
        pricing = PricingParameters(**raw)
        options = dict(samples=settings["scenarios"], seed=scenario_seed,
                       max_observation_age_seconds=case.rules.max_mark_age_seconds)
        true_source = ConditionalGBMScenarios(
            pricing, replay_volatilities=tuple(case.metadata["replay_volatilities"]),
            replay_correlation=tuple(tuple(row) for row in case.metadata["replay_correlation"]), **options)
        learned_source = LearnedScenarios(model, pricing, **options)
        candidates = tuple(case.rules.hedge_universe)
        costs = dict.fromkeys(candidates, settings["cost_proxy"])
        policies = [UnhedgedPolicy(), ScheduledDeltaPolicy(),
                    ScenarioHedge(true_source, candidates, costs, alpha=settings["alpha"]),
                    ScenarioHedge(learned_source, candidates, costs, alpha=settings["alpha"])]
        results, rows = [], []
        for name, policy in zip(POLICIES, policies):
            # Labels only: the policy, objective and target interface are unchanged.
            policy.name = name
            result = Engine(case.instruments, case.opening, case.rules,
                            UnnettedLiabilityStressMargin(case.rules.expected_stress_scenarios)).run(case.events(), policy)
            results.append(result)
            metrics = _metrics(result)
            completed = result.status == "COMPLETED"
            rows.append(dict(
                path=index, replay_seed=replay_seed, scenario_seed=scenario_seed, policy=name,
                status=result.status, reason=result.reason,
                terminal_pnl=metrics["net_pnl"] if completed else None,
                terminal_hedge_error=metrics["gross_hedge_error_currency"] if completed else None,
                observed_pnl=metrics["net_pnl"], observed_at=metrics["observed_at"].isoformat() if metrics["observed_at"] else None,
                turnover=metrics["stock_turnover_currency"] + metrics["option_premium_turnover_currency"],
                transaction_costs=metrics["transaction_costs"], financing_costs=metrics["financing_costs"],
                fills=metrics["accepted_hedge_fill_count"], breaches=metrics["breach_count"],
                optimizer_decisions=len(policy.diagnostics) if isinstance(policy, ScenarioHedge) else 0,
                skipped_decisions=sum(d["status"] == "skipped" for d in result.decisions),
                rejected_orders=sum(o.outcome == "rejected" for o in result.order_outcomes),
                residual_inventory=dict(result.ledger.positions),
            ))
        # For any stopped run also preserve a genuine common pretrade checkpoint;
        # never compare its stopping P&L against another policy's terminal P&L.
        snapshots = [{(s.time, s.sequence, s.event_id): s for s in r.snapshots
                      if s.phase == "pretrade" and s.pnl is not None} for r in results]
        common = set(snapshots[0]).intersection(*(set(s) for s in snapshots[1:]))
        checkpoint = max(common) if common else None
        for row, points in zip(rows, snapshots):
            row["matched_checkpoint"] = checkpoint[0].isoformat() if checkpoint else None
            row["matched_marked_pnl"] = points[checkpoint].pnl["net_pnl"] if checkpoint else None
            # CSV-friendly, explicit inventory rather than silently dropping it.
            row["residual_inventory"] = repr(row["residual_inventory"])
        return dict(rows=rows, replay_sha256=_hash(case.events_path), case_sha256=_hash(path),
                    replay_seed=replay_seed, scenario_seed=scenario_seed,
                    case_metadata=dict(case.metadata) if index == 0 else None)


def aggregate_results(rows: list[dict]) -> dict:
    """Terminal risk on matched completed paths, failure frequencies on all paths."""
    paths = sorted({row["path"] for row in rows})
    if not paths or len(rows) != len(paths) * len(POLICIES):
        raise InputError("evaluation requires exactly four outcomes per path")
    for path in paths:
        if sorted(r["policy"] for r in rows if r["path"] == path) != sorted(POLICIES):
            raise InputError("evaluation policy outcomes are missing or duplicated")
    matched = {path for path in paths if all(r["status"] == "COMPLETED" for r in rows if r["path"] == path)}
    outcomes = []
    for policy in POLICIES:
        all_rows = [r for r in rows if r["policy"] == policy]
        complete = [r for r in all_rows if r["path"] in matched]
        pnl = np.array([r["terminal_pnl"] for r in complete], dtype=float)
        error = np.array([r["terminal_hedge_error"] for r in complete], dtype=float)
        outcomes.append(dict(
            policy=policy, paths=len(paths), matched_completed_paths=len(complete),
            mean_terminal_pnl=float(np.mean(pnl)) if len(pnl) else None,
            terminal_pnl_sd=float(np.std(pnl, ddof=1)) if len(pnl) > 1 else None,
            hedge_error_mean=float(np.mean(error)) if len(error) else None,
            hedge_error_sd=float(np.std(error, ddof=1)) if len(error) > 1 else None,
            hedge_error_rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
            cvar95_loss=float(np.mean(np.sort(-pnl)[-max(1, int(np.ceil(.05 * len(pnl)))):])) if len(pnl) else None,
            mean_turnover=float(np.mean([r["turnover"] for r in all_rows])),
            mean_transaction_costs=float(np.mean([r["transaction_costs"] for r in all_rows])),
            mean_fills=float(np.mean([r["fills"] for r in all_rows])),
            breach_frequency=sum(r["breaches"] > 0 for r in all_rows) / len(all_rows),
            completion_frequency=sum(r["status"] == "COMPLETED" for r in all_rows) / len(all_rows),
            total_skipped_decisions=sum(r["skipped_decisions"] for r in all_rows),
            total_rejected_orders=sum(r["rejected_orders"] for r in all_rows),
            total_optimizer_decisions=sum(r["optimizer_decisions"] for r in all_rows),
        ))
    deviations = {r["policy"]: r["hedge_error_sd"] for r in outcomes}
    retained = None
    if all(deviations[key] is not None for key in ("unhedged", "true_simulator", "learned_generator")):
        denominator = deviations["unhedged"]**2 - deviations["true_simulator"]**2
        if denominator != 0:
            retained = (deviations["unhedged"]**2 - deviations["learned_generator"]**2) / denominator
    return dict(outcomes=outcomes, variance_reduction_retained=retained,
                matched_completed_paths=len(matched), total_paths=len(paths),
                definitions=dict(
                    hedge_error="-(liability P&L + hedge P&L), before actual trading and financing costs",
                    terminal_metrics="Only paths completed by all four policies; separate failure counts include every path",
                    sd="Sample standard deviation, ddof=1; currency units",
                    cvar95="Mean worst ceil(5% * matched path count) terminal net P&L losses; empirical, not a calibrated tail guarantee",
                    activity="Turnover, costs and fills through each observed stopping time, averaged across every path",
                    turnover="Gross two-sided traded stock currency plus option-premium currency; this experiment trades stocks only",
                    retained="(Var(unhedged error)-Var(learned error))/(Var(unhedged error)-Var(true error)); unclamped"))


def run_stage2(output_directory: str | Path, *, test_paths: int = 200, workers: int = 1,
               train_paths: int = 1024, validation_paths: int = 256, epochs: int = 40,
               days: int = 20, scenarios: int = 256, pricing_samples: int = 1024,
               accounting_samples: int = 8192, diagnostic_samples: int = 20000,
               progress: Callable[[str], None] = print) -> dict:
    """Fixed model/financial defaults; smaller counts exist for software tests only."""
    for name, value in locals().copy().items():
        if name in {"test_paths", "workers", "train_paths", "validation_paths", "epochs", "days",
                    "scenarios", "pricing_samples", "accounting_samples", "diagnostic_samples"}:
            minimum = 2 if name in {"scenarios", "pricing_samples", "accounting_samples", "diagnostic_samples"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise InputError(f"{name} must be an integer >= {minimum}")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(master_seed=MASTER_SEED, train_paths=train_paths, validation_paths=validation_paths,
                    test_paths=test_paths, epochs=epochs, days=days, scenarios=scenarios,
                    pricing_samples=pricing_samples, accounting_samples=accounting_samples,
                    diagnostic_samples=diagnostic_samples, alpha=.1, cost_proxy=.02,
                    training_seed=271828, hidden=24, batch_size=512, learning_rate=.001)
    progress("Generating disjoint training and validation paths")
    train = generate_learning_data(STOCKS, SPOTS, VOLATILITIES, CORRELATION, split="train",
                                   paths=train_paths, start=datetime(2024, 1, 8, 16, tzinfo=timezone.utc), days=days)
    validation = generate_learning_data(STOCKS, SPOTS, VOLATILITIES, CORRELATION, split="validation",
                                        paths=validation_paths, start=datetime(2025, 1, 6, 16, tzinfo=timezone.utc), days=days)
    test_seeds = [experiment_seed("test-path", index, MASTER_SEED) for index in range(test_paths)]
    if set(test_seeds) & (set(train.seeds) | set(validation.seeds)):
        raise InputError("training/validation/test seeds overlap")
    progress("Training conditional Gaussian generator; selecting checkpoint by validation NLL only")
    model = fit_generator(train, validation, seed=271828, epochs=epochs, hidden=24, batch_size=512,
                          learning_rate=.001, available_at=datetime(2025, 12, 31, tzinfo=timezone.utc))
    model_path = output / "model.json"
    model.save(model_path)
    frozen_hash = _hash(model_path)
    write_json(output / "experiment.json", dict(
        settings=settings, model_sha256=frozen_hash,
        train=train.manifest(), validation=validation.manifest(),
        test=dict(paths=test_paths, seed_stream="test-path", seeds=test_seeds,
                  start="2026-01-05T16:00:00+00:00", used_for_training_or_selection=False),
        seed_derivation="SHA256 namespaced JSON [namespace, master, stream, index], first 128 bits",
        scenario_pairing="Same per-row PCG64 standard normals for true and learned providers",
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        source_sha256={p.name: _hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}))
    _write_csv(output / "training_history.csv", [dict(row) for row in model.training_history])
    progress("Validating conditional distributions before running any held-out hedge path")
    quality = validate_generator(model, initial_spots=SPOTS, volatilities=VOLATILITIES,
                                  correlation=CORRELATION, coefficients=(1/3,) * 3,
                                  samples=diagnostic_samples, seed=314159)
    write_json(output / "generator_validation.json", quality)
    progress(f"Frozen model {frozen_hash}; evaluating {test_paths} fresh paths with {workers} workers")
    completed = {}
    if workers == 1:
        for index in range(test_paths):
            completed[index] = _evaluate_path(index, str(model_path.resolve()), settings)
            if (index + 1) % 10 == 0 or index + 1 == test_paths:
                progress(f"Held-out paths complete: {index + 1}/{test_paths}")
    else:
        # Spawned workers import NumPy under these limits. Restore parent settings.
        variables = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        previous = {key: os.environ.get(key) for key in variables}
        try:
            for key in variables:
                os.environ[key] = "1"
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_evaluate_path, i, str(model_path.resolve()), settings): i for i in range(test_paths)}
                for future in as_completed(futures):
                    completed[futures[future]] = future.result()
                    if len(completed) % 10 == 0 or len(completed) == test_paths:
                        progress(f"Held-out paths complete: {len(completed)}/{test_paths}")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    if _hash(model_path) != frozen_hash:
        raise InputError("frozen model changed during test evaluation")
    rows = [row for index in sorted(completed) for row in completed[index]["rows"]]
    _write_csv(output / "path_results.csv", rows)
    summary = aggregate_results(rows)
    summary.update(settings=settings, model_sha256=frozen_hash,
                   margin_label="uncalibrated research margin proxy",
                   interpretation="Held-out synthetic validation, not historical performance or exact VolGAN reproduction")
    write_json(output / "hedging_summary.json", summary)
    write_json(output / "test_manifest.json", dict(
        paths=[dict(index=index, **{k: v for k, v in completed[index].items()
                                   if k not in {"rows", "case_metadata"}}) for index in sorted(completed)],
        case_metadata=completed[0]["case_metadata"]))
    return summary
