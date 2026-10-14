"""Replay the four solver-failure episodes and one unchanged historical control.

Uses frozen data, checkpoints and experiment settings. Writes only to a new
output directory; no training, parameter selection or full-benchmark replacement.
"""

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from basket_hedging import historical as historical
from basket_hedging import scenario_hedge
from basket_hedging.engine import Engine
from basket_hedging.etf_data import load_prepared
from basket_hedging.historical_actions import verified_training_parent
from basket_hedging.historical_hedge import SurfaceBasketValuator
from basket_hedging.replay import Case, jsonable, write_json
from basket_hedging.reporting import write_reports
from basket_hedging.risk import UnnettedLiabilityStressMargin
from basket_hedging.valuation import StressScenario
from basket_hedging.volgan import VolGANModel


EPISODES = ("2020-01-02", "2022-06-02", "2022-09-01", "2022-12-01", "2023-02-02")
ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_original(path: Path):
    """Optional comparison with the pre-change solver saved before editing."""
    name = "basket_hedging._original_solver_validation"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.solve_hedge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-solver", type=Path,
                        help="optional pre-change source snapshot, for direct successful-solve comparison")
    args = parser.parse_args()
    output = args.output.resolve()
    frozen = ROOT / "reports/historical/benchmark"
    models = ROOT / "reports/historical/models"
    if output.exists() or output.is_relative_to(ROOT / "reports/historical"):
        parser.error("output must be a new directory outside frozen reports/historical")

    # Protect all prior reports, six raw files, prepared inputs and actions.
    paths = {p.resolve() for directory in (ROOT / "reports", ROOT / "data", args.raw, args.prepared)
             for p in directory.rglob("*") if p.is_file()}
    base_prepared = args.prepared.parent / "prepared-etf-v1"
    paths.update(p.resolve() for p in base_prepared.rglob("*") if p.is_file())
    hashes = {str(p): file_hash(p) for p in sorted(paths)}
    output.mkdir(parents=True)
    write_json(output / "frozen_hashes.json", hashes)
    frozen_summary = json.loads((frozen / "summary.json").read_text(encoding="utf-8"))
    config = historical.HistoricalConfig(**frozen_summary["config"])
    arrays, manifest = load_prepared(args.prepared)
    historical.validate_corporate_actions(arrays)
    parent_hash = verified_training_parent(args.prepared, arrays, manifest)
    model = VolGANModel.load(models / "volgan.json")
    assert file_hash(models / "volgan.json") == frozen_summary["model_hashes"]["volgan"]
    assert manifest["prepared_sha256"] == frozen_summary["data_manifest"]["prepared_sha256"]
    assert model.metadata["prepared_sha256"] == parent_hash
    assert model.available_at == historical.MODEL_AVAILABLE
    assert model.trained_through <= historical.MODEL_AVAILABLE
    mean, covariance, correlation = historical.training_dynamics(arrays)
    arrays = historical.expand_calendar(arrays, manifest)
    stocks = tuple(map(str, arrays["symbols"]))
    valuator = SurfaceBasketValuator(stocks, correlation, historical.MODEL_AVAILABLE,
        samples=config.pricing_samples, pricing_seed=config.pricing_seed,
        stress_scenarios=(StressScenario("unchanged"), StressScenario("spot_down_20", .8),
                         StressScenario("spot_up_20", 1.2), StressScenario("vol_up_25", 1, 1.25)))
    first = int(np.searchsorted(arrays["dates"], historical.TEST_START))
    scheduled = set(range(first, len(arrays["dates"]) - config.holding_sessions - 1,
                          config.holding_sessions + 1))
    failed = {r["episode"]: r for r in json.loads((frozen / "failed_runs.json").read_text())}
    frozen_rows = list(csv.DictReader((frozen / "episodes.csv").open(encoding="utf-8")))
    original_solver = load_original(args.original_solver) if args.original_solver else None
    fixtures = ROOT / "tests/fixtures/solver_regressions"
    captures = {case["episode"]: case for case in
                json.loads((fixtures / "manifest.json").read_text())["cases"]}
    solver = scenario_hedge.solve_hedge
    rows, checks, solver_calls = [], [], []

    for episode_id in EPISODES:
        start = int(np.searchsorted(arrays["dates"], episode_id))
        assert start in scheduled and str(arrays["dates"][start]) == episode_id
        instruments, opening, rules, events = historical.episode(arrays, start, config, valuator)
        provider = historical.HistoricalScenarios("volgan", valuator, config,
                                                mean=mean, covariance=covariance, model=model)
        weights = {key: config.half_spread * float(arrays["close"][start, j])
                   for j, key in enumerate(stocks)}
        policy = scenario_hedge.ScenarioHedge(provider, stocks, weights, alpha=config.alpha)
        policy.name = "volgan"
        control_matches = []
        capture_checks = []
        current_decision = {}
        generate = provider.generate

        def recorded_generate(view, fixed, ids):
            current_decision["time"] = view.time.isoformat()
            return generate(view, fixed, ids)

        provider.generate = recorded_generate

        def checked_solve(*inputs, **settings):
            before = [np.array(value, copy=True) for value in inputs]
            captured = captures.get(episode_id)
            if captured and current_decision["time"] == captured["decision_time"]:
                with np.load(fixtures / captured["file"], allow_pickle=False) as saved_inputs:
                    names = ("delta_v", "delta_h", "previous_holdings", "cost_weights")
                    assert all(np.array_equal(a, saved_inputs[key]) for a, key in zip(before, names))
                assert settings == captured["kwargs"]
                capture_checks.append(True)
            solution = solver(*inputs, **settings)
            assert all(np.array_equal(a, b) for a, b in zip(before, inputs))
            if original_solver is not None and episode_id == EPISODES[0]:
                expected = original_solver(*inputs, **settings)
                for key, value in asdict(expected).items():
                    actual = getattr(solution, key)
                    assert np.array_equal(actual, value), (episode_id, key)
                control_matches.append(True)
            solver_calls.append(dict(episode=episode_id, decision_time=current_decision["time"],
                input_sha256=sha256(b"".join(a.astype(np.float64).tobytes() for a in before)).hexdigest(),
                settings=settings, iterations=solution.iterations, kkt_residual=solution.kkt_residual,
                polish_attempted=solution.polish_attempted, polish_succeeded=solution.polish_succeeded))
            return solution

        scenario_hedge.solve_hedge = checked_solve
        try:
            result, failure = historical._run_policy(
                Engine(instruments, opening, rules, UnnettedLiabilityStressMargin()), events, policy)
        finally:
            scenario_hedge.solve_hedge = solver
        directory = output / "episodes" / episode_id
        write_json(directory / "solver_diagnostics.json", policy.diagnostics)
        if failure is not None:
            write_json(directory / "failed_run.json", failure)
        else:
            replay_path = directory / "events.jsonl"
            with replay_path.open("w", encoding="utf-8") as stream:
                for event in events:
                    stream.write(json.dumps(jsonable(event), allow_nan=False) + "\n")
            case = Case(instruments, opening, rules, replay_path,
                        dict(label="Solver-polish validation; frozen experiment settings",
                             original_benchmark=str(frozen), episode=episode_id))
            write_reports([result], directory, case=case)
        components = result.ledger.components()
        completed = failure is None and result.status == "COMPLETED"
        previous = next(r for r in frozen_rows if r["episode"] == episode_id and r["policy"] == "volgan")
        rows.append(dict(episode=episode_id, original_status=previous["status"],
            corrected_status=failure["status"] if failure else result.status,
            net_pnl=components["net_pnl"] if completed else None,
            hedge_error=-(components["liability_pnl"] + components["hedge_pnl"]) if completed else None,
            trading_costs=components["trading_costs"], turnover=components["stock_turnover"],
            fills=len(result.fills), polished_decisions=sum(d["polish_succeeded"] for d in policy.diagnostics)))
        if episode_id == EPISODES[0]:
            assert completed
            for file in (directory / "volgan").glob("*.csv"):
                assert file.read_bytes() == (frozen / "representative/volgan" / file.name).read_bytes(), file
            saved = json.loads((frozen / "representative/volgan/summary.json").read_text())
            assert jsonable(result.decisions) == saved["decisions"]
            checks.append(dict(episode=episode_id, all_five_financial_csvs_byte_identical=True,
                               decisions_exactly_unchanged=True, original_solver_calls_identical=len(control_matches)))
        else:
            assert capture_checks == [True]
            old = failed[episode_id]
            for key, actual in (("postings", result.ledger.postings), ("order_outcomes", result.order_outcomes),
                                ("decisions", result.decisions)):
                assert jsonable(actual[:len(old[key])]) == old[key], (episode_id, key)
            checks.append(dict(episode=episode_id, original_successful_prefix_exactly_unchanged=True,
                               captured_regression_exactly_reproduced=True))
        print(f"{episode_id}: {previous['status']} -> {rows[-1]['corrected_status']}; "
              f"polished decisions={rows[-1]['polished_decisions']}; fills={len(result.fills)}", flush=True)

    with (output / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    changed = [path for path, digest in hashes.items() if file_hash(Path(path)) != digest]
    assert not changed, changed
    write_json(output / "validation.json", dict(
        label="Corrected solver validation only; original 71-episode benchmark remains frozen",
        config=asdict(config), episodes=rows, unchanged_checks=checks, solver_calls=solver_calls,
        frozen_file_count=len(hashes), frozen_files_unchanged=True,
        original_solver_sha256=file_hash(args.original_solver) if args.original_solver else None,
        current_source_hashes={p.name: file_hash(p) for p in (ROOT / "src/basket_hedging").glob("*.py")},
        failures=sum(row["corrected_status"] != "COMPLETED" for row in rows)))
    print(f"Validation reports: {output}; {len(hashes)} frozen hashes unchanged", flush=True)


if __name__ == "__main__":
    main()
