"""Bounded checkpoint selection on 2018, followed once by 2019 confirmation.

No financial backtests or hedge outcomes are loaded. Run each declared seed once,
then select-confirm. The plan is frozen before training; original files are read only.
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from basket_hedging.contracts import InputError
from basket_hedging.etf_data import load_prepared
from basket_hedging.historical_training import close_time, gaussian_history_data
from basket_hedging.learned_model import ConditionalGaussianModel
from basket_hedging.replay import write_json
from basket_hedging.volgan import (VolGANModel, fit_volgan, prepared_training_pairs,
                                   validate_prepared_volgan)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "reports/historical_calibration/plan.json"
ORIGINAL = ROOT / "reports/historical/models/volgan.json"
GAUSSIAN = ROOT / "reports/historical/models/gaussian.json"


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acceptance(candidate: dict, original: dict, gates: dict) -> dict[str, bool]:
    """Fixed joint-score and calibration safeguards; no financial metric enters."""
    for metrics in (candidate, original):
        required = [metrics["energy_score"], metrics["spread_qqq_minus_iwm"]["interval_score_90"]]
        for key in ("return_5_95_percentile_coverage", "return_interval_score_90",
                    "return_mean_forecast_rmse", "log_iv_mean_forecast_rmse"):
            required.extend(metrics[key])
        if not np.isfinite(required).all():
            raise InputError("checkpoint acceptance requires finite generated diagnostics")
    original_error = np.abs(np.asarray(original["return_5_95_percentile_coverage"]) - .9)
    error = np.abs(np.asarray(candidate["return_5_95_percentile_coverage"]) - .9)
    return {
        "energy_score": candidate["energy_score"] <= gates["energy_score_ratio"] * original["energy_score"],
        "mean_coverage_error": float(original_error.mean() - error.mean()) >= gates["coverage_error_improvement"],
        "asset_coverage_errors": bool(np.all(error <= original_error + gates["max_asset_coverage_deterioration"])),
        **{key: bool(np.all(np.asarray(candidate[key]) <= gates["max_diagnostic_ratio"] * np.asarray(original[key])))
           for key in ("return_interval_score_90", "return_mean_forecast_rmse", "log_iv_mean_forecast_rmse")},
        "spread_interval_score": candidate["spread_qqq_minus_iwm"]["interval_score_90"] <=
            gates["spread_interval_score_ratio"] * original["spread_qqq_minus_iwm"]["interval_score_90"],
    }


def dated_data(prepared: Path, year: int, plan: dict) -> dict:
    if file_hash(prepared / "manifest.json") != plan["inputs"]["prepared_manifest_sha256"]:
        raise InputError("prepared manifest differs from the preregistered experiment")
    data, manifest = load_prepared(prepared)
    if manifest["prepared_sha256"] != plan["inputs"]["prepared_sha256"]:
        raise InputError("prepared input differs from the preregistered experiment")
    count = int(np.searchsorted(data["dates"], f"{year + 1}-01-01"))
    n = len(data["dates"])
    return {key: value[:count].copy() if value.ndim and value.shape[0] == n else value.copy()
            for key, value in data.items()}


def evaluate(model: VolGANModel, data: dict, year: int, plan: dict, sampler=None) -> dict:
    return validate_prepared_volgan(model, data, year=year, sampler=sampler,
        samples=plan["validation"]["samples"], seed=plan["validation"]["noise_seed"])


def gaussian_reference(original: VolGANModel, data: dict, year: int, plan: dict) -> dict:
    gaussian = ConditionalGaussianModel.load(GAUSSIAN)
    history = gaussian_history_data(data, split="validation", lags=gaussian.lags)
    rows = {time.date().isoformat(): index for index, time in enumerate(history.target_times)}

    def sampler(feature, target_index, normals):
        row = rows[str(data["dates"][target_index])]
        returns = gaussian.sample_log_returns(history.features[row], history.horizon_days[row], normals[:, :3])
        return (data["close"][target_index - 1] * np.exp(returns),
                np.broadcast_to(data["iv"][target_index - 1], (len(normals), *data["iv"].shape[1:])))

    result = evaluate(original, data, year, plan, sampler)
    result["label"] = "Frozen Gaussian reference: calendar-day scaling and carried-current IV"
    result["selection_caveat"] = "Original Gaussian checkpoint was selected using combined 2018-2019 validation."
    return result


def write_candidates(path: Path, records: list[dict]) -> None:
    fields = ("seed", "epoch", "eligible", "energy_score", "coverage_SPY", "coverage_QQQ",
              "coverage_IWM", "mean_coverage_error", "spread_interval_score_90", "failed_gates",
              "generator_sha256", "discriminator_sha256")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            metrics = row["diagnostics"]
            if metrics is None:
                writer.writerow({key: row[key] for key in ("seed", "epoch", "eligible", "generator_sha256",
                                                           "discriminator_sha256")} | {"failed_gates": row["error"]})
                continue
            coverage = metrics["return_5_95_percentile_coverage"]
            writer.writerow(dict(seed=row["seed"], epoch=row["epoch"], eligible=row["eligible"],
                energy_score=metrics["energy_score"],
                **dict(zip(("coverage_SPY", "coverage_QQQ", "coverage_IWM"), coverage)),
                mean_coverage_error=float(np.mean(np.abs(np.asarray(coverage) - .9))),
                spread_interval_score_90=metrics["spread_qqq_minus_iwm"]["interval_score_90"],
                failed_gates=";".join(key for key, passed in row["gates"].items() if not passed),
                generator_sha256=row["generator_sha256"], discriminator_sha256=row["discriminator_sha256"]))


def develop(prepared: Path, output: Path, plan: dict, seed: int) -> None:
    directory = output / f"seed_{seed}"
    directory.mkdir()  # Never silently repeat an attempted candidate seed.
    data = dated_data(prepared, 2018, plan)
    original = VolGANModel.load(ORIGINAL)
    baseline = evaluate(original, data, 2018, plan)
    write_json(directory / "original_2018.json", baseline)
    x, y, indices = prepared_training_pairs(data, original.grid, split="train")
    arrays_hash = sha256(x.astype("<f8").tobytes() + y.astype("<f8").tobytes()).hexdigest()
    if arrays_hash != plan["inputs"]["training_arrays_sha256"]:
        raise InputError("training rows differ from the original frozen training inputs")
    config = replace(original.config, seed=seed)
    best, records = None, []
    began = perf_counter()
    write_json(directory / "attempt.json", dict(seed=seed, config=asdict(config),
        plan_sha256=file_hash(output / "plan.json"), training_arrays_sha256=arrays_hash,
        source_sha256={path.name: file_hash(path) for path in (
            ROOT / "src/basket_hedging/volgan.py", ROOT / "src/basket_hedging/generator_validation.py", Path(__file__))},
        started_at=datetime.now(timezone.utc).isoformat(), development_last_date=str(data["dates"][-1])))

    def checkpoint(model):
        nonlocal best
        epoch = model.metadata["selected_epoch"]
        error = None
        try:
            diagnostics = evaluate(model, data, 2018, plan)
            gates = acceptance(diagnostics, baseline, plan["acceptance"])
        except (InputError, FloatingPointError) as problem:
            diagnostics, gates, error = None, {"finite_generation": False}, str(problem)
        record = dict(seed=seed, epoch=epoch, eligible=all(gates.values()), gates=gates, diagnostics=diagnostics,
            generator_sha256=model.metadata["final_generator_sha256"],
            discriminator_sha256=model.metadata["final_discriminator_sha256"], error=error)
        records.append(record)
        rank = (diagnostics["energy_score"], epoch, seed) if diagnostics else None
        if record["eligible"] and (best is None or rank < best):
            model.metadata = {**model.metadata, "checkpoint_selection": "2018 joint energy score subject to preregistered safeguards",
                              "calibration_plan_sha256": file_hash(output / "plan.json")}
            model.save(directory / "best_eligible.json")
            best = rank
        write_json(directory / "checkpoints.json", records)
        write_candidates(directory / "candidates.csv", records)
        score = f"{diagnostics['energy_score']:.8g}" if diagnostics else "invalid"
        print(f"seed={seed} epoch={epoch} ES={score} eligible={record['eligible']}", flush=True)

    def progress(row):
        if row["epoch"] == 1 or row["epoch"] % 1000 == 0:
            print(f"seed={seed} {row['phase']} epoch={row['epoch']} elapsed={perf_counter()-began:.1f}s", flush=True)

    try:
        final = fit_volgan(x, y, asset_ids=original.asset_ids, grid=original.grid, config=config,
            available_at=original.available_at, trained_through=close_time(str(data["dates"][indices[-1]])),
            metadata={**original.metadata, "calibration_experiment": "unchanged training; validation checkpoint selection"},
            progress=progress, checkpoint_epochs=tuple(plan["checkpoint_epochs"]), checkpoint=checkpoint)
    except Exception as error:
        write_json(directory / "result.json", dict(status="TRAINING_FAILED", seed=seed, error=str(error),
                                                    completed_checkpoint_count=len(records)))
        raise
    write_json(directory / "result.json", dict(status="DEVELOPMENT_COMPLETE", seed=seed,
        selected_epoch=best[1] if best else None, selected_energy_score=best[0] if best else None,
        selected_checkpoint_sha256=file_hash(directory / "best_eligible.json") if best else None,
        final_generator_sha256=final.metadata["final_generator_sha256"],
        final_discriminator_sha256=final.metadata["final_discriminator_sha256"],
        final_matches_original=(final.metadata["final_generator_sha256"] == original.metadata["final_generator_sha256"]
                                and final.metadata["final_discriminator_sha256"] == original.metadata["final_discriminator_sha256"])
                               if seed == original.config.seed else None,
        checkpoint_count=len(records), elapsed_seconds=perf_counter()-began))
    write_json(directory / "training_history.json", list(final.training_history))


def select_confirm(prepared: Path, output: Path, plan: dict) -> None:
    if (output / "selection.json").exists():
        raise InputError("selection is already frozen; no second confirmation attempt")
    records, choices = [], []
    for seed in plan["seeds"]:
        directory = output / f"seed_{seed}"
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if (directory / "checkpoints.json").exists():
            records.extend(json.loads((directory / "checkpoints.json").read_text(encoding="utf-8")))
        if result["status"] == "DEVELOPMENT_COMPLETE" and result["selected_epoch"] is not None:
            model_path = directory / "best_eligible.json"
            if file_hash(model_path) != result["selected_checkpoint_sha256"]:
                raise InputError("selected development checkpoint hash mismatch")
            choices.append((result["selected_energy_score"], result["selected_epoch"], seed, model_path))
    write_candidates(output / "candidates.csv", records)
    original = VolGANModel.load(ORIGINAL)
    development = dated_data(prepared, 2018, plan)
    write_json(output / "gaussian_2018.json", gaussian_reference(original, development, 2018, plan))
    consistent = len(choices) == len(plan["seeds"])
    choice = min(choices) if consistent else None
    selection = dict(status="FROZEN_FOR_CONFIRMATION" if consistent else "NO CONFIRMED IMPROVEMENT",
        frozen_at=datetime.now(timezone.utc).isoformat(), plan_sha256=file_hash(output / "plan.json"),
        both_seeds_have_eligible_checkpoints=consistent,
        selected_seed=choice[2] if choice else None, selected_epoch=choice[1] if choice else None,
        checkpoint_sha256=file_hash(choice[3]) if choice else None,
        reason=None if consistent else "No eligible checkpoint for both seeds; confirmation not inspected.")
    write_json(output / "selection.json", selection)
    if not consistent:
        write_json(output / "result.json", selection)
        print("NO CONFIRMED IMPROVEMENT: development safeguards not met consistently; 2019 not inspected.")
        return
    # Selection is persisted and hashed before reading any 2019 observations.
    selection_hash = file_hash(output / "selection.json")
    confirmation = dated_data(prepared, 2019, plan)
    selected = VolGANModel.load(choice[3])
    baseline = evaluate(original, confirmation, 2019, plan)
    candidate = evaluate(selected, confirmation, 2019, plan)
    gates = acceptance(candidate, baseline, plan["acceptance"])
    write_json(output / "original_2019.json", baseline)
    write_json(output / "selected_2019.json", candidate)
    write_json(output / "gaussian_2019.json", gaussian_reference(original, confirmation, 2019, plan))
    accepted = all(gates.values())
    if accepted:
        (output / "accepted.json").write_bytes(choice[3].read_bytes())
    write_json(output / "result.json", dict(status="ACCEPTED" if accepted else "NO CONFIRMED IMPROVEMENT",
        selection_sha256=selection_hash, confirmation_gates=gates,
        accepted_checkpoint="accepted.json" if accepted else None,
        limitation="2018 and 2019 are developmental checks: their combined period was previously inspected."))
    print("ACCEPTED" if accepted else "NO CONFIRMED IMPROVEMENT: frozen candidate failed 2019 safeguards.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop", "select-confirm"))
    parser.add_argument("--seed", type=int, choices=(1729, 1731))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/historical_calibration")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if args.phase == "develop" and args.seed is None:
        parser.error("develop requires one preregistered --seed")
    for path, key in ((ORIGINAL, "original_volgan_sha256"), (GAUSSIAN, "original_gaussian_sha256")):
        if file_hash(path) != plan["inputs"][key]:
            raise InputError("a frozen source checkpoint changed")
    if args.output.resolve().is_relative_to((ROOT / "reports/historical").resolve()):
        raise InputError("output must be outside frozen historical reports")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "plan.json"
    if destination.exists() and destination.read_bytes() != PLAN.read_bytes():
        raise InputError("output already has a different frozen plan")
    if not destination.exists():
        destination.write_bytes(PLAN.read_bytes())
    if args.phase == "develop":
        develop(args.prepared, args.output, plan, args.seed)
    else:
        select_confirm(args.prepared, args.output, plan)


if __name__ == "__main__":
    main()
