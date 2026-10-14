"""Attach verified IWM actions without altering the historical learning inputs."""

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .contracts import InputError
from .etf_data import load_prepared, sha256
from .replay import write_json


def _same(left: np.ndarray, right: np.ndarray) -> bool:
    return np.array_equal(left, right, equal_nan=True) if left.dtype.kind == "f" else np.array_equal(left, right)


def assert_unchanged_market_inputs(original: dict, supplemented: dict) -> None:
    """Exact array comparison, including all surfaces/dates and SPY/QQQ actions."""
    if original.keys() != supplemented.keys():
        raise InputError("corporate-action integration changed the prepared schema")
    for key in original:
        left, right = original[key], supplemented[key]
        if key in {"dividends", "splits"}:
            left, right = left[:, :2], right[:, :2]
        if not _same(left, right):
            raise InputError(f"corporate-action integration changed unrelated {key}")


def _merge(original: dict, supplement: Path) -> tuple[dict, dict]:
    from .corporate_actions import load_iwm_distributions

    rows, distributions = load_iwm_distributions(supplement)
    split_path = supplement / "split_verification.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    start, end = str(original["dates"][0]), str(original["dates"][-1])
    for record in (distributions, splits):
        if record["coverage_start"] > start or record["coverage_end"] < end:
            raise InputError("verified corporate-action coverage does not span the prepared dates")
    sources = splits.get("sources", [])
    hosts = {urlparse(source["url"]).hostname for source in sources}
    if (splits.get("verified_no_splits") is not True or splits.get("split_factor") != 1.0
            or splits.get("in_period_events") != [] or len(hosts - {None}) < 2):
        raise InputError("two-source verification of no in-period IWM splits is required")
    dates = dict(zip(map(str, original["dates"]), range(len(original["dates"]))))
    result = {key: value.copy() for key, value in original.items()}
    if tuple(map(str, result["symbols"])) != ("SPY", "QQQ", "IWM"):
        raise InputError("unexpected constituent ordering")
    # Absence means zero only inside the issuer-verified complete history.
    result["dividends"][:, 2] = 0.0
    result["splits"][:, 2] = 1.0
    outside_joint = []
    for row in rows:
        if row["ex_date"] in dates:
            result["dividends"][dates[row["ex_date"]], 2] = float(row["cash_per_share"])
        else:
            outside_joint.append(row["ex_date"])
    for key in ("dividends", "splits"):
        known = np.isfinite(original[key][:, 2])
        if not np.array_equal(original[key][known, 2], result[key][known, 2]):
            raise InputError(f"supplement conflicts with already recorded IWM {key}")
    assert_unchanged_market_inputs(original, result)
    record = dict(distributions_manifest=distributions,
                  distributions_manifest_sha256=sha256(supplement / "distributions_manifest.json"),
                  split_verification=splits, split_verification_sha256=sha256(split_path),
                  distributions_outside_joint_dates=outside_joint,
                  accounting_timing="existing ex-date opening cash convention; payable date retained as provenance",
                  strategy_inputs="unchanged: actions are released accounting inputs, not model features")
    return result, record


def supplement_prepared(prepared: Path, supplement: Path, output: Path) -> dict:
    """Rebuild into a new directory; preserve raw Parquets and the training NPZ."""
    prepared, supplement, output = Path(prepared).resolve(), Path(supplement).resolve(), Path(output).resolve()
    if output == prepared:
        raise InputError("supplemented output must not overwrite the original prepared dataset")
    arrays, manifest = load_prepared(prepared)
    if "corporate_action_supplement" in manifest:
        raise InputError("use the original prepared dataset, not an already supplemented copy")
    updated, provenance = _merge(arrays, supplement)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "surfaces.npz", **updated)
    provenance.update(parent_directory=os.path.relpath(prepared, output),
                      parent_prepared_sha256=manifest["prepared_sha256"],
                      supplement_directory=os.path.relpath(supplement, output),
                      unchanged_arrays_verified=True)
    ready = all(np.isfinite(updated[key]).all() for key in ("dividends", "splits"))
    result = dict(manifest, prepared_sha256=sha256(output / "surfaces.npz"),
                  corporate_action_supplement=provenance, benchmark_ready=ready,
                  benchmark_blockers=[] if ready else ["other corporate-action records remain unavailable"])
    write_json(output / "manifest.json", result)
    return result


def verified_training_parent(prepared: Path, arrays: dict, manifest: dict) -> str:
    """Accept old checkpoints only after proving the rebuild changed actions alone."""
    provenance = manifest.get("corporate_action_supplement")
    if provenance is None:
        return manifest["prepared_sha256"]
    parent_path = Path(prepared) / provenance["parent_directory"]
    original, parent_manifest = load_prepared(parent_path)
    if parent_manifest["prepared_sha256"] != provenance["parent_prepared_sha256"]:
        raise InputError("original training dataset differs from the recorded parent hash")
    for key in ("split", "common", "common_date_gaps", "source_file_hashes", "grid", "smoothing"):
        if manifest.get(key) != parent_manifest.get(key):
            raise InputError(f"corporate-action rebuild changed {key}")
    supplement = Path(prepared) / provenance["supplement_directory"]
    expected, checked_provenance = _merge(original, supplement)
    for key, value in checked_provenance.items():
        if value != provenance.get(key):
            raise InputError("corporate-action supplement changed after preparation")
    if any(not _same(expected[key], arrays[key]) for key in expected):
        raise InputError("supplemented arrays do not equal the verified original plus IWM actions")
    return parent_manifest["prepared_sha256"]
