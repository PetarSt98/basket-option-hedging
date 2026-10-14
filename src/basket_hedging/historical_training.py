"""Chronological real-history adapter for the existing Gaussian baseline.

This reuses the existing feature construction and likelihood training unchanged.
It does not describe historical observations as independent synthetic paths or
invent simulator seeds. The baseline generates returns only; historical scenario
pricing must explicitly retain the current surface as its surface assumption.
"""

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timezone
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np

from .contracts import InputError
from .learned_model import ConditionalGaussianModel, fit_generator
from .learning_data import FEATURE_VERSION, conditioning_features
from .market import Observation


TRAIN_END = date(2017, 12, 29)
VALIDATION_START = date(2018, 1, 2)
VALIDATION_END = date(2019, 12, 31)
MODEL_AVAILABLE = datetime(2020, 1, 1, tzinfo=timezone.utc)


def close_time(day: str | date) -> datetime:
    """Declared regular US close clock; date-only source does not record release latency."""
    day = date.fromisoformat(str(day)) if not isinstance(day, date) else day
    return datetime.combine(day, time(16), ZoneInfo("America/New_York")).astimezone(timezone.utc)


@dataclass(frozen=True)
class HistoricalTrainingData:
    stock_ids: tuple[str, ...]
    lags: int
    features: np.ndarray
    log_returns: np.ndarray
    horizon_days: np.ndarray
    decision_times: tuple[datetime, ...]
    target_times: tuple[datetime, ...]
    split: str
    input_hashes: Mapping[str, str]

    @property
    def seeds(self) -> tuple:
        return ()  # Observed data has no simulator seed; training has its own seed.

    @property
    def path_ids(self) -> tuple[str, ...]:
        return (f"historical-{self.split}-segment",) * len(self.features)

    def manifest(self) -> dict:
        record = dict(split=self.split, rows=len(self.features), stock_ids=self.stock_ids,
                      lags=self.lags, feature_version=FEATURE_VERSION,
                      decision_start=min(self.decision_times).isoformat(),
                      decision_end=max(self.decision_times).isoformat(),
                      target_start=min(self.target_times).isoformat(),
                      target_end=max(self.target_times).isoformat(),
                      data_kind="one joint historical ETF time series; rows are not independent paths",
                      input_hashes=dict(self.input_hashes), simulator_seeds=None,
                      feature_normalization="fit only on the declared training segment",
                      target_units="joint raw close-to-close log returns; calendar-day Gaussian scaling",
                      surface_assumption="restricted-information baseline: current IV surfaces retained")
        digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        for array in (self.features, self.log_returns, self.horizon_days):
            digest.update(array.astype("<f8").tobytes())
        for decisions, targets in zip(self.decision_times, self.target_times):
            digest.update((decisions.isoformat() + targets.isoformat()).encode())
        return {**record, "dataset_sha256": digest.hexdigest()}


def historical_training_indices(arrays: Mapping[str, np.ndarray], *, split: str,
                                warmup: int = 21, train_end: date = TRAIN_END,
                                validation_start: date = VALIDATION_START,
                                validation_end: date = VALIDATION_END) -> tuple[int, ...]:
    """Shared Gaussian/VolGAN chronological decision indices, label at index+1.

    Both endpoint dates belong to their declared split. Feature history stops at
    the decision; target values never enter a feature. All assets share each row.
    The 21-session warm-up matches the VolGAN comparison's information availability.
    """
    if split not in {"train", "validation"}:
        raise InputError("historical learning split must be train or validation")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
        raise InputError("historical warm-up must be an integer >= 1")
    if not train_end < validation_start <= validation_end:
        raise InputError("historical train and validation dates must be ordered and disjoint")
    try:
        dates = tuple(date.fromisoformat(str(value)) for value in arrays["dates"])
        stocks = tuple(str(value) for value in arrays["symbols"])
        prices = np.asarray(arrays["close"], dtype=float)
        consecutive = np.asarray(arrays["consecutive"])
        valid_surface = np.asarray(arrays["valid_surface"])
    except (KeyError, TypeError, ValueError) as error:
        raise InputError("malformed prepared historical arrays") from error
    if (not dates or any(later <= earlier for earlier, later in zip(dates, dates[1:]))
            or not stocks or len(set(stocks)) != len(stocks)
            or prices.shape != (len(dates), len(stocks))
            or consecutive.shape != (len(dates),) or consecutive.dtype.kind != "b"
            or valid_surface.shape != prices.shape or valid_surface.dtype.kind != "b"):
        raise InputError("historical dates, asset ordering, and validity masks are inconsistent")
    valid = valid_surface.all(axis=1) & np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    indices = []
    for index in range(warmup, len(dates) - 1):
        if split == "train":
            if dates[index + 1] > train_end:
                continue
        elif not (dates[index] >= validation_start and dates[index + 1] <= validation_end):
            continue
        # Do not let missing observations turn a several-session move into one day.
        start = index - warmup
        if not valid[start:index + 2].all() or not consecutive[start + 1:index + 2].all():
            continue
        indices.append(index)
    if not indices:
        raise InputError(f"no valid historical {split} examples after chronological and data-quality checks")
    return tuple(indices)


def gaussian_history_data(arrays: Mapping[str, np.ndarray], *, split: str,
                          input_hashes: Mapping[str, str] | None = None,
                          lags: int = 3, warmup: int = 21,
                          train_end: date = TRAIN_END, validation_start: date = VALIDATION_START,
                          validation_end: date = VALIDATION_END) -> HistoricalTrainingData:
    """Apply the unchanged Gaussian features to the common historical samples."""
    if isinstance(lags, bool) or not isinstance(lags, int) or lags < 0 or warmup < lags:
        raise InputError("history lags must be an integer between zero and warm-up")
    indices = historical_training_indices(arrays, split=split, warmup=warmup, train_end=train_end,
                                          validation_start=validation_start, validation_end=validation_end)
    dates = tuple(date.fromisoformat(str(value)) for value in arrays["dates"])
    stocks = tuple(str(value) for value in arrays["symbols"])
    prices = np.asarray(arrays["close"], dtype=float)
    times = tuple(close_time(day) for day in dates)
    features, labels, horizons, decisions, targets = [], [], [], [], []
    for index in indices:
        history = tuple(Observation(f"close/{dates[row]}/{stock}", stock, float(prices[row, asset]),
                                    times[row], times[row])
                        for row in range(index - lags, index + 1)
                        for asset, stock in enumerate(stocks))
        current = dict(zip(stocks, map(float, prices[index])))
        feature, _ = conditioning_features(stocks, history, current, times[index], times[index + 1], lags=lags)
        features.append(feature)
        labels.append(np.log(prices[index + 1] / prices[index]))
        horizons.append((times[index + 1] - times[index]).total_seconds() / 86400)
        decisions.append(times[index])
        targets.append(times[index + 1])
    immutable = []
    for values in (features, labels, horizons):
        array = np.asarray(values, dtype=float)
        immutable.append(np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape))
    return HistoricalTrainingData(stocks, lags, *immutable, tuple(decisions), tuple(targets),
                                  split, dict(input_hashes or {}))


def train_historical_gaussian(prepared_directory: str | Path, output_directory: str | Path, *,
                               epochs: int = 40, seed: int = 271828) -> ConditionalGaussianModel:
    """Fit a separate historical checkpoint; never load/overwrite the toy checkpoint."""
    from .etf_data import load_prepared

    arrays, manifest = load_prepared(prepared_directory)
    manifest_path = Path(prepared_directory) / "manifest.json"
    input_hashes = {"prepared_manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    train = gaussian_history_data(arrays, split="train", input_hashes=input_hashes)
    validation = gaussian_history_data(arrays, split="validation", input_hashes=input_hashes)
    started = perf_counter()
    model = fit_generator(train, validation, seed=seed, epochs=epochs, available_at=MODEL_AVAILABLE)
    training_seconds = perf_counter() - started
    model = replace(model, metadata={**model.metadata, "historical_data_manifest": manifest,
                                     "training_seconds": training_seconds,
                                     "comparison_label": "restricted-information Gaussian baseline; frozen current surfaces"})
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.json")
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "train_nll", "validation_nll"))
        writer.writeheader()
        writer.writerows(dict(row) for row in model.training_history)
    return model
