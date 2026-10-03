"""Past-only conditioning features and separate synthetic train/validation data.

Feature construction contains no fitting or normalization learned from a dataset.
Supervised labels use the next declared business-day close; feature histories stop
at the decision. The existing zero-drift replay generator supplies prices only.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import log, sqrt
from typing import Mapping, Sequence

import numpy as np

from .contracts import InputError, number, timestamp
from .market import Observation
from .synthetic import _next_business_date, market_path


FEATURE_VERSION = "joint-log-price-and-causal-return-lags-v1"
SEED_NAMESPACE = "basket-hedging-learning-v1"


def _stock_ids(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raise InputError("stock_ids must be a sequence of instrument names")
    try:
        stocks = tuple(value)
    except TypeError as error:
        raise InputError("stock_ids must be a sequence of instrument names") from error
    if not stocks or any(not isinstance(stock, str) or not stock for stock in stocks) or len(set(stocks)) != len(stocks):
        raise InputError("stock_ids must be nonempty unique strings")
    return stocks


def _integer(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InputError(f"{name} must be an integer >= {minimum}")


def experiment_seed(stream: str, index: int, master_seed: int = 20260920) -> int:
    """Stable first 128 SHA256 bits; separate stream names create separate roots."""
    if not isinstance(stream, str) or not stream:
        raise InputError("seed stream must be a nonempty string")
    _integer(index, "seed index", 0)
    _integer(master_seed, "master seed", 0)
    payload = json.dumps([SEED_NAMESPACE, master_seed, stream, index], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:16], "big")


def conditioning_features(stock_ids: Sequence[str], history: Sequence[Observation],
                          current_spots: Mapping[str, float], time: datetime,
                          horizon: datetime, *, lags: int = 3) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return raw features and the observation IDs used, without future records.

    Layout: log current spots[D]; newest-first joint log-return bars divided by
    sqrt(calendar elapsed days)[lags*D]; presence masks[lags]; log elapsed days
    [lags, zero when missing]; log forward horizon days[1]. Thus D=3,lags=3
    gives 19 features. Only complete joint observed-time frames form lag bars.
    Missing history is zero-filled with explicit masks; no future backfilling.
    """
    stocks = _stock_ids(stock_ids)
    _integer(lags, "lags", 0)
    timestamp(time, "decision time")
    timestamp(horizon, "target time")
    if horizon <= time:
        raise InputError("feature horizon must follow decision time")
    prices = []
    for stock in stocks:
        if stock not in current_spots:
            raise InputError(f"missing current spot for {stock}")
        price = current_spots[stock]
        number(price, "current spot")
        if price <= 0:
            raise InputError("current spots must be positive")
        prices.append(price)

    frames: dict[datetime, dict[str, Observation]] = {}
    identities: dict[str, Observation] = {}
    for observation in history:
        if not isinstance(observation, Observation):
            raise InputError("history must contain Observation records")
        if (observation.instrument_id not in stocks or observation.available_at > time
                or observation.observed_at > time):
            continue
        if observation.value <= 0:
            raise InputError("released stock observations must be positive")
        same_id = identities.get(observation.id)
        if same_id is not None and same_id != observation:
            raise InputError("conflicting observation ID in feature history")
        identities[observation.id] = observation
        frame = frames.setdefault(observation.observed_at, {})
        prior = frame.get(observation.instrument_id)
        if prior is not None and prior.value != observation.value:
            raise InputError("conflicting stock values at the same observed time")
        if prior is None or (observation.available_at, observation.id) < (prior.available_at, prior.id):
            frame[observation.instrument_id] = observation
        if observation.observed_at == time and observation.value != current_spots[observation.instrument_id]:
            raise InputError("current spot conflicts with the released decision-time observation")

    complete = [(observed, frame) for observed, frame in sorted(frames.items()) if all(s in frame for s in stocks)]
    returns = np.zeros((lags, len(stocks)))
    masks, durations = np.zeros(lags), np.zeros(lags)
    used: dict[tuple[datetime, str], Observation] = {}
    # Current-price provenance where an observation is available; cold-start
    # callers may supply known current prices without earlier observations.
    for stock in stocks:
        available = [(observed, frame[stock]) for observed, frame in sorted(frames.items()) if stock in frame]
        if available and available[-1][1].value == current_spots[stock]:
            used[(available[-1][0], stock)] = available[-1][1]
    for lag in range(min(lags, max(len(complete) - 1, 0))):
        later_time, later = complete[-1-lag]
        earlier_time, earlier = complete[-2-lag]
        days = (later_time-earlier_time).total_seconds()/86400
        returns[lag] = [(log(later[s].value)-log(earlier[s].value))/sqrt(days) for s in stocks]
        masks[lag], durations[lag] = 1, log(days)
        for stock in stocks:
            used[(earlier_time, stock)] = earlier[stock]
            used[(later_time, stock)] = later[stock]
    forward_days = (horizon-time).total_seconds()/86400
    features = np.concatenate((np.log(prices), returns.ravel(), masks, durations, [log(forward_days)]))
    if not np.isfinite(features).all():
        raise InputError("feature arithmetic produced nonfinite values")
    features = np.frombuffer(features.astype("<f8").tobytes(), dtype="<f8")
    order = {stock: index for index, stock in enumerate(stocks)}
    input_ids = tuple(record.id for (_, stock), record in sorted(used.items(), key=lambda item: (item[0][0], order[item[0][1]])))
    return features, input_ids


def _readonly_array(value, name: str, dimensions: int) -> np.ndarray:
    try:
        if np.iscomplexobj(value):
            raise InputError(f"{name} must be real")
        array = np.asarray(value, dtype="<f8")
    except (TypeError, ValueError) as error:
        raise InputError(f"{name} must be a numeric array") from error
    if array.ndim != dimensions or not np.isfinite(array).all():
        raise InputError(f"{name} must be a finite {dimensions}-dimensional array")
    return np.frombuffer(array.tobytes(), dtype="<f8").reshape(array.shape)


@dataclass(frozen=True)
class TrainingData:
    stock_ids: tuple[str, ...]
    lags: int
    features: np.ndarray
    log_returns: np.ndarray
    horizon_days: np.ndarray
    decision_times: tuple[datetime, ...]
    target_times: tuple[datetime, ...]
    path_ids: tuple[str, ...]  # One path ID per row.
    split: str
    seeds: tuple[int, ...]  # One unique seed per path, in first-occurrence order.

    def __post_init__(self) -> None:
        object.__setattr__(self, "stock_ids", _stock_ids(self.stock_ids))
        _integer(self.lags, "lags", 0)
        if self.split not in {"train", "validation"}:
            raise InputError("learning split must be 'train' or 'validation'")
        for name, dimensions in (("features", 2), ("log_returns", 2), ("horizon_days", 1)):
            object.__setattr__(self, name, _readonly_array(getattr(self, name), name, dimensions))
        for name in ("decision_times", "target_times", "path_ids", "seeds"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        count, dimensions = self.log_returns.shape
        features = len(self.stock_ids)*(self.lags+1) + 2*self.lags + 1
        if (count < 1 or dimensions != len(self.stock_ids) or self.features.shape != (count, features)
                or self.horizon_days.shape != (count,)
                or any(len(getattr(self, name)) != count for name in ("decision_times", "target_times", "path_ids"))):
            raise InputError("learning arrays, row metadata and feature dimensions must agree")
        if any(not isinstance(key, str) or not key for key in self.path_ids):
            raise InputError("learning rows need nonempty path IDs")
        if len(self.seeds) != len(set(self.path_ids)) or len(set(self.seeds)) != len(self.seeds):
            raise InputError("one unique seed is required per distinct path in first-occurrence order")
        for seed in self.seeds:
            _integer(seed, "path seed", 0)
        previous: dict[str, datetime] = {}
        for row, (decision, target, path) in enumerate(zip(self.decision_times, self.target_times, self.path_ids)):
            timestamp(decision, "decision time")
            timestamp(target, "target time")
            days = (target-decision).total_seconds()/86400
            if target <= decision or abs(self.horizon_days[row]-days) > 1e-12:
                raise InputError("horizon_days must equal positive target-minus-decision calendar days")
            if path in previous and decision <= previous[path]:
                raise InputError("rows must move forward in decision time within each path")
            previous[path] = decision

    def manifest(self) -> dict:
        """Small reproducibility record; includes no simulator volatility/correlation."""
        utc = lambda value: value.astimezone(timezone.utc).isoformat()
        metadata = dict(stock_ids=self.stock_ids, lags=self.lags, split=self.split,
                        decision_times=tuple(map(utc, self.decision_times)),
                        target_times=tuple(map(utc, self.target_times)), path_ids=self.path_ids, seeds=self.seeds,
                        feature_version=FEATURE_VERSION)
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        for array in (self.features, self.log_returns, self.horizon_days):
            digest.update(array.astype("<f8", copy=False).tobytes())
        return dict(split=self.split, rows=len(self.features), paths=len(self.seeds),
                    stock_ids=self.stock_ids, lags=self.lags, feature_count=self.features.shape[1],
                    feature_version=FEATURE_VERSION, dataset_sha256=digest.hexdigest(),
                    decision_start=utc(min(self.decision_times)), decision_end=utc(max(self.decision_times)),
                    target_start=utc(min(self.target_times)), target_end=utc(max(self.target_times)),
                    path_ids=tuple(dict.fromkeys(self.path_ids)), seeds=self.seeds,
                    seed_namespace=SEED_NAMESPACE,
                    seed_derivation="SHA256(JSON compact [namespace, master_seed, '<split>-path', path_index]), first128 bits big-endian",
                    target_units="Joint log returns over target-minus-decision calendar days",
                    feature_normalization="Raw features only; any fitted scaler belongs to model training")


def generate_learning_data(stock_ids: Sequence[str], initial_spots: Sequence[float],
                           volatilities: Sequence[float], correlation: Sequence[Sequence[float]], *,
                           split: str, paths: int, start: datetime, days: int = 20,
                           lags: int = 3, master_seed: int = 20260920) -> TrainingData:
    """Generate paths*days labeled rows without valuation or simulator-parameter features.

    ``start`` selects the first UTC weekday date; its initial observation is at
    16:00 UTC. As in the existing fixture, each of the following ``days`` business
    days releases an open at 09:00 and close at 16:00. Each close before the final
    close is a decision; its label is the next business close (24 or 72 hours).
    """
    stocks = _stock_ids(stock_ids)
    _integer(paths, "paths", 1)
    _integer(days, "days", 1)
    _integer(lags, "lags", 0)
    _integer(master_seed, "master seed", 0)
    if split not in {"train", "validation"}:
        raise InputError("learning split must be 'train' or 'validation'")
    timestamp(start, "learning start")
    day = start.astimezone(timezone.utc).replace(hour=16, minute=0, second=0, microsecond=0)
    if day.weekday() >= 5:
        raise InputError("learning start must select a UTC weekday")
    times = [day]
    for _ in range(days):
        day = _next_business_date(day)
        times.extend((day.replace(hour=9), day.replace(hour=16)))
    features, targets, horizons, decision_times, target_times, path_ids, seeds = [], [], [], [], [], [], []
    for path_index in range(paths):
        path_id = f"{split}/{path_index:05d}"
        seed = experiment_seed(f"{split}-path", path_index, master_seed)
        seeds.append(seed)
        frames = tuple(market_path(stocks, initial_spots, volatilities, correlation, times, seed))
        history = []
        for index, (time, spots) in enumerate(frames):
            history.extend(Observation(f"{path_id}/{index}/{stock}", stock, spots[stock], time, time) for stock in stocks)
            if time.hour != 16 or index == len(frames)-1:
                continue
            target_time, target_spots = frames[index+2]
            vector, _ = conditioning_features(stocks, history, spots, time, target_time, lags=lags)
            features.append(vector)
            targets.append([log(target_spots[stock])-log(spots[stock]) for stock in stocks])
            horizons.append((target_time-time).total_seconds()/86400)
            decision_times.append(time)
            target_times.append(target_time)
            path_ids.append(path_id)
    return TrainingData(stocks, lags, np.asarray(features), np.asarray(targets), np.asarray(horizons),
                        tuple(decision_times), tuple(target_times), tuple(path_ids), split, tuple(seeds))
