"""Small conditional Gaussian density model; NumPy inference, optional CPU training.

This is deliberately not VolGAN. The network predicts a log-return mean rate and
full Cholesky factor from released-state features. All fitted preprocessing uses
training data; validation selects an epoch, never a hedge or engine parameter.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from math import log, pi
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

from .contracts import InputError, number, readonly, timestamp
from .replay import jsonable, write_json

if TYPE_CHECKING:
    from .learning_data import TrainingData


MODEL_VERSION = "conditional-neural-gaussian-v1"
RETURN_SCALE = 0.01
MIN_CHOLESKY_DIAGONAL = 1e-4
WEIGHT_DECAY = 1e-4


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InputError(f"{name} must be an integer >= {minimum}")


def _array(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InputError(f"{name} must be numeric") from error
    if not np.isfinite(result).all():
        raise InputError(f"{name} must be finite")
    return np.frombuffer(result.tobytes(), dtype=np.float64).reshape(result.shape)


def _json_metadata(value: Any) -> Any:
    try:
        return readonly(json.loads(json.dumps(jsonable(value), allow_nan=False)))
    except (TypeError, ValueError) as error:
        raise InputError("model metadata must be finite JSON values") from error


@dataclass(frozen=True)
class ConditionalGaussianModel:
    stock_ids: tuple[str, ...]
    lags: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    available_at: datetime
    trained_through: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)
    training_history: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.stock_ids, (tuple, list)):
            raise InputError("model stock IDs must be an explicit sequence")
        ids = tuple(self.stock_ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(key, str) or not key for key in ids):
            raise InputError("model stock IDs must be unique nonempty strings")
        _integer(self.lags, "history lags")
        timestamp(self.available_at, "model availability")
        timestamp(self.trained_through, "model training information cutoff")
        if self.trained_through > self.available_at:
            raise InputError("model information cutoff exceeds its availability")
        mean, scale = _array(self.feature_mean, "feature mean"), _array(self.feature_scale, "feature scale")
        count = len(ids)
        features = count + count * self.lags + 2 * self.lags + 1
        if mean.shape != (features,) or scale.shape != mean.shape or np.any(scale <= 0):
            raise InputError("model feature normalization has invalid dimensions/scales")
        weights = tuple(_array(value, "model weights") for value in self.weights)
        biases = tuple(_array(value, "model biases") for value in self.biases)
        output = count + count * (count + 1) // 2
        if len(weights) != 3 or len(biases) != 3 or weights[0].ndim != 2:
            raise InputError("model requires exactly three dense layers")
        hidden = weights[0].shape[0]
        dimensions = ((hidden, features), (hidden, hidden), (output, hidden))
        if hidden < 1 or any(w.shape != shape or b.shape != (shape[0],)
                             for w, b, shape in zip(weights, biases, dimensions)):
            raise InputError("model layer dimensions are inconsistent")
        if not isinstance(self.metadata, Mapping) or any(not isinstance(row, Mapping) for row in self.training_history):
            raise InputError("model metadata/history must contain mappings")
        for name, value in (("stock_ids", ids), ("feature_mean", mean), ("feature_scale", scale),
                            ("weights", weights), ("biases", biases),
                            ("metadata", _json_metadata(self.metadata)),
                            ("training_history", _json_metadata(self.training_history))):
            object.__setattr__(self, name, value)

    def _parameters(self, features: np.ndarray, horizon_days: float) -> tuple[np.ndarray, np.ndarray]:
        values = _array(features, "conditioning features")
        if values.shape != self.feature_mean.shape:
            raise InputError("conditioning features must be a single model-sized vector")
        number(horizon_days, "scenario horizon in calendar days")
        if horizon_days <= 0:
            raise InputError("scenario horizon must be positive")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                output = (values - self.feature_mean) / self.feature_scale
                for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                    output = weight @ output + bias
                    if index < 2:
                        output = np.tanh(output)
                count = len(self.stock_ids)
                lower = np.zeros((count, count))
                lower[np.tril_indices(count)] = output[count:]
                diagonal = np.diag_indices(count)
                lower[diagonal] = np.logaddexp(0, lower[diagonal]) + MIN_CHOLESKY_DIAGONAL
                mean = horizon_days * RETURN_SCALE * output[:count]
                factor = np.sqrt(horizon_days) * RETURN_SCALE * lower
        except FloatingPointError as error:
            raise InputError("model inference produced nonfinite values") from error
        if not np.isfinite(mean).all() or not np.isfinite(factor).all():
            raise InputError("model inference produced nonfinite values")
        return mean, factor

    def moments(self, features: np.ndarray, horizon_days: float) -> tuple[np.ndarray, np.ndarray]:
        """Conditional log-return mean and covariance, in unscaled return units."""
        mean, factor = self._parameters(features, horizon_days)
        with np.errstate(over="raise", invalid="raise"):
            try:
                covariance = factor @ factor.T
            except FloatingPointError as error:
                raise InputError("model covariance is nonfinite") from error
        return mean, covariance

    def sample_log_returns(self, features: np.ndarray, horizon_days: float, normals: np.ndarray) -> np.ndarray:
        """Transform supplied independent normals; inference owns no random state."""
        draws = _array(normals, "scenario normal draws")
        if draws.ndim != 2 or draws.shape[0] < 1 or draws.shape[1] != len(self.stock_ids):
            raise InputError("normal draws must have shape (N, constituents), N >= 1")
        mean, factor = self._parameters(features, horizon_days)
        try:
            with np.errstate(over="raise", invalid="raise"):
                return mean + draws @ factor.T
        except FloatingPointError as error:
            raise InputError("generated log returns are nonfinite") from error

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        write_json(destination, dict(
            model_version=MODEL_VERSION, stock_ids=self.stock_ids, lags=self.lags,
            feature_mean=self.feature_mean.tolist(), feature_scale=self.feature_scale.tolist(),
            weights=[value.tolist() for value in self.weights], biases=[value.tolist() for value in self.biases],
            available_at=self.available_at, trained_through=self.trained_through,
            metadata=self.metadata, training_history=self.training_history))
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalGaussianModel":
        def nonfinite(value: str) -> None:
            raise InputError(f"nonfinite checkpoint number {value}")

        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=nonfinite)
            expected = {"model_version", "stock_ids", "lags", "feature_mean", "feature_scale", "weights", "biases",
                        "available_at", "trained_through", "metadata", "training_history"}
            if not isinstance(raw, dict) or set(raw) != expected or raw.pop("model_version") != MODEL_VERSION:
                raise InputError("unsupported or malformed conditional Gaussian checkpoint")
            raw["available_at"] = datetime.fromisoformat(raw["available_at"])
            raw["trained_through"] = datetime.fromisoformat(raw["trained_through"])
            return cls(**raw)
        except (TypeError, ValueError, KeyError) as error:
            raise InputError(f"invalid conditional Gaussian checkpoint: {error}") from error


def fit_generator(train: "TrainingData", validation: "TrainingData", *, seed: int = 271828,
                  epochs: int = 40, hidden: int = 24, batch_size: int = 512,
                  learning_rate: float = .001, available_at: datetime) -> ConditionalGaussianModel:
    """Fit on TRAIN; select the smallest VALIDATION NLL among completed epochs.

    Loss is the Gaussian NLL of log_returns/(.01*sqrt(days)), with conditional
    mean sqrt(days)*mu and Cholesky L. Its return-unit Jacobian is parameter-free
    and omitted. Training is CPU float64, one thread, without global RNG changes.
    """
    for value, name, minimum in ((seed, "training seed", 0), (epochs, "epochs", 1),
                                 (hidden, "hidden width", 1), (batch_size, "batch size", 1)):
        _integer(value, name, minimum)
    if seed >= 2**64:
        raise InputError("training seed must fit PyTorch's unsigned 64-bit seed range")
    number(learning_rate, "learning rate")
    if learning_rate <= 0:
        raise InputError("learning rate must be positive")
    timestamp(available_at, "model availability")
    if train.split != "train" or validation.split != "validation":
        raise InputError("fitting requires declared train and validation datasets")
    if train.stock_ids != validation.stock_ids or train.lags != validation.lags:
        raise InputError("train and validation feature schemas differ")
    if set(train.path_ids) & set(validation.path_ids) or set(train.seeds) & set(validation.seeds):
        raise InputError("train and validation paths/seeds must be disjoint")
    if max(train.target_times) >= min(validation.decision_times):
        raise InputError("all training targets must precede validation decisions")
    trained_through = max(validation.target_times)
    if trained_through > available_at:
        raise InputError("validation targets were unavailable at model availability")
    try:
        import torch
    except ImportError as error:
        raise InputError("training requires the optional dependency: install basket-hedging[learn]") from error

    mean = np.mean(train.features, axis=0)
    scale = np.std(train.features, axis=0)
    scale = np.where(scale > 0, scale, 1.0)  # A constant TRAIN feature remains unscaled.
    count = len(train.stock_ids)
    outputs = count + count * (count + 1) // 2
    random = np.random.Generator(np.random.PCG64(seed))
    previous_threads = torch.get_num_threads()
    history, best_weights, best_biases = [], None, None
    best_nll, best_epoch = float("inf"), 0
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            layers = torch.nn.ModuleList([
                torch.nn.Linear(len(mean), hidden, device="cpu", dtype=torch.float64),
                torch.nn.Linear(hidden, hidden, device="cpu", dtype=torch.float64),
                torch.nn.Linear(hidden, outputs, device="cpu", dtype=torch.float64)])
            optimizer = torch.optim.Adam(layers.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
            rows, columns = torch.tril_indices(count, count, device="cpu")
            diagonal = torch.arange(count, device="cpu")

            def tensors(dataset):
                x = torch.tensor((dataset.features - mean) / scale, dtype=torch.float64, device="cpu")
                days = torch.tensor(dataset.horizon_days, dtype=torch.float64, device="cpu")
                y = torch.tensor(dataset.log_returns, dtype=torch.float64, device="cpu") / (
                    RETURN_SCALE * torch.sqrt(days[:, None]))
                return x, days, y

            train_x, train_days, train_y = tensors(train)
            validation_x, validation_days, validation_y = tensors(validation)

            def nll(x, days, y):
                output = layers[2](torch.tanh(layers[1](torch.tanh(layers[0](x)))))
                lower = torch.zeros((len(x), count, count), dtype=torch.float64, device="cpu")
                lower[:, rows, columns] = output[:, count:]
                lower[:, diagonal, diagonal] = torch.nn.functional.softplus(
                    output[:, count:][:, rows == columns]) + MIN_CHOLESKY_DIAGONAL
                residual = y - torch.sqrt(days[:, None]) * output[:, :count]
                whitened = torch.linalg.solve_triangular(lower, residual[:, :, None], upper=False).squeeze(-1)
                return (.5 * torch.sum(whitened**2, dim=1)
                        + torch.log(lower[:, diagonal, diagonal]).sum(dim=1) + .5 * count * log(2*pi)).mean()

            def evaluate(x, days, y):
                with torch.no_grad():
                    total = sum(float(nll(x[start:start+batch_size], days[start:start+batch_size], y[start:start+batch_size]))
                                * len(x[start:start+batch_size]) for start in range(0, len(x), batch_size))
                return total / len(x)

            for epoch in range(1, epochs + 1):
                order = random.permutation(len(train_x))
                for start in range(0, len(order), batch_size):
                    indices = order[start:start+batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss = nll(train_x[indices], train_days[indices], train_y[indices])
                    if not torch.isfinite(loss):
                        raise InputError("training produced a nonfinite Gaussian loss")
                    loss.backward()
                    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                           for parameter in layers.parameters()):
                        raise InputError("training produced nonfinite gradients")
                    optimizer.step()
                train_nll = evaluate(train_x, train_days, train_y)
                validation_nll = evaluate(validation_x, validation_days, validation_y)
                if not np.isfinite([train_nll, validation_nll]).all():
                    raise InputError("checkpoint evaluation produced a nonfinite Gaussian loss")
                history.append(dict(epoch=epoch, train_nll=train_nll, validation_nll=validation_nll))
                if validation_nll < best_nll:
                    best_nll, best_epoch = validation_nll, epoch
                    best_weights = tuple(layer.weight.detach().numpy().copy() for layer in layers)
                    best_biases = tuple(layer.bias.detach().numpy().copy() for layer in layers)
    finally:
        torch.set_num_threads(previous_threads)
    return ConditionalGaussianModel(
        train.stock_ids, train.lags, mean, scale, best_weights, best_biases, available_at, trained_through,
        metadata=dict(model_version=MODEL_VERSION, architecture="Dense-tanh-Dense-tanh-Gaussian full Cholesky",
                      seed=seed, epochs=epochs, hidden=hidden, batch_size=batch_size, learning_rate=learning_rate,
                      optimizer="Adam", weight_decay=WEIGHT_DECAY, dtype="float64", device="CPU", threads=1,
                      return_scale=RETURN_SCALE, minimum_cholesky_diagonal=MIN_CHOLESKY_DIAGONAL,
                      best_epoch=best_epoch, best_validation_nll=best_nll,
                      loss_units="Gaussian NLL of log return divided by .01*sqrt(calendar days)",
                      train=train.manifest(), validation=validation.manifest(),
                      numpy_version=np.__version__, torch_version=torch.__version__),
        training_history=tuple(history))
