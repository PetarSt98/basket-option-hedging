"""Joint multi-asset VolGAN adaptation, independently implemented from the paper.

One conditional GAN generates all asset returns and log-IV changes together.
No Gaussian-model fallback, fitted standardization, arbitrage training penalty,
or calendar-time rescaling. See docs/volgan_sources.md for source discrepancies.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import InputError, number, timestamp
from .replay import jsonable, write_json
from .surfaces import PAPER_GRID, SurfaceGrid, flatten_surface, unflatten_surface

try:
    import torch
    from torch import nn
except ImportError as error:
    raise InputError("VolGAN requires the optional dependency: install basket-hedging[learn]") from error


MODEL_VERSION = "multi-asset-VolGAN-adaptation-v1"
UPSTREAM_COMMIT = "a93f63362f2540954743ceded4f11ca1f14a9e6b"
RETURN_SCALE = float(np.sqrt(252))


def _array(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InputError(f"{name} must contain numbers") from error
    if not np.isfinite(result).all():
        raise InputError(f"{name} must be finite")
    return result


def conditioning_features(log_returns: np.ndarray, surfaces: np.ndarray,
                          grid: SurfaceGrid = PAPER_GRID) -> np.ndarray:
    """Use only returns through the decision and its current IV surfaces.

    Asset-major blocks: annualized latest return, previous return, 21-session
    realized volatility, log IV (moneyness varies fastest within each maturity).
    """
    returns, iv = _array(log_returns, "history returns"), _array(surfaces, "current IV")
    if returns.ndim != 2 or len(returns) < 21 or iv.shape != (returns.shape[1], *grid.shape):
        raise InputError("conditioning requires >=21 joint returns and one surface per asset")
    if np.any(iv <= 0):
        raise InputError("implied volatility must be positive")
    realized = np.sqrt(252 / 21 * np.sum(returns[-21:] ** 2, axis=0))
    blocks = np.column_stack((RETURN_SCALE * returns[-1], RETURN_SCALE * returns[-2],
                              realized, np.log(flatten_surface(iv, grid))))
    return blocks.reshape(-1)


def transition_target(spots: np.ndarray, next_spots: np.ndarray, surfaces: np.ndarray,
                      next_surfaces: np.ndarray, grid: SurfaceGrid = PAPER_GRID) -> np.ndarray:
    """Joint next-observation labels: annualized returns and log-IV increments."""
    current, future = _array(spots, "spots"), _array(next_spots, "next spots")
    iv, future_iv = _array(surfaces, "IV"), _array(next_surfaces, "next IV")
    if current.ndim != 1 or future.shape != current.shape or iv.shape != (len(current), *grid.shape):
        raise InputError("transition dimensions differ")
    if future_iv.shape != iv.shape or any(np.any(x <= 0) for x in (current, future, iv, future_iv)):
        raise InputError("transition prices and IV must be positive with matching dimensions")
    return np.column_stack((RETURN_SCALE * (np.log(future) - np.log(current)),
                            flatten_surface(np.log(future_iv) - np.log(iv), grid))).reshape(-1)


def reconstruct_state(condition: np.ndarray, spots: np.ndarray, outputs: np.ndarray,
                      grid: SurfaceGrid = PAPER_GRID) -> tuple[np.ndarray, np.ndarray]:
    """Invert training transforms without clipping, rescaling time, or reweighting."""
    current = _array(spots, "current spots")
    features, draws = _array(condition, "condition"), _array(outputs, "generator output")
    if current.ndim != 1 or not len(current) or np.any(current <= 0):
        raise InputError("current spots must be a positive vector")
    assets = len(current)
    if features.shape != (assets * (grid.size + 3),) or draws.ndim != 2 or draws.shape[1] != assets * (grid.size + 1):
        raise InputError("generated state dimensions differ from condition/grid")
    blocks = draws.reshape(len(draws), assets, grid.size + 1)
    log_iv = features.reshape(assets, grid.size + 3)[:, 3:]
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            future_spots = current * np.exp(blocks[:, :, 0] / RETURN_SCALE)
            future_iv = unflatten_surface(np.exp(log_iv + blocks[:, :, 1:]), grid)
    except FloatingPointError as error:
        raise InputError("VolGAN reconstruction overflow/underflow; generated state rejected") from error
    return future_spots, future_iv


def build_training_pairs(spots: np.ndarray, surfaces: np.ndarray, times: Sequence[datetime],
                         grid: SurfaceGrid = PAPER_GRID, *, cutoff: datetime
                         ) -> tuple[np.ndarray, np.ndarray, tuple[datetime, ...], tuple[datetime, ...]]:
    """Build chronologically aligned next-session pairs, using no post-cutoff values.

    Caller must supply consecutive common exchange sessions; this function does
    not guess an exchange holiday calendar from dates or interpolate missing days.
    """
    timestamp(cutoff, "training cutoff")
    for time in times:
        timestamp(time, "observation time")
    if any(a >= b for a, b in zip(times, times[1:])):
        raise InputError("training observations must be strictly chronological")
    prices, iv = np.asarray(spots), np.asarray(surfaces)
    if prices.ndim != 2 or len(prices) != len(times) or iv.shape != (len(times), prices.shape[1], *grid.shape):
        raise InputError("joint training series dimensions differ")
    count = sum(time <= cutoff for time in times)
    prices, iv = _array(prices[:count], "training spots"), _array(iv[:count], "training IV")
    if count < 23 or np.any(prices <= 0) or np.any(iv <= 0):
        raise InputError("training needs >=23 positive joint observations before the cutoff")
    returns = np.diff(np.log(prices), axis=0)
    conditions, targets, decision_times, target_times = [], [], [], []
    for index in range(21, count - 1):
        conditions.append(conditioning_features(returns[index-21:index], iv[index], grid))
        targets.append(transition_target(prices[index], prices[index+1], iv[index], iv[index+1], grid))
        decision_times.append(times[index])
        target_times.append(times[index+1])
    return np.asarray(conditions), np.asarray(targets), tuple(decision_times), tuple(target_times)


@dataclass(frozen=True)
class VolGANConfig:
    noise_dim: int = 32
    hidden_dim: int = 16
    calibration_epochs: int = 25
    epochs: int = 10_000
    batch_size: int = 100
    learning_rate: float = .0001
    seed: int = 1729

    def __post_init__(self) -> None:
        for name in ("noise_dim", "hidden_dim", "calibration_epochs", "epochs", "batch_size", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 1):
                raise InputError(f"{name} must be a positive integer (seed may be zero)")
        if self.seed >= 2**63:
            raise InputError("seed must be less than 2**63")
        number(self.learning_rate, "learning rate")
        if self.learning_rate <= 0:
            raise InputError("learning rate must be positive")


class VolGANGenerator(nn.Module):
    """Published Softplus/Softplus/affine generator, jointly across all assets."""

    def __init__(self, condition_size: int, output_size: int, config: VolGANConfig):
        super().__init__()
        width = config.hidden_dim
        self.layers = nn.Sequential(nn.Linear(condition_size + config.noise_dim, width), nn.Softplus(),
                                    nn.Linear(width, 2 * width), nn.Softplus(), nn.Linear(2 * width, output_size))

    def forward(self, condition: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((noise, condition), dim=-1))


class VolGANDiscriminator(nn.Module):
    """One joint real/fake classifier; no per-asset independent discriminators."""

    def __init__(self, condition_size: int, output_size: int, config: VolGANConfig):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(condition_size + output_size, config.hidden_dim),
                                    nn.Softplus(), nn.Linear(config.hidden_dim, 1), nn.Sigmoid())

    def forward(self, condition: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((condition, outcome), dim=-1))


def smoothness_penalties(log_surfaces: torch.Tensor, grid: SurfaceGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper equations 11-12: sum over grid/asset axes, average across scenarios."""
    if log_surfaces.ndim != 4 or tuple(log_surfaces.shape[-2:]) != grid.shape:
        raise InputError("smoothness input requires (batch, assets, moneyness, maturity)")
    dm = torch.as_tensor(np.diff(grid.moneyness), dtype=log_surfaces.dtype, device=log_surfaces.device)
    dt = torch.as_tensor(np.diff(grid.maturities), dtype=log_surfaces.dtype, device=log_surfaces.device)
    m_slopes = torch.diff(log_surfaces, dim=-2) / dm[:, None]
    t_slopes = torch.diff(log_surfaces, dim=-1) / dt
    return m_slopes.square().sum((1, 2, 3)).mean(), t_slopes.square().sum((1, 2, 3)).mean()


def _state_hash(module: nn.Module) -> str:
    result = sha256()
    for name, value in module.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().numpy().astype("<f4").tobytes())
    return result.hexdigest()


@dataclass
class VolGANModel:
    asset_ids: tuple[str, ...]
    grid: SurfaceGrid
    config: VolGANConfig
    generator: VolGANGenerator
    discriminator: VolGANDiscriminator
    available_at: datetime
    trained_through: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)
    training_history: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        self.asset_ids = tuple(self.asset_ids)
        if not self.asset_ids or len(set(self.asset_ids)) != len(self.asset_ids) or any(not isinstance(x, str) or not x for x in self.asset_ids):
            raise InputError("VolGAN asset IDs must be unique nonempty strings")
        timestamp(self.available_at, "model availability")
        timestamp(self.trained_through, "training cutoff")
        if self.trained_through > self.available_at:
            raise InputError("training cutoff exceeds model availability")
        self.generator.eval()
        self.discriminator.eval()

    def sample(self, condition: np.ndarray, spots: np.ndarray,
               normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One next-observation transition; supplied shared noise controls randomness."""
        features, noise = _array(condition, "condition"), _array(normals, "latent draws")
        size = len(self.asset_ids) * (self.grid.size + 3)
        if features.shape != (size,) or noise.ndim != 2 or noise.shape[1] != self.config.noise_dim or len(noise) < 1:
            raise InputError("VolGAN condition/noise dimensions are invalid")
        if np.asarray(spots).shape != (len(self.asset_ids),):
            raise InputError("spot order must match checkpoint asset order")
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).repeat(len(noise), 1)
            output = self.generator(x, torch.tensor(noise, dtype=torch.float32)).numpy()
        return reconstruct_state(features, spots, output, self.grid)

    def save(self, path: str | Path) -> Path:
        state = lambda module: {key: value.detach().cpu().tolist() for key, value in module.state_dict().items()}
        write_json(Path(path), dict(model_version=MODEL_VERSION, upstream_commit=UPSTREAM_COMMIT,
                   asset_ids=self.asset_ids, grid=asdict(self.grid), config=asdict(self.config),
                   generator=state(self.generator), discriminator=state(self.discriminator),
                   available_at=self.available_at, trained_through=self.trained_through,
                   metadata=self.metadata, training_history=self.training_history))
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> "VolGANModel":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if raw.get("model_version") != MODEL_VERSION or raw.get("upstream_commit") != UPSTREAM_COMMIT:
                raise InputError("not a supported VolGAN checkpoint; no Gaussian fallback")
            config, grid = VolGANConfig(**raw["config"]), SurfaceGrid(**raw["grid"])
            assets = tuple(raw["asset_ids"])
            with torch.random.fork_rng(devices=[]):
                generator = VolGANGenerator(len(assets)*(grid.size+3), len(assets)*(grid.size+1), config)
                discriminator = VolGANDiscriminator(len(assets)*(grid.size+3), len(assets)*(grid.size+1), config)
            for module, key in ((generator, "generator"), (discriminator, "discriminator")):
                module.load_state_dict({name: torch.tensor(_array(value, "checkpoint weights"), dtype=torch.float32)
                                        for name, value in raw[key].items()}, strict=True)
            return cls(assets, grid, config, generator, discriminator,
                       datetime.fromisoformat(raw["available_at"]), datetime.fromisoformat(raw["trained_through"]),
                       raw["metadata"], tuple(raw["training_history"]))
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise InputError(f"invalid VolGAN checkpoint: {error}") from error


def fit_volgan(conditions: np.ndarray, targets: np.ndarray, *, asset_ids: Sequence[str],
               available_at: datetime, trained_through: datetime, grid: SurfaceGrid = PAPER_GRID,
               config: VolGANConfig = VolGANConfig(), metadata: Mapping[str, Any] | None = None,
               progress: Callable[[Mapping[str, Any]], None] | None = None,
               checkpoint_epochs: Sequence[int] = (),
               checkpoint: Callable[[VolGANModel], None] | None = None) -> VolGANModel:
    """Train BCE calibration, restart from identical weights, then train full loss.

    Returns the final epoch; optional detached snapshots permit selection by a
    caller's predeclared validation rule without changing the training updates.
    Input rows must already be chronological TRAIN transitions from the loader.
    The shuffle applies once to whole joint rows, never independently to assets.
    """
    x, y = _array(conditions, "training conditions"), _array(targets, "training targets")
    assets = tuple(asset_ids)
    if not assets or len(set(assets)) != len(assets) or any(not isinstance(a, str) or not a for a in assets):
        raise InputError("asset IDs must be unique nonempty strings")
    timestamp(available_at, "model availability")
    timestamp(trained_through, "training cutoff")
    if trained_through > available_at:
        raise InputError("training information postdates model availability")
    if x.ndim != 2 or x.shape[1] != len(assets)*(grid.size+3) or y.shape != (len(x), len(assets)*(grid.size+1)) or not len(x):
        raise InputError("training matrices must contain matched joint asset transitions")
    if np.any(x.reshape(len(x), len(assets), grid.size+3)[:, :, 2] < 0):
        raise InputError("realized volatility must be nonnegative")
    checkpoints = tuple(checkpoint_epochs)
    if (any(isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= config.epochs
            for epoch in checkpoints) or tuple(sorted(set(checkpoints))) != checkpoints
            or bool(checkpoints) != (checkpoint is not None)
            or (checkpoint is not None and not callable(checkpoint))):
        raise InputError("checkpoint epochs require an increasing unique in-budget schedule and callback")
    extra = json.loads(json.dumps(jsonable(metadata or {}), allow_nan=False))
    start, old_threads = perf_counter(), torch.get_num_threads()
    history, ratios = [], []
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            generator = VolGANGenerator(x.shape[1], y.shape[1], config)
            discriminator = VolGANDiscriminator(x.shape[1], y.shape[1], config)
            initial_g, initial_d = deepcopy(generator.state_dict()), deepcopy(discriminator.state_dict())
            initial_hashes = [_state_hash(generator), _state_hash(discriminator)]
            train_x, train_y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
            random = torch.Generator(device="cpu").manual_seed(config.seed + 1)
            bce = nn.BCELoss()
            weights = (0., 0.)

            def snapshot(epoch: int, *, detached: bool) -> VolGANModel:
                training_metadata = dict(extra, model_identifier=MODEL_VERSION, label="multi-asset VolGAN adaptation",
                    upstream_commit=UPSTREAM_COMMIT,
                    architecture=f"Joint G: Softplus {config.hidden_dim}, Softplus {2*config.hidden_dim}, affine; D: Softplus {config.hidden_dim}, sigmoid",
                    feature_order="asset-major: sqrt252*latest_return, sqrt252*previous_return, RV21, maturity-major logIV",
                    target_order="asset-major: sqrt252*next_return, maturity-major logIV_change", annualization=252,
                    normalization="none", transition="one consecutive common exchange session; no dt rescaling",
                    objective="0.5 generator BCE + calibrated log-surface Sobolev penalties; discriminator mean real/fake BCE",
                    arbitrage_penalty_in_training=False, scenario_reweighting=False, optimizer="RMSprop",
                    smoothness_m=weights[0], smoothness_t=weights[1], restart_after_calibration=True,
                    checkpoint_selection="scheduled snapshot" if detached else "fixed final training epoch",
                    selected_epoch=epoch, checkpoint_epochs=checkpoints,
                    history_sampling="first, final, every 100th epoch and scheduled checkpoints per phase",
                    training_rows=len(x), training_arrays_sha256=sha256(x.astype("<f8").tobytes()+y.astype("<f8").tobytes()).hexdigest(),
                    initial_generator_sha256=initial_hashes[0], initial_discriminator_sha256=initial_hashes[1],
                    final_generator_sha256=_state_hash(generator), final_discriminator_sha256=_state_hash(discriminator),
                    torch_version=str(torch.__version__), numpy_version=np.__version__, device="cpu", dtype="float32",
                    training_seconds=perf_counter()-start)
                return VolGANModel(assets, grid, config,
                    deepcopy(generator) if detached else generator,
                    deepcopy(discriminator) if detached else discriminator,
                    available_at, trained_through,
                    deepcopy(training_metadata) if detached else training_metadata,
                    tuple(deepcopy(history)) if detached else tuple(history))

            for phase, epochs in (("calibration", config.calibration_epochs), ("training", config.epochs)):
                # Paper section 3.4 restarts networks AND optimizers after calibration.
                generator.load_state_dict(initial_g)
                discriminator.load_state_dict(initial_d)
                g_opt = torch.optim.RMSprop(generator.parameters(), lr=config.learning_rate)
                d_opt = torch.optim.RMSprop(discriminator.parameters(), lr=config.learning_rate)
                generator.train()
                discriminator.train()
                for epoch in range(1, epochs + 1):
                    order = torch.randperm(len(x), generator=random)
                    total = np.zeros(4)
                    for first in range(0, len(x), config.batch_size):
                        index = order[first:first + config.batch_size]
                        condition, real = train_x[index], train_y[index]
                        noise = torch.randn(len(index), config.noise_dim, generator=random)
                        fake = generator(condition, noise)
                        d_opt.zero_grad(set_to_none=True)
                        p_real, p_fake = discriminator(condition, real), discriminator(condition, fake.detach())
                        d_loss = .5*(bce(p_real, torch.ones_like(p_real)) + bce(p_fake, torch.zeros_like(p_fake)))
                        d_loss.backward()
                        d_opt.step()
                        g_opt.zero_grad(set_to_none=True)
                        fake = generator(condition, torch.randn(len(index), config.noise_dim, generator=random))
                        p_fake = discriminator(condition, fake)
                        adversarial = .5*bce(p_fake, torch.ones_like(p_fake))
                        current = condition.reshape(len(index), len(assets), grid.size+3)[:, :, 3:]
                        future = current + fake.reshape(len(index), len(assets), grid.size+1)[:, :, 1:]
                        log_surfaces = future.reshape(len(index), len(assets), len(grid.maturities), len(grid.moneyness)).transpose(-1, -2)
                        m_loss, t_loss = smoothness_penalties(log_surfaces, grid)
                        if phase == "calibration":
                            norms = []
                            for loss in (adversarial, m_loss, t_loss):
                                gradients = torch.autograd.grad(loss, tuple(generator.parameters()), retain_graph=True)
                                norms.append(float(torch.sqrt(sum(g.square().sum() for g in gradients))))
                            if not np.isfinite(norms).all() or min(norms) <= 0:
                                raise InputError("degenerate gradient-norm calibration; training stopped")
                            ratios.append((norms[0]/norms[1], norms[0]/norms[2]))
                        g_loss = adversarial + weights[0]*m_loss + weights[1]*t_loss
                        if not torch.isfinite(d_loss + g_loss):
                            raise InputError("VolGAN produced a nonfinite loss; training stopped")
                        g_loss.backward()
                        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in generator.parameters()):
                            raise InputError("VolGAN produced nonfinite gradients; training stopped")
                        g_opt.step()
                        total += len(index)*np.array([d_loss.item(), adversarial.item(), m_loss.item(), t_loss.item()])
                    record = dict(phase=phase, epoch=epoch, discriminator_loss=total[0]/len(x),
                                  generator_bce=total[1]/len(x), m_smoothness=total[2]/len(x), t_smoothness=total[3]/len(x))
                    if epoch == 1 or epoch == epochs or epoch % 100 == 0 or (phase == "training" and epoch in checkpoints):
                        history.append(record)
                    if progress is not None:
                        progress(record)
                    if phase == "training" and epoch in checkpoints:
                        # Validation cannot consume training RNG or mutate live networks.
                        with torch.random.fork_rng(devices=[]):
                            checkpoint(snapshot(epoch, detached=True))
                if phase == "calibration":
                    weights = tuple(float(v) for v in np.mean(ratios, axis=0))
            model = snapshot(config.epochs, detached=False)
    finally:
        torch.set_num_threads(old_threads)
    return model


def prepared_training_pairs(data: Mapping[str, np.ndarray], grid: SurfaceGrid, *, split: str
                            ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Select joint dated windows without bridging missing common sessions/surfaces.

    Returned indices identify target dates. Gaussian and VolGAN use the same
    declared splits and valid windows; both endpoints belong to the same split.
    """
    from .historical_training import historical_training_indices

    spots, surfaces = data["close"], data["iv"]
    conditions, targets, indices = [], [], []
    for decision in historical_training_indices(data, split=split):
        target = decision + 1
        returns = np.diff(np.log(spots[target-22:target]), axis=0)
        conditions.append(conditioning_features(returns, surfaces[target-1], grid))
        targets.append(transition_target(spots[target-1], spots[target], surfaces[target-1], surfaces[target], grid))
        indices.append(target)
    if not indices:
        raise InputError("no complete 23-observation joint windows in the requested split")
    return np.asarray(conditions), np.asarray(targets), tuple(indices)


def validate_prepared_volgan(model: VolGANModel, data: Mapping[str, np.ndarray], *,
                            samples: int = 256, seed: int = 1730, year: int | None = None,
                            sampler: Callable[[np.ndarray, int, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None
                            ) -> dict:
    """Joint validation diagnostics; an optional sampler compares existing references.

    Year selects target dates, never hedge outcomes. Earlier rows consume the
    same noise draws as combined validation, but are not evaluated. Truncation
    precedes feature construction so later observations cannot affect selection.
    """
    from .generator_validation import energy_score

    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise InputError("validation scenario count must be an integer >=2")
    if year is not None and (isinstance(year, bool) or year not in (2018, 2019)):
        raise InputError("validation year must be 2018 or 2019")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise InputError("validation seed must be a nonnegative integer")
    count = int(np.searchsorted(data["dates"], f"{year or 2019}-12-31", side="right"))
    data = {key: value[:count] if value.ndim and len(value) == len(data["dates"]) else value
            for key, value in data.items()}
    if tuple(data["symbols"]) != model.asset_ids:
        raise InputError("validation asset order must match the checkpoint")
    features, targets, indices = prepared_training_pairs(data, model.grid, split="validation")
    random = np.random.Generator(np.random.PCG64(seed))
    true_returns = targets.reshape(len(targets), len(model.asset_ids), model.grid.size+1)[:, :, 0]/RETURN_SCALE
    predicted, coverages, sample_stds, covariances = [], [], [], []
    actuals, selected, widths, interval_scores, energies = [], [], [], [], []
    iv_squared_errors, iv_means, iv_stds, iv_actuals, return_iv_correlations = [], [], [], [], []
    pairs = [(i, j) for i in range(len(model.asset_ids)) for j in range(i+1, len(model.asset_ids))]
    pair_correlations, spreads = [], []

    def interval(draws, actual):
        low, high = np.quantile(draws, [.05, .95], axis=0)
        width = high - low
        return ((actual >= low) & (actual <= high), width,
                width + 20 * (np.maximum(low-actual, 0) + np.maximum(actual-high, 0)))

    def correlation(left, right):
        return float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else np.nan

    def correlation_summary(rows):
        values = np.asarray(rows)
        return [dict(quantiles_05_50_95=np.quantile(column[np.isfinite(column)], [.05, .5, .95]).tolist()
                     if np.isfinite(column).any() else [None]*3,
                     undefined_states=int((~np.isfinite(column)).sum())) for column in values.T]

    for feature, target, actual in zip(features, indices, true_returns):
        normals = random.standard_normal((samples, model.config.noise_dim))
        if year is not None and not str(data["dates"][target]).startswith(str(year)):
            continue
        spots, surfaces = (model.sample(feature, data["close"][target-1], normals) if sampler is None
                           else sampler(feature, target, normals))
        if (spots.shape != (samples, len(model.asset_ids))
                or surfaces.shape != (samples, len(model.asset_ids), *model.grid.shape)
                or not np.isfinite(spots).all() or not np.isfinite(surfaces).all()
                or np.any(spots <= 0) or np.any(surfaces <= 0)):
            raise InputError("validation samples must be finite positive joint prices and surfaces")
        returns = np.log(spots/data["close"][target-1])
        actuals.append(actual)
        selected.append(target)
        predicted.append(returns.mean(axis=0))
        covered, width, score = interval(returns, actual)
        coverages.append(covered)
        widths.append(width)
        interval_scores.append(score)
        energies.append(energy_score(actual, returns))
        sample_stds.append(returns.std(axis=0, ddof=1))
        covariances.append(np.cov(returns, rowvar=False))
        pair_correlations.append([correlation(returns[:, i], returns[:, j]) for i, j in pairs])
        log_iv = np.log(surfaces)
        iv_squared_errors.append(np.mean((log_iv.mean(axis=0)-np.log(data["iv"][target]))**2, axis=(-2, -1)))
        changes = (log_iv - np.log(data["iv"][target-1])).mean(axis=(-2, -1))
        iv_means.append(changes.mean(axis=0))
        iv_stds.append(changes.std(axis=0, ddof=1))
        iv_actuals.append((np.log(data["iv"][target])-np.log(data["iv"][target-1])).mean(axis=(-2, -1)))
        return_iv_correlations.append([correlation(returns[:, i], changes[:, i]) for i in range(len(model.asset_ids))])
        if "QQQ" in model.asset_ids and "IWM" in model.asset_ids:
            i, j = model.asset_ids.index("QQQ"), model.asset_ids.index("IWM")
            spread, observed = returns[:, i]-returns[:, j], actual[i]-actual[j]
            covered, width, score = interval(spread, observed)
            spreads.append([observed, spread.mean(), spread.std(ddof=1), covered, width, score])
    if len(selected) < 2:
        raise InputError("validation segment requires at least two eligible transitions")
    predicted = np.asarray(predicted)
    true_returns = np.asarray(actuals)
    spread = np.asarray(spreads)
    conditional_mean_cov = np.cov(predicted, rowvar=False, bias=True)
    return dict(label="frozen VolGAN validation diagnostics; not hedge performance or independent test paths",
                start=str(data["dates"][selected[0]]), end=str(data["dates"][selected[-1]]), year=year,
                rows=len(selected), samples_per_condition=samples, seed=seed, asset_ids=model.asset_ids,
                energy_score=float(np.mean(energies)),
                return_mean_forecast_rmse=np.sqrt(np.mean((predicted-true_returns)**2, axis=0)).tolist(),
                return_5_95_percentile_coverage=np.mean(coverages, axis=0).tolist(),
                return_interval_width_90=np.mean(widths, axis=0).tolist(),
                return_interval_score_90=np.mean(interval_scores, axis=0).tolist(),
                mean_generated_return=predicted.mean(axis=0).tolist(), observed_return_mean=true_returns.mean(axis=0).tolist(),
                conditional_return_mean_std=predicted.std(axis=0, ddof=1).tolist(),
                mean_generated_return_std=np.mean(sample_stds, axis=0).tolist(),
                observed_return_std=true_returns.std(axis=0, ddof=1).tolist(),
                mean_generated_return_covariance=np.mean(covariances, axis=0).tolist(),
                covariance_of_conditional_means=conditional_mean_cov.tolist(),
                generated_mixture_covariance=(np.mean(covariances, axis=0)*(samples-1)/samples+conditional_mean_cov).tolist(),
                observed_return_covariance=np.cov(true_returns, rowvar=False).tolist(),
                generated_pairwise_correlations={f"{model.asset_ids[i]}/{model.asset_ids[j]}": stats
                    for (i, j), stats in zip(pairs, correlation_summary(pair_correlations))},
                spread_qqq_minus_iwm=None if not len(spread) else dict(
                    observed_mean=float(spread[:, 0].mean()), observed_std=float(spread[:, 0].std(ddof=1)),
                    generated_mean=float(spread[:, 1].mean()), conditional_mean_std=float(spread[:, 1].std(ddof=1)),
                    mean_conditional_std=float(spread[:, 2].mean()), coverage_90=float(spread[:, 3].mean()),
                    interval_width_90=float(spread[:, 4].mean()), interval_score_90=float(spread[:, 5].mean())),
                log_iv_mean_forecast_rmse=np.sqrt(np.mean(iv_squared_errors, axis=0)).tolist(),
                surface_changes=dict(generated_mean=np.mean(iv_means, axis=0).tolist(),
                    mean_conditional_std=np.mean(iv_stds, axis=0).tolist(),
                    conditional_mean_std=np.std(iv_means, axis=0, ddof=1).tolist(),
                    observed_mean=np.mean(iv_actuals, axis=0).tolist(), observed_std=np.std(iv_actuals, axis=0, ddof=1).tolist(),
                    return_mean_log_iv_change_correlations=correlation_summary(return_iv_correlations)),
                definitions=dict(energy_score="Euclidean empirical joint energy score; lower is better, raw log-return units",
                    interval_score_90="width + 20 times distance outside the central 90% interval; lower is better",
                    conditional_covariance="average within-state sample covariance, not unconditional realized covariance",
                    mixture_covariance="within-state population covariance plus population covariance of conditional means",
                    surface_changes="per-asset grid-average log-IV changes; correlations computed within each state"),
                checkpoint_selection=model.metadata.get("checkpoint_selection", "unspecified"))


def train_prepared_volgan(prepared_directory: str | Path, output_path: str | Path, *,
                          config: VolGANConfig = VolGANConfig()) -> VolGANModel:
    """Train the declared historical adaptation and save model plus validation record."""
    from .etf_data import load_prepared
    from .historical_training import MODEL_AVAILABLE, close_time

    data, manifest = load_prepared(prepared_directory)
    grid = SurfaceGrid(tuple(data["moneyness"]), tuple(data["maturities"]))
    split = manifest["split"]
    x, y, indices = prepared_training_pairs(data, grid, split="train")
    cutoff = close_time(str(data["dates"][indices[-1]]))
    began = perf_counter()

    def progress(row):
        if row["epoch"] == 1 or row["epoch"] % 100 == 0 or row["epoch"] == config.calibration_epochs and row["phase"] == "calibration":
            print(f"VolGAN {row['phase']} epoch {row['epoch']}; elapsed {perf_counter()-began:.1f}s", flush=True)

    model = fit_volgan(x, y, asset_ids=tuple(data["symbols"]), grid=grid, config=config,
                       available_at=MODEL_AVAILABLE, trained_through=cutoff, progress=progress,
                       metadata=dict(dataset=manifest["data_kind"], prepared_sha256=manifest["prepared_sha256"],
                                     source_file_hashes=manifest["source_file_hashes"], split=split,
                                     first_training_target=str(data["dates"][indices[0]]),
                                     final_training_target=str(data["dates"][indices[-1]]),
                                     excluded_windows="Any missing surface or nonconsecutive session within a 23-observation window",
                                     time_labels="Declared 16:00 America/New_York closes; not observed release timestamps"))
    destination = model.save(output_path)
    diagnostic = validate_prepared_volgan(model, data)
    write_json(destination.with_name(destination.stem+"_validation.json"), diagnostic)
    print(f"Saved {destination}; trained on {len(x)} joint historical transitions.", flush=True)
    return model
