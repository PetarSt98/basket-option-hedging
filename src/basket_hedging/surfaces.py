"""Small, explicit volatility grids; interpolated values are never quotes.

The published VolGAN grid differs from the older 10-by-8 public example.
Flattening visits all moneyness points for each maturity (maturity-major).
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .contracts import InputError, number, timestamp


@dataclass(frozen=True)
class SurfaceGrid:
    moneyness: tuple[float, ...]
    maturities: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("moneyness", "maturities"):
            values = tuple(getattr(self, name))
            if len(values) < 2:
                raise InputError(f"{name} needs at least two points")
            for value in values:
                number(value, name)
            if min(values) <= 0 or any(b <= a for a, b in zip(values, values[1:])):
                raise InputError(f"{name} must be positive and strictly increasing")
            object.__setattr__(self, name, values)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.moneyness), len(self.maturities)

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]


PAPER_GRID = SurfaceGrid(
    (0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4),
    (1 / 252, 1 / 52, 2 / 52, 1 / 12, 1 / 6, 1 / 4, 1 / 2, 3 / 4, 1.0),
)


def flatten_surface(values: np.ndarray, grid: SurfaceGrid = PAPER_GRID) -> np.ndarray:
    """Preserve arbitrary leading batch/asset axes; flatten only the last two."""
    array = np.asarray(values, dtype=float)
    if array.ndim < 2 or array.shape[-2:] != grid.shape or not np.isfinite(array).all():
        raise InputError("surface dimensions or finite values do not match the grid")
    return array.swapaxes(-1, -2).reshape(array.shape[:-2] + (grid.size,)).copy()


def unflatten_surface(values: np.ndarray, grid: SurfaceGrid = PAPER_GRID) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or array.shape[-1] != grid.size or not np.isfinite(array).all():
        raise InputError("surface vector dimensions or values are invalid")
    return array.reshape(array.shape[:-1] + (grid.shape[1], grid.shape[0])).swapaxes(-1, -2).copy()


def interpolate_surface(values: np.ndarray, grid: SurfaceGrid,
                        moneyness: float, maturity: float) -> float | np.ndarray:
    """Linear in K/S, then maturity, with extrapolation and leading batch axes."""
    number(moneyness, "moneyness")
    number(maturity, "maturity")
    array = np.asarray(values, dtype=float)
    if (moneyness <= 0 or maturity <= 0 or array.ndim < 2 or array.shape[-2:] != grid.shape
            or not np.isfinite(array).all() or np.any(array <= 0)):
        raise InputError("invalid positive IV surface/interpolation inputs")
    i = int(np.clip(np.searchsorted(grid.moneyness, moneyness, side="right") - 1, 0, grid.shape[0] - 2))
    j = int(np.clip(np.searchsorted(grid.maturities, maturity, side="right") - 1, 0, grid.shape[1] - 2))
    wm = (moneyness - grid.moneyness[i]) / (grid.moneyness[i + 1] - grid.moneyness[i])
    wt = (maturity - grid.maturities[j]) / (grid.maturities[j + 1] - grid.maturities[j])
    lower = array[..., i, j] + wm * (array[..., i + 1, j] - array[..., i, j])
    upper = array[..., i, j + 1] + wm * (array[..., i + 1, j + 1] - array[..., i, j + 1])
    result = lower + wt * (upper - lower)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise InputError("linear surface extrapolation produced nonpositive volatility")
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class IVSurface:
    grid: SurfaceGrid
    values: np.ndarray
    observed_at: datetime
    available_at: datetime
    source: str

    def __post_init__(self) -> None:
        timestamp(self.observed_at)
        timestamp(self.available_at)
        array = np.array(self.values, dtype=float, copy=True)
        if self.available_at < self.observed_at or not self.source:
            raise InputError("surface provenance/availability is invalid")
        if array.shape != self.grid.shape or not np.isfinite(array).all() or np.any(array <= 0):
            raise InputError("surface must contain positive finite IVs on its declared grid")
        # A bytes-backed view cannot be made writable by a recipient.
        frozen = np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)
        object.__setattr__(self, "values", frozen)

    def volatility(self, moneyness: float, maturity: float) -> float:
        return interpolate_surface(self.values, self.grid, moneyness, maturity)


def smooth_surface(moneyness: np.ndarray, maturities: np.ndarray,
                   values: np.ndarray, vegas: np.ndarray, *,
                   grid: SurfaceGrid = PAPER_GRID,
                   bandwidth_m: float = 0.01, bandwidth_tau: float = 0.01) -> np.ndarray:
    """Vega-weighted Nadaraya-Watson Gaussian estimator, paper Eq. (21).

    h enters exp(-distance**2/(2*h)), not exp(-distance**2/(2*h**2)).
    Defaults are predeclared ETF adaptation choices, not a claimed reproduction
    of the authors' data-calibrated bandwidths. Log weights prevent underflow.
    """
    number(bandwidth_m, "moneyness bandwidth")
    number(bandwidth_tau, "maturity bandwidth")
    arrays = [np.asarray(a, dtype=float) for a in (moneyness, maturities, values, vegas)]
    m, tau, value, vega = arrays
    if (bandwidth_m <= 0 or bandwidth_tau <= 0 or m.ndim != 1 or len(m) == 0
            or any(a.shape != m.shape or not np.isfinite(a).all() for a in arrays)
            or any(np.any(a <= 0) for a in arrays)):
        raise InputError("kernel smoothing requires matching positive finite observations")
    log_weight = (np.log(vega)[None, None, :]
                  - (np.asarray(grid.moneyness)[:, None, None] - m) ** 2 / (2 * bandwidth_m)
                  - (np.asarray(grid.maturities)[None, :, None] - tau) ** 2 / (2 * bandwidth_tau))
    if not np.isfinite(log_weight).all():
        raise InputError("kernel weight calculation overflowed")
    log_weight -= log_weight.max(axis=-1, keepdims=True)
    weight = np.exp(log_weight)
    result = np.sum(weight * value, axis=-1) / weight.sum(axis=-1)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise InputError("kernel smoothing produced invalid volatility")
    return result
