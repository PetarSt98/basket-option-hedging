"""Independent grid ordering, kernel weights and interpolation checks."""

from datetime import datetime, timezone
from math import exp

import numpy as np
import pytest

from basket_hedging.contracts import InputError
from basket_hedging.surfaces import (
    IVSurface, PAPER_GRID, SurfaceGrid, flatten_surface, interpolate_surface,
    smooth_surface, unflatten_surface,
)


GRID = SurfaceGrid((0.9, 1.1), (0.1, 0.3))


def test_maturity_major_order_with_separate_batch_and_asset_dimensions():
    matrix = np.array([[1., 3.], [2., 4.]])
    assert flatten_surface(matrix, GRID).tolist() == [1, 2, 3, 4]
    batch = np.stack((matrix, matrix + 10))[None, ...]
    flat = flatten_surface(batch, GRID)
    assert flat.shape == (1, 2, 4)
    np.testing.assert_array_equal(flat[0, 1], [11, 12, 13, 14])
    np.testing.assert_array_equal(unflatten_surface(flat, GRID), batch)
    assert PAPER_GRID.shape == (11, 9)


def test_linear_surface_interpolation_and_extrapolation_are_not_constant_edges():
    # Affine surface sigma=.1+.1*m+.2*T is reproduced even outside the grid.
    values = np.array([[.21, .25], [.23, .27]])
    assert interpolate_surface(values, GRID, 1., .2) == pytest.approx(.24)
    assert interpolate_surface(values, GRID, .5, .5) == pytest.approx(.25)
    np.testing.assert_allclose(interpolate_surface(np.stack((values, values + .1)), GRID, 1., .2), [.24, .34])
    with pytest.raises(InputError, match="nonpositive"):
        interpolate_surface(np.array([[.3, .1], [.3, .1]]), GRID, 1., .5)


def test_kernel_uses_vega_weights_and_bandwidth_not_squared_bandwidth():
    # At grid point m=.9,T=.1, the second point is .2 away in m:
    # exponent=-.2^2/(2*.02)=-1. Other coordinates agree.
    actual = smooth_surface(np.array([.9, 1.1]), np.array([.1, .1]),
                            np.array([.2, .4]), np.array([1., 3.]), grid=GRID,
                            bandwidth_m=.02, bandwidth_tau=.01)
    expected = (.2 + 3 * exp(-1) * .4) / (1 + 3 * exp(-1))
    assert actual[0, 0] == pytest.approx(expected, abs=1e-14)
    constant = smooth_surface(np.array([.01]), np.array([10.]), np.array([.37]),
                             np.array([1.]), grid=GRID, bandwidth_m=1e-6, bandwidth_tau=1e-6)
    np.testing.assert_allclose(constant, .37)  # Log weights avoid all-zero underflow.


def test_surface_is_detached_readonly_and_does_not_carry_quotes():
    time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    source = np.full(GRID.shape, .2)
    surface = IVSurface(GRID, source, time, time, "test-smoothed-IV")
    source[:] = .9
    assert surface.volatility(1., .2) == pytest.approx(.2)
    with pytest.raises(ValueError):
        surface.values.setflags(write=True)
    assert not hasattr(surface, "bid")


@pytest.mark.parametrize("maturities", [(0., 1.), (1., 1.), (1., float("nan"))])
def test_invalid_grids_rejected(maturities):
    with pytest.raises(InputError):
        SurfaceGrid((.9, 1.1), maturities)


def test_invalid_surface_and_nonfinite_transform_rejected():
    with pytest.raises(InputError):
        flatten_surface(np.full(GRID.shape, np.nan), GRID)
    with pytest.raises(InputError):
        unflatten_surface(np.ones(3), GRID)
    with pytest.raises(InputError):
        smooth_surface(np.array([1.]), np.array([.1]), np.array([.2]), np.array([0.]), grid=GRID)
