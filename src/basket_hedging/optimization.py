"""Convex scenario hedge regression with an L1 penalty on changes in holdings.

This component solves the supplied numerical regression only. The caller owns
the liability/hedge sign convention and returns the resulting holdings to the
engine; the regression intercept is never an account cash balance.
"""

from dataclasses import dataclass

import numpy as np

from .contracts import InputError, number


class SolverConvergenceError(ArithmeticError):
    """The solver could not verify the requested first-order tolerance."""


@dataclass(frozen=True)
class HedgeSolution:
    holdings: np.ndarray
    changes: np.ndarray
    intercept: float
    mse: float
    penalty: float
    objective: float
    iterations: int
    kkt_residual: float

    def __post_init__(self) -> None:
        for name in ("holdings", "changes"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _array(value, name: str, dimensions: int) -> np.ndarray:
    try:
        if np.iscomplexobj(value):
            raise InputError(f"{name} must be real")
        result = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise InputError(f"{name} must be a real numeric array") from error
    if result.ndim != dimensions or not np.isfinite(result).all():
        raise InputError(f"{name} must be a finite {dimensions}-dimensional array")
    return result


def _kkt_residual(gradient: np.ndarray, changes: np.ndarray,
                  penalties: np.ndarray, scales: np.ndarray) -> float:
    # At beta=0, the subgradient interval is [-lambda, lambda].
    violation = np.where(changes != 0,
                         np.abs(gradient + penalties * np.sign(changes)),
                         np.maximum(np.abs(gradient) - penalties, 0.0))
    return float(np.max(violation / scales))


def solve_hedge(delta_v, delta_h, previous_holdings, cost_weights, *,
                alpha: float = 0.0, gross_scale: float = 1.0,
                tolerance: float = 1e-9, max_iterations: int = 20000) -> HedgeSolution:
    """Minimize mean((dv - A - X phi)**2) + alpha*g0*sum(c*abs(phi-prev)).

    Rows of ``delta_h`` are scenarios and columns are hedge instruments; all
    changes use the caller's consistent currency/share/contract units. ``A`` is
    a free, unpenalized regression intercept. Centering removes it exactly:
    y=dv-X*prev, beta=phi-prev, A=mean(y)-mean(X)*beta.

    With zero penalties, least squares returns the minimum-norm *change* in
    holdings. Otherwise deterministic cyclic coordinate descent uses the
    weighted soft threshold lambda/2 for this mean-square normalization.
    No feature standardization, ridge, rounding or position constraints apply.

    Convergence requires each KKT violation divided by
    max(1, abs(2*X_centered.T*y_centered/N), lambda) to be <= tolerance.
    ``kkt_residual`` is the maximum of these dimensionless violations. This is
    a floating-point stopping check, not an exact optimality certificate.
    """
    response = _array(delta_v, "delta_v", 1)
    design = _array(delta_h, "delta_h", 2)
    previous = _array(previous_holdings, "previous_holdings", 1)
    costs = _array(cost_weights, "cost_weights", 1)
    scenarios, instruments = design.shape
    if (scenarios < 2 or instruments < 1 or response.shape != (scenarios,)
            or previous.shape != (instruments,) or costs.shape != (instruments,)):
        raise InputError("need N>=2 scenarios, J>=1 instruments, dv[N], X[N,J], prev[J], costs[J]")
    if np.any(costs < 0):
        raise InputError("cost_weights must be nonnegative")
    number(alpha, "alpha", minimum=0)
    number(gross_scale, "gross_scale")
    number(tolerance, "tolerance")
    if gross_scale <= 0 or tolerance <= 0:
        raise InputError("gross_scale and tolerance must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise InputError("max_iterations must be a positive integer")

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _solve(response, design, previous, costs, float(alpha), float(gross_scale),
                          float(tolerance), int(max_iterations))
    except FloatingPointError as error:
        raise InputError("regression arithmetic overflowed; check input units and scale") from error


def _solve(response, design, previous, costs, alpha, gross_scale, tolerance, max_iterations):
    scenarios, instruments = design.shape
    mean_design = design.mean(axis=0)
    adjusted = response - design @ previous
    mean_adjusted = float(adjusted.mean())
    x = design - mean_design
    y = adjusted - mean_adjusted
    penalties = alpha * gross_scale * costs
    covariance = x.T @ y / scenarios
    scales = np.maximum(1.0, np.maximum(2 * np.abs(covariance), penalties))
    iterations = 0

    if not np.any(penalties):
        try:
            changes = np.linalg.lstsq(x, y, rcond=None)[0]
        except np.linalg.LinAlgError as error:
            raise SolverConvergenceError("least-squares decomposition did not converge") from error
    else:
        changes = np.zeros(instruments)
        variance = np.sum(x * x, axis=0) / scenarios
        residual = y.copy()
        kkt = _kkt_residual(-2 * covariance, changes, penalties, scales)
        while kkt > tolerance and iterations < max_iterations:
            for column in range(instruments):
                if variance[column] == 0:
                    continue  # Constant scenario changes are absorbed by the intercept.
                partial = residual + x[:, column] * changes[column]
                correlation = float(x[:, column] @ partial / scenarios)
                updated = (np.sign(correlation) * max(abs(correlation) - penalties[column] / 2, 0.0)
                           / variance[column])
                residual = partial - x[:, column] * updated
                changes[column] = updated
            iterations += 1
            residual = y - x @ changes
            gradient = -2 * (x.T @ residual) / scenarios
            kkt = _kkt_residual(gradient, changes, penalties, scales)

    holdings = previous + changes
    intercept = mean_adjusted - float(mean_design @ changes)
    residual = response - intercept - design @ holdings
    gradient = -2 * (x.T @ residual) / scenarios
    kkt = _kkt_residual(gradient, changes, penalties, scales)
    if not np.isfinite(kkt) or kkt > tolerance:
        raise SolverConvergenceError(
            f"hedge regression did not converge after {iterations} iterations: "
            f"scaled KKT residual {kkt:.6g} exceeds {tolerance:.6g}")
    mse = float(residual @ residual / scenarios)
    penalty = float(penalties @ np.abs(changes))
    return HedgeSolution(holdings, changes, intercept, mse, penalty, mse + penalty, iterations, kkt)
