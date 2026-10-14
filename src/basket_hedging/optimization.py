"""Convex scenario hedge regression with an L1 penalty on changes in holdings.

This component solves the supplied numerical regression only. The caller owns
the liability/hedge sign convention and returns the resulting holdings to the
engine; the regression intercept is never an account cash balance.
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from .contracts import InputError, number


_MAX_POLISH_INSTRUMENTS = 3


class SolverConvergenceError(ArithmeticError):
    """The solver could not verify the requested first-order tolerance."""

    def __init__(self, message: str, *, polish_attempted: bool = False,
                 iterations: int | None = None, kkt_residual: float | None = None):
        super().__init__(message)
        self.polish_attempted = polish_attempted
        self.polish_succeeded = False
        self.iterations = iterations
        self.kkt_residual = kkt_residual


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
    polish_attempted: bool = False
    polish_succeeded: bool = False

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


def _active_set_candidates(x: np.ndarray, y: np.ndarray,
                           penalties: np.ndarray) -> Iterator[np.ndarray]:
    """At most 27 sign patterns; candidates still require the full KKT check."""
    scenarios, instruments = x.shape
    if instruments > _MAX_POLISH_INSTRUMENTS:
        return
    for pattern in product((-1, 0, 1), repeat=instruments):
        signs = np.asarray(pattern)
        active = np.flatnonzero(signs)
        changes = np.zeros(instruments)
        if active.size:
            try:
                q, r = np.linalg.qr(x[:, active], mode="reduced")
                # A redundant optimum has an independent-support representative.
                # QR avoids forming the more poorly conditioned normal matrix.
                if r.shape[0] != active.size or np.linalg.matrix_rank(r) < active.size:
                    continue
                adjustment = np.linalg.solve(r.T, scenarios * penalties[active] * signs[active] / 2)
                changes[active] = np.linalg.solve(r, q.T @ y - adjustment)
            except (np.linalg.LinAlgError, FloatingPointError):
                continue
            if not np.isfinite(changes).all() or np.any(changes[active] * signs[active] <= 0):
                continue
        yield changes


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
    Only failed, budget-exhausted penalized solves with at most three columns
    receive active-set polishing. Successful solves and zero penalties retain
    their original path. ``iterations`` remains the coordinate-descent count.

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

    def assess(candidate: np.ndarray) -> HedgeSolution:
        holdings = previous + candidate
        intercept = mean_adjusted - float(mean_design @ candidate)
        residual = response - intercept - design @ holdings
        gradient = -2 * (x.T @ residual) / scenarios
        kkt = _kkt_residual(gradient, candidate, penalties, scales)
        mse = float(residual @ residual / scenarios)
        penalty = float(penalties @ np.abs(candidate))
        return HedgeSolution(holdings, candidate, intercept, mse, penalty, mse + penalty, iterations, kkt)

    solution = assess(changes)
    if not np.isfinite(solution.kkt_residual) or solution.kkt_residual > tolerance:
        attempted = bool(np.any(penalties) and iterations == max_iterations
                         and instruments <= _MAX_POLISH_INSTRUMENTS)
        best = None
        if attempted:
            for candidate in _active_set_candidates(x, y, penalties):
                try:
                    polished = assess(candidate)
                except FloatingPointError:
                    continue
                # Includes inactive inequalities and the original uncentered
                # residual after converting changes back to absolute holdings.
                if (np.isfinite(polished.objective) and np.isfinite(polished.kkt_residual)
                        and polished.kkt_residual <= tolerance
                        and (best is None or polished.objective < best.objective)):
                    best = polished
        if best is not None:
            return replace(best, polish_attempted=True, polish_succeeded=True)
        raise SolverConvergenceError(
            f"hedge regression did not converge after {iterations} iterations: "
            f"scaled KKT residual {solution.kkt_residual:.6g} exceeds {tolerance:.6g}; "
            f"polishing attempted={attempted}, succeeded=False",
            polish_attempted=attempted, iterations=iterations, kkt_residual=solution.kkt_residual)
    return solution
