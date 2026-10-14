"""Independent objectives and KKT checks for the bounded numerical fallback."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from basket_hedging import optimization
from basket_hedging.contracts import InputError
from basket_hedging.optimization import SolverConvergenceError, solve_hedge


FIXTURES = Path(__file__).parent / "fixtures" / "solver_regressions"
CASES = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["cases"]


def independent_kkt(response, design, previous, penalties, changes, intercept):
    """Scalar subgradient checks, independent of the production KKT helper."""
    residual = response - intercept - design @ (previous + changes)
    centered_response = response - design @ previous
    centered_response -= centered_response.mean()
    violations = []
    for column in range(design.shape[1]):
        feature = design[:, column] - design[:, column].mean()
        gradient = -2 * np.mean(feature * residual)
        scale = max(1., abs(2 * np.mean(feature * centered_response)), penalties[column])
        if changes[column] > 0:
            violation = abs(gradient + penalties[column])
        elif changes[column] < 0:
            violation = abs(gradient - penalties[column])
        else:
            violation = max(0., abs(gradient) - penalties[column])
        violations.append(violation / scale)
    return max(violations)


def correlated_case():
    first = np.array([-2., -1., 0., 1., 2.])
    second = first + .2 * np.array([1., -2., 2., -2., 1.])
    return 3 * first - 2 * second, np.column_stack((first, second))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["episode"])
def test_captured_failures_reach_independently_verified_same_objective(case):
    path = FIXTURES / case["file"]
    assert sha256(path.read_bytes()).hexdigest() == case["sha256"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    original = {key: value.tobytes() for key, value in arrays.items()}
    for value in arrays.values():
        value.setflags(write=False)
    response, design = arrays["delta_v"], arrays["delta_h"]
    previous, costs = arrays["previous_holdings"], arrays["cost_weights"]
    parameters = case["kwargs"]
    penalties = parameters["alpha"] * parameters["gross_scale"] * costs

    # The frozen last iterate genuinely misses the original acceptance test.
    last = arrays["final_changes"]
    intercept = (response - design @ previous).mean() - design.mean(axis=0) @ last
    original_kkt = independent_kkt(response, design, previous, penalties, last, intercept)
    assert original_kkt > parameters["tolerance"]
    assert original_kkt == pytest.approx(case["original_kkt_residual"], rel=2e-5, abs=1e-14)

    result = solve_hedge(response, design, previous, costs, **parameters)

    assert result.polish_attempted and result.polish_succeeded
    assert result.iterations == case["original_iterations"] == 20000
    assert result.kkt_residual <= parameters["tolerance"]
    assert independent_kkt(response, design, previous, penalties, result.changes,
                           result.intercept) <= parameters["tolerance"]
    reference = float(case["reference_objective"])
    assert result.objective == pytest.approx(reference, rel=2e-13, abs=2e-11)
    residual = response - result.intercept - design @ result.holdings
    assert np.mean(residual ** 2) + penalties @ abs(result.changes) == pytest.approx(reference, rel=2e-13)
    assert {key: value.tobytes() for key, value in arrays.items()} == original


def test_successful_coordinate_descent_keeps_prechange_result(monkeypatch):
    def forbidden(*args):
        pytest.fail("Successful coordinate descent must not attempt polishing")

    monkeypatch.setattr(optimization, "_active_set_candidates", forbidden)
    response, design = correlated_case()
    result = solve_hedge(response, design, [0, 0], [1, 1], alpha=.02)
    # Snapshot captured before the correction, alongside the independent optimum.
    assert result.iterations == 339
    assert result.objective == pytest.approx(.09637857142857147, abs=1e-15)
    assert result.mse == pytest.approx(.003621429283292607, abs=1e-14)
    assert result.kkt_residual == pytest.approx(9.694641660631942e-10, abs=1e-14)
    assert result.holdings == pytest.approx([19715/7000, -1275/700], abs=1e-7)
    assert not result.polish_attempted and not result.polish_succeeded


def test_alpha_zero_keeps_minimum_norm_change_without_polishing(monkeypatch):
    monkeypatch.setattr(optimization, "_active_set_candidates",
                        lambda *args: pytest.fail("OLS must not invoke polishing"))
    first = np.array([-1., 1., -1., 1.])
    design = np.column_stack((first, first))
    result = solve_hedge(7 + 3 * first, design, [2, -1], [1, 5], alpha=0)
    assert result.holdings == pytest.approx([3, 0], abs=1e-12)
    assert result.changes == pytest.approx([1, 1], abs=1e-12)
    assert result.intercept == pytest.approx(7, abs=1e-12)
    assert result.objective < 1e-25 and result.iterations == 0
    assert not result.polish_attempted and not result.polish_succeeded


def test_inactive_coordinate_unequal_penalties_and_absolute_holdings():
    response, pair = correlated_case()
    third = np.array([1., -2., 0., 2., -1.])
    design = np.column_stack((pair, third))
    previous = np.array([.25, -.75, 1.2])
    response = 7 + design @ previous + response + .025 * third
    costs = np.array([1., 2., 10.])
    result = solve_hedge(response, design, previous, costs, alpha=.02, max_iterations=1)
    expected = np.array([1909/700, -97/56, 0])
    assert result.changes == pytest.approx(expected, abs=1e-12)
    assert result.changes[2] == 0
    assert result.holdings == pytest.approx(previous + expected, abs=1e-12)
    assert result.intercept == pytest.approx(7, abs=1e-12)
    assert result.mse == pytest.approx(1307/140000, abs=1e-12)
    assert result.penalty == pytest.approx(2167/17500, abs=1e-12)
    assert result.objective == pytest.approx(18643/140000, abs=1e-12)
    assert independent_kkt(response, design, previous, .02 * costs, result.changes,
                           result.intercept) <= 1e-9
    assert result.polish_attempted and result.polish_succeeded and result.iterations == 1


def test_nearly_collinear_columns_use_the_unchanged_weighted_l1_objective():
    first = np.array([-2., -1., 0., 1., 2.])
    epsilon, penalty = .001, 1e-9
    second = first + epsilon * np.array([1., -2., 2., -2., 1.])
    design = np.column_stack((first, second))
    response = 3 * first - 2 * second
    result = solve_hedge(response, design, [0, 0], [1, 1], alpha=penalty, max_iterations=1)
    # Orthogonal perturbation has variance 2.8; solve the two signed equations.
    expected = [3-penalty/4-penalty/(2.8*epsilon**2), -2+penalty/(2.8*epsilon**2)]
    objective = 5*penalty-penalty**2/(2.8*epsilon**2)-penalty**2/8
    assert result.holdings == pytest.approx(expected, abs=1e-9)
    assert result.objective == pytest.approx(objective, abs=1e-15)
    assert result.polish_attempted and result.polish_succeeded
    assert independent_kkt(response, design, np.zeros(2), np.full(2, penalty),
                           result.changes, result.intercept) <= 1e-9


def test_exactly_redundant_columns_compare_objective_and_optimality():
    response, pair = correlated_case()
    design = np.column_stack((pair, pair[:, 0]))
    result = solve_hedge(response, design, [0, 0, 0], [1, 1, 1], alpha=.02, max_iterations=1)
    expected_prediction = pair @ np.array([19715/7000, -1275/700])
    assert design @ result.holdings == pytest.approx(expected_prediction, abs=1e-12)
    assert result.objective == pytest.approx(13493/140000, abs=1e-12)
    assert independent_kkt(response, design, np.zeros(3), np.full(3, .02),
                           result.changes, result.intercept) <= 1e-9
    assert result.polish_attempted and result.polish_succeeded


def test_polishing_is_deterministic_and_leaves_all_input_arrays_unchanged():
    response, design = correlated_case()
    previous, costs = np.array([.25, -.75]), np.array([1., 2.])
    original = [value.copy() for value in (response, design, previous, costs)]
    first = solve_hedge(response, design, previous, costs, alpha=.02, max_iterations=1)
    second = solve_hedge(response, design, previous, costs, alpha=.02, max_iterations=1)
    assert first.polish_succeeded and second.polish_succeeded
    for name in ("holdings", "changes"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    for name in ("objective", "kkt_residual", "intercept", "iterations"):
        assert getattr(first, name) == getattr(second, name)
    for before, after in zip(original, (response, design, previous, costs)):
        assert np.array_equal(before, after)
    with pytest.raises(ValueError):
        first.holdings[0] = 99


def test_unsupported_dimension_keeps_explicit_failure_without_enumeration(monkeypatch):
    monkeypatch.setattr(optimization, "_active_set_candidates",
                        lambda *args: pytest.fail("Enumeration must be dimension bounded"))
    response, pair = correlated_case()
    design = np.column_stack((pair, np.zeros((len(response), 2))))
    with pytest.raises(SolverConvergenceError, match="did not converge after 1 iterations") as caught:
        solve_hedge(response, design, np.zeros(4), np.ones(4), alpha=.02, max_iterations=1)
    assert caught.value.iterations == 1 and caught.value.kkt_residual > 1e-9
    assert not caught.value.polish_attempted and not caught.value.polish_succeeded


@pytest.mark.parametrize("proposed", [(), (np.array([.995, 0.]),)])
def test_no_verified_candidate_preserves_solver_failure(monkeypatch, proposed):
    # (.995,0) satisfies active stationarity but violates the inactive inequality.
    monkeypatch.setattr(optimization, "_active_set_candidates", lambda *args: iter(proposed))
    response, design = correlated_case()
    before = response.copy(), design.copy()
    with pytest.raises(SolverConvergenceError) as caught:
        solve_hedge(response, design, [0, 0], [1, 1], alpha=.02, max_iterations=1)
    assert caught.value.polish_attempted and not caught.value.polish_succeeded
    assert caught.value.iterations == 1 and caught.value.kkt_residual > 1e-9
    assert np.array_equal(response, before[0]) and np.array_equal(design, before[1])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_invalid_input_keeps_input_error_classification_before_fallback(monkeypatch, invalid):
    monkeypatch.setattr(optimization, "_active_set_candidates",
                        lambda *args: pytest.fail("Invalid inputs cannot reach polishing"))
    response, design = correlated_case()
    response[0] = invalid
    with pytest.raises(InputError):
        solve_hedge(response, design, [0, 0], [1, 1], alpha=.02, max_iterations=1)
