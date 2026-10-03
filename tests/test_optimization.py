"""Small analytic regressions verify the optimizer independently of the engine."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from basket_hedging.contracts import InputError
from basket_hedging.optimization import SolverConvergenceError, solve_hedge


ORTHOGONAL = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])


def test_perfect_hedge_and_free_intercept_with_nonzero_current_holdings():
    x = np.array([-2., -1., 0., 1., 2.])[:, None]
    result = solve_hedge(7 + 2 * x[:, 0], x, [.25], [1])
    assert result.holdings == pytest.approx([2], abs=1e-12)
    assert result.changes == pytest.approx([1.75], abs=1e-12)
    assert result.intercept == pytest.approx(7, abs=1e-12)
    assert result.mse < 1e-25 and result.penalty == 0
    assert result.objective == result.mse
    assert result.kkt_residual < 1e-12
    assert result.iterations == 0  # Direct least squares, no coordinate iterations.


@pytest.mark.parametrize("sign", [-1, 1])
def test_regression_returns_signed_coefficients_without_hidden_sign_flip(sign):
    x = np.array([-1., 1.])[:, None]
    result = solve_hedge(sign * 2 * x[:, 0], x, [0], [1])
    assert result.holdings == pytest.approx([sign * 2])
    assert result.intercept == pytest.approx(0)


def test_orthogonal_soft_threshold_and_objective_have_hand_derived_values():
    y = 5 + 3 * ORTHOGONAL[:, 0] - .4 * ORTHOGONAL[:, 1]
    result = solve_hedge(y, ORTHOGONAL, [0, 0], [1, 1], alpha=1)
    # Each feature has variance 1. Threshold lambda/2=.5 gives (2.5,0).
    assert result.holdings == pytest.approx([2.5, 0], abs=1e-12)
    assert result.intercept == pytest.approx(5, abs=1e-12)
    assert result.mse == pytest.approx(.41, abs=1e-12)
    assert result.penalty == pytest.approx(2.5, abs=1e-12)
    assert result.objective == pytest.approx(2.91, abs=1e-12)
    assert result.iterations == 1


def test_unregularized_matches_independent_augmented_ols_with_noise():
    design = np.array([[1., 2.], [3., 1.], [-1., 0.], [2., -2.], [0., 1.]])
    response = np.array([4., 8., -2., 7., 1.])
    coefficients = np.linalg.lstsq(np.column_stack([np.ones(5), design]), response, rcond=None)[0]
    result = solve_hedge(response, design, [2, -3], [.3, .7], alpha=0)
    assert result.intercept == pytest.approx(coefficients[0], abs=1e-12)
    assert result.holdings == pytest.approx(coefficients[1:], abs=1e-12)
    independent_residual = response - np.column_stack([np.ones(5), design]) @ coefficients
    assert result.mse == pytest.approx(float(np.mean(independent_residual ** 2)), abs=1e-12)


def test_unregularized_redundancy_minimizes_change_preserving_existing_nullspace():
    x = np.array([-1., 1., -1., 1.])
    result = solve_hedge(3 * x, np.column_stack([x, x]), [2, -1], [1, 1])
    # Existing total coefficient1 requires +2. Minimum-norm change is (+1,+1).
    assert result.holdings == pytest.approx([3, 0], abs=1e-12)
    assert result.changes == pytest.approx([1, 1], abs=1e-12)
    assert result.mse < 1e-25


def test_turnover_penalty_is_around_current_holdings_and_gross_scale_is_applied():
    x = np.array([-1., 1.])[:, None]
    # y=0 and prior2: minimize phi^2 + 2*abs(phi-2), hence target1/change-1.
    result = solve_hedge([0, 0], x, [2], [1], alpha=.5, gross_scale=4)
    assert result.holdings == pytest.approx([1])
    assert result.changes == pytest.approx([-1])
    assert result.mse == pytest.approx(1)
    assert result.penalty == pytest.approx(2)
    assert result.objective == pytest.approx(3)


def test_no_trade_region_retains_current_hedge():
    x = np.array([-1., 1.])[:, None]
    result = solve_hedge(3 * x[:, 0], x, [2], [1], alpha=2)
    assert result.holdings == pytest.approx([2])
    assert result.changes.tolist() == [0]
    assert result.iterations == 0
    assert result.penalty == 0


def test_increasing_regularization_weakly_reduces_turnover_in_controlled_case():
    y = 3 * ORTHOGONAL[:, 0] + .4 * ORTHOGONAL[:, 1]
    changes = [solve_hedge(y, ORTHOGONAL, [0, 0], [1, 1], alpha=a).changes
               for a in (0, .5, 2, 8)]
    turnover = [float(np.abs(change).sum()) for change in changes]
    assert turnover == pytest.approx([3.4, 2.9, 2, 0])
    assert changes[2] == pytest.approx([2, 0])  # One sparse change remains.


def test_weights_choose_cheaper_of_redundant_instruments():
    x = np.array([-1., 1., -1., 1.])
    design = np.column_stack([x, x])
    left = solve_hedge(2 * x, design, [0, 0], [2, 1], alpha=.4)
    right = solve_hedge(2 * x, design, [0, 0], [1, 2], alpha=.4)
    assert left.holdings == pytest.approx([0, 1.8], abs=1e-8)
    assert right.holdings == pytest.approx([1.8, 0], abs=1e-8)
    assert left.mse == pytest.approx(.04, abs=1e-8)
    assert left.penalty == pytest.approx(.72, abs=1e-8)
    assert left.objective == pytest.approx(.76, abs=1e-8)


def test_zero_cost_instrument_remains_unpenalized():
    x = np.array([-1., 1.])
    design = np.column_stack([x, x])
    result = solve_hedge(2 * x, design, [0, 0], [0, 1], alpha=1)
    assert result.holdings == pytest.approx([2, 0])
    assert result.penalty == 0 and result.mse < 1e-25
    all_free = solve_hedge(2 * x, design, [0, 0], [0, 0], alpha=1)
    assert all_free.holdings == pytest.approx([1, 1])
    assert all_free.iterations == 0


def test_feature_units_do_not_silently_standardize_the_cost_penalty():
    # One contract changes by +/-10 currency; target response changes by +/-2.
    # Minimize (2-10*phi)^2 + 2*abs(phi): phi=(20-1)/100=.19.
    result = solve_hedge([-2, 2], [[-10], [10]], [0], [1], alpha=2)
    assert result.holdings == pytest.approx([.19], abs=1e-12)
    assert result.mse == pytest.approx(.01, abs=1e-12)
    assert result.penalty == pytest.approx(.38, abs=1e-12)
    assert result.objective == pytest.approx(.39, abs=1e-12)


@pytest.mark.parametrize("alpha", [0, .2])
def test_constant_columns_are_absorbed_by_intercept(alpha):
    x = np.array([-1., 1.])
    design = np.column_stack([x, np.full(2, 4.)])
    result = solve_hedge(7 + 2 * x, design, [0, 5], [1, 1], alpha=alpha)
    assert result.changes[1] == 0
    assert result.holdings[1] == 5
    assert result.intercept == pytest.approx(-13)
    assert result.holdings[0] == pytest.approx(2-alpha/2)
    only_constant = solve_hedge([2, 4], [[5], [5]], [3], [1], alpha=alpha)
    assert only_constant.holdings.tolist() == [3]
    assert only_constant.intercept == -12
    assert only_constant.mse == 1 and only_constant.penalty == 0


def correlated_case():
    first = np.array([-2., -1., 0., 1., 2.])
    second = first + .2 * np.array([1., -2., 2., -2., 1.])
    return 3 * first - 2 * second, np.column_stack([first, second])


def test_highly_correlated_case_converges_to_independent_active_set_solution():
    response, design = correlated_case()
    result = solve_hedge(response, design, [0, 0], [1, 1], alpha=.02)
    # G=[[2,2],[2,2.112]], cov=(2,1.776), signs=(+,-).
    # Normal equations G*phi=(1.99,1.786) give the following exact decimals.
    assert result.holdings == pytest.approx([2.816428571428571, -1.821428571428571], abs=1e-6)
    assert result.kkt_residual <= 1e-9
    assert 1 < result.iterations < 20000


def test_nonconvergence_raises_instead_of_returning_unverified_holdings():
    response, design = correlated_case()
    with pytest.raises(SolverConvergenceError, match="did not converge after 1 iterations"):
        solve_hedge(response, design, [0, 0], [1, 1], alpha=.02, max_iterations=1)


def test_deterministic_results_and_detached_readonly_arrays():
    response, design = correlated_case()
    previous = np.zeros(2)
    left = solve_hedge(response, design, previous, [1, 1], alpha=.02)
    right = solve_hedge(response, design, previous, [1, 1], alpha=.02)
    assert np.array_equal(left.holdings, right.holdings)
    assert left.objective == right.objective and left.iterations == right.iterations
    previous[0] = 100
    assert left.holdings[0] < 3
    with pytest.raises(ValueError):
        left.holdings[0] = 100
    with pytest.raises(FrozenInstanceError):
        left.intercept = 100


@pytest.mark.parametrize("response,design,previous,costs", [
    ([1], [[1]], [0], [1]),  # Only one scenario.
    ([1, 2], [[], []], [], []),
    ([1], [[1], [2]], [0], [1]),
    ([1, 2], [[1], [2]], [0, 1], [1]),
    ([1, 2], [[1], [2]], [0], [1, 2]),
    ([[1], [2]], [[1], [2]], [0], [1]),
    ([1, 2], [1, 2], [0], [1]),
    ([1, float("nan")], [[1], [2]], [0], [1]),
    ([1, 2], [[1], [float("inf")]], [0], [1]),
    ([1, 2], [[1], [2]], [float("nan")], [1]),
    ([1, 2], [[1], [2]], [0], [float("inf")]),
    ([1, 2], [[1], [2]], [0], [-1]),
    ([1, 2j], [[1], [2]], [0], [1]),
])
def test_invalid_regression_inputs_rejected(response, design, previous, costs):
    with pytest.raises(InputError):
        solve_hedge(response, design, previous, costs)


@pytest.mark.parametrize("parameters", [
    {"alpha": -1}, {"alpha": float("nan")}, {"alpha": True},
    {"gross_scale": 0}, {"gross_scale": -1}, {"gross_scale": float("inf")},
    {"tolerance": 0}, {"tolerance": float("nan")},
    {"max_iterations": 0}, {"max_iterations": 1.5}, {"max_iterations": True},
])
def test_invalid_solver_configuration_rejected(parameters):
    with pytest.raises(InputError):
        solve_hedge([1, 2], [[1], [2]], [0], [1], **parameters)


def test_finite_inputs_whose_arithmetic_overflows_are_rejected():
    with pytest.raises(InputError, match="overflowed"):
        solve_hedge([1, 2], [[1e308], [-1e308]], [1e308], [1])
