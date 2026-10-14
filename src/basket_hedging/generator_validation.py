"""Held-out conditional distribution checks, separate from financial test paths."""

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from .contracts import InputError, timestamp
from .learning_data import conditioning_features, experiment_seed
from .market import Observation
from .synthetic import market_path

if TYPE_CHECKING:
    from .learned_model import ConditionalGaussianModel


QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)


def energy_score(observation: np.ndarray, samples: np.ndarray) -> float:
    """Lower-is-better joint score of an equally weighted empirical forecast.

    ES = mean_i ||x_i-y|| - sum_ij ||x_i-x_j|| / (2*N**2), with Euclidean
    distances in the input return units. This is the energy score of the
    empirical distribution (including diagonal pairs), not the unbiased
    estimator of an underlying continuous distribution's score. Compare
    forecasts using the same ensemble size; no marginal standardization is
    applied. See Gneiting and Raftery (2007), doi:10.1198/016214506000001437.
    """
    observation = np.asarray(observation, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if (observation.ndim != 1 or samples.ndim != 2 or samples.shape[0] < 2
            or observation.size < 1 or samples.shape[1] != observation.size
            or not np.isfinite(observation).all() or not np.isfinite(samples).all()):
        raise InputError("energy score requires finite D-vector and N-by-D samples with N >= 2")
    with np.errstate(over="ignore", invalid="ignore"):
        outcome_distance = np.linalg.norm(samples - observation, axis=1).mean()
        pair_distance = np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=2).mean()
        score = float(outcome_distance - 0.5 * pair_distance)
    if not np.isfinite(score):
        raise InputError("energy score distances exceed finite numerical range")
    return score


def _json_numbers(value):
    """Undefined correlations/relative errors are null, never JSON NaN."""
    array = np.asarray(value, dtype=float)
    output = array.astype(object)
    output[~np.isfinite(array)] = None
    return output.tolist()


def _compare_metric(reference, generated) -> dict:
    reference, generated = np.asarray(reference), np.asarray(generated)
    difference = generated - reference
    relative = np.divide(np.abs(difference), np.abs(reference),
                         out=np.full_like(difference, np.nan, dtype=float), where=reference != 0)
    return {"reference": _json_numbers(reference), "generated": _json_numbers(generated),
            "signed_error": _json_numbers(difference), "absolute_error": _json_numbers(abs(difference)),
            "relative_absolute_error": _json_numbers(relative)}


def _covariance_and_correlation(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = samples - samples.mean(axis=0)
    covariance = centered.T @ centered / (len(samples) - 1)
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    correlation = np.divide(covariance, scale, out=np.full_like(covariance, np.nan), where=scale > 0)
    return covariance, correlation


def compare_return_samples(reference: np.ndarray, generated: np.ndarray,
                           *, basket_weights: Sequence[float]) -> dict:
    """Compare equally sized joint simple-return samples; no acceptance score.

    Basket weights are current constituent values divided by current basket value,
    computed from fixed share coefficients by the caller. Variances/covariances
    use ddof=1; quantiles use NumPy's linear interpolation. W1 is the mean absolute
    difference between sorted marginal samples (equal empirical probabilities).
    """
    reference, generated = np.asarray(reference, dtype=float), np.asarray(generated, dtype=float)
    if (reference.ndim != 2 or reference.shape != generated.shape or reference.shape[0] < 2
            or reference.shape[1] < 1 or not np.isfinite(reference).all()
            or not np.isfinite(generated).all()):
        raise InputError("return samples must be equal finite N-by-D arrays with N >= 2")
    weights = np.asarray(basket_weights, dtype=float)
    if (weights.shape != (reference.shape[1],) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)):
        raise InputError("basket value weights must be nonnegative and sum to one")
    ref_cov, ref_corr = _covariance_and_correlation(reference)
    gen_cov, gen_corr = _covariance_and_correlation(generated)
    ref_basket, gen_basket = reference @ weights, generated @ weights
    return {
        "sample_count": len(reference),
        "assets": {
            "mean": _compare_metric(reference.mean(axis=0), generated.mean(axis=0)),
            "variance": _compare_metric(np.diag(ref_cov), np.diag(gen_cov)),
            "covariance": _compare_metric(ref_cov, gen_cov),
            "correlation": _compare_metric(ref_corr, gen_corr),
            "quantiles": {str(q): _compare_metric(np.quantile(reference, q, axis=0),
                                                   np.quantile(generated, q, axis=0)) for q in QUANTILES},
            "wasserstein_1": _json_numbers(np.mean(abs(np.sort(reference, axis=0)
                                                        - np.sort(generated, axis=0)), axis=0)),
        },
        "basket": {
            "mean": _compare_metric(ref_basket.mean(), gen_basket.mean()),
            "variance": _compare_metric(ref_basket.var(ddof=1), gen_basket.var(ddof=1)),
            "quantiles": {str(q): _compare_metric(np.quantile(ref_basket, q),
                                                   np.quantile(gen_basket, q)) for q in QUANTILES},
            "wasserstein_1": float(np.mean(abs(np.sort(ref_basket) - np.sort(gen_basket)))),
        },
    }


def _ordered_values(values: Mapping[str, float] | Sequence[float], stocks: tuple[str, ...],
                    name: str) -> tuple[float, ...]:
    if isinstance(values, Mapping):
        if set(values) != set(stocks):
            raise InputError(f"{name} must cover exactly the model stocks")
        values = tuple(values[stock] for stock in stocks)
    array = np.asarray(values, dtype=float)
    if array.shape != (len(stocks),) or not np.isfinite(array).all():
        raise InputError(f"{name} must be finite and match model stocks")
    return tuple(float(value) for value in array)


def validate_generator(model: "ConditionalGaussianModel", *,
                       initial_spots: Mapping[str, float] | Sequence[float],
                       volatilities: Sequence[float], correlation: Sequence[Sequence[float]],
                       coefficients: Mapping[str, float] | Sequence[float],
                       samples: int = 20000, seed: int = 314159) -> dict:
    """Evaluate frozen generator on declared states, not on hedge-test paths.

    Reference paths use the existing zero-drift synthetic GBM. Each reference
    draw and learned-noise block has a separate stable seed. Recent histories
    differ while current spots stay equal: the fixed-parameter reference model
    has independent future increments and no latent high-volatility regime.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise InputError("validation sample count must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise InputError("validation seed must be a nonnegative integer")
    timestamp(model.available_at, "model availability")
    stocks = tuple(model.stock_ids)
    spots = _ordered_values(initial_spots, stocks, "initial spots")
    shares = _ordered_values(coefficients, stocks, "basket coefficients")
    if any(value < 0 for value in shares) or sum(shares) <= 0:
        raise InputError("basket coefficients require nonnegative, positive total exposure")
    time = max(datetime(2026, 1, 5, 16, tzinfo=timezone.utc),
               model.available_at + timedelta(days=model.lags + 1)).astimezone(timezone.utc)
    time = time.replace(hour=16, minute=0, second=0, microsecond=0)
    while time.weekday() >= 5:
        time += timedelta(days=1)
    # Reuse the simulator's own validation before constructing any observations.
    tuple(market_path(stocks, spots, volatilities, correlation, (time,), seed))
    current_spots = dict(zip(stocks, spots))
    basket_values = np.asarray(shares) * np.asarray(spots)
    basket_weights = basket_values / basket_values.sum()
    states = (("normal", 0.0), ("recent_positive", 0.01), ("recent_negative", -0.01),
              ("cold_start", 0.0))
    cases = []
    for state_name, past_daily_log_return in states:
        for horizon_days in (1.0, 3.0):
            decision = time
            if horizon_days == 3:
                decision += timedelta(days=(4 - decision.weekday()) % 7)
            elif decision.weekday() == 4:
                decision += timedelta(days=3)
            # Match replay's weekday 09:00/16:00 bars, including weekend gaps.
            history_times = [decision]
            for _ in range(model.lags):
                previous = history_times[-1]
                if previous.hour == 16:
                    previous = previous.replace(hour=9)
                else:
                    previous = (previous - timedelta(days=1)).replace(hour=16)
                    while previous.weekday() >= 5:
                        previous -= timedelta(days=1)
                history_times.append(previous)
            state_times = [decision] if state_name == "cold_start" else reversed(history_times)
            history = tuple(
                Observation(f"validation/{state_name}/{observed.isoformat()}/{stock}", stock,
                            float(spot * np.exp(-past_daily_log_return * (decision-observed).total_seconds()/86400)),
                            observed, observed)
                for observed in state_times for stock, spot in zip(stocks, spots)
            )
            horizon = decision + timedelta(days=horizon_days)
            features, input_ids = conditioning_features(stocks, history, current_spots, decision, horizon,
                                                        lags=model.lags)
            stream = f"generator-validation/{state_name}/{horizon_days:g}d"
            reference_seed = experiment_seed(stream + "/reference", 0, seed)
            learned_seed = experiment_seed(stream + "/learned", 0, seed)
            reference_returns = np.empty((samples, len(stocks)))
            for index in range(samples):
                draw_seed = experiment_seed(stream + "/reference-draw", index, reference_seed)
                future = tuple(market_path(stocks, spots, volatilities, correlation,
                                           (decision, horizon), draw_seed))[-1][1]
                reference_returns[index] = [future[stock] / spot - 1 for stock, spot in zip(stocks, spots)]
            random = np.random.Generator(np.random.PCG64(learned_seed))
            normals = random.standard_normal((samples, len(stocks)))
            log_returns = np.asarray(model.sample_log_returns(features, horizon_days, normals), dtype=float)
            if log_returns.shape != reference_returns.shape or not np.isfinite(log_returns).all():
                raise InputError("learned generator returned invalid joint log-return samples")
            with np.errstate(over="ignore", invalid="ignore"):
                generated_returns = np.expm1(log_returns)
            metrics = compare_return_samples(reference_returns, generated_returns,
                                             basket_weights=basket_weights)
            cases.append({"state": state_name, "decision_time": decision.isoformat(),
                          "horizon": horizon.isoformat(), "horizon_days": horizon_days,
                          "past_daily_log_return": past_daily_log_return,
                          "history_times": sorted({record.observed_at.isoformat() for record in history}),
                          "features": features.tolist(), "input_ids": list(input_ids),
                          "reference_seed": reference_seed, "learned_seed": learned_seed,
                          **metrics})
    return {
        "purpose": "conditional distribution validation; no hedge-test paths or P&L selection",
        "reference": "existing synthetic zero-drift GBM; fixed volatility and correlation, ACT/365",
        "stock_ids": list(stocks), "initial_spots": list(spots),
        "volatilities": list(volatilities), "correlation": [list(row) for row in correlation],
        "basket_share_coefficients": list(shares), "basket_value_weights": basket_weights.tolist(),
        "model_available_at": model.available_at.isoformat(), "samples_per_case": samples, "seed": seed,
        "units": "simple returns; variances and covariances in squared simple-return units",
        "conventions": {"variance_ddof": 1, "quantile_interpolation": "linear",
                        "relative_error": "absolute error / abs(reference); null for zero reference",
                        "undefined_correlation": "null when either marginal variance is zero",
                        "wasserstein_1": "mean absolute difference of sorted equally weighted samples"},
        "limitations": ["No learned or simulated volatility state exists in this fixed-volatility environment.",
                        "Recent positive/negative histories do not change the reference conditional law.",
                        "Differences include finite-sample noise; no arbitrary pass score or profitability claim."],
        "cases": cases,
    }
