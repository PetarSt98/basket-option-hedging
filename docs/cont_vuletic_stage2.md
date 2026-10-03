# Stage 2: a learned conditional scenario provider

This extension trains a small conditional distribution model on synthetic stock
paths. The existing Stage 1 optimizer still chooses absolute target holdings.
The engine, ledger, execution, margin, strategy interface and pricing formulas
are unchanged. The original GBM scenario provider remains available.

## Method and sources

Before implementation, we inspected [Cont and Vuletić, *Data-driven hedging with
generative models*](https://link.springer.com/article/10.1007/s10479-025-06867-3),
particularly the scenario/pricing separation in Section 2, the Section 3
optimization objective and the empirical generator discussion. We also inspected
the [official VolGAN implementation](https://github.com/milenavuletic/VolGAN/blob/main/VolGAN.py)
and [example](https://github.com/milenavuletic/VolGAN/blob/main/VolGAN-example.py).

The reproduced structure is: released history → conditional joint scenarios →
common instrument repricing → existing variance/L1 optimizer → target holdings.
Physical return scenarios remain distinct from risk-neutral pricing. The cash
account remains independent of the fitted regression intercept.

**This is not a VolGAN reproduction.** Official VolGAN jointly generates an
underlying return and implied-volatility surface changes using adversarial
training. This prototype uses Gaussian conditional density estimation for
multiple stocks, trained by likelihood. It has no discriminator, volatility
surface, surface penalties, empirical calibration or reproduction of the
paper's results. A Gaussian output is a deliberate simplification appropriate
to testing the existing fixed-volatility GBM environment; it supplies substantial
distributional structure in advance.

## Exact model

For three stocks and three recent joint return bars, the network is:

```text
19 standardized inputs → Dense(24) → tanh → Dense(24) → tanh → Dense(9)
outputs: 3 mean rates and 6 lower-triangular Cholesky entries
diagonal entries: softplus(raw value) + 0.0001

joint log returns = 0.01 * days * mean_rate(features)
                  + 0.01 * sqrt(days) * L(features) @ independent_normals
future spots = current spots * exp(joint log returns)
```

There are 1,305 fitted network parameters. Feature means and standard deviations
are fitted on TRAIN only; constant training features use scale 1. The factor
0.01 is a declared numerical return scale, not supplied true volatility. Linear
mean-time and square-root diffusion-time scaling are built-in assumptions.
The horizon is also a network input, so the conditional mean and factor may
change with it; extrapolation to horizons absent from training is unvalidated.

Training uses PyTorch CPU float64, one thread, Adam, learning rate 0.001,
weight decay 0.0001, batch size 512, seed 271828 and 40 epochs. The checkpoint
with the lowest validation negative log likelihood is retained. The loss uses
log returns divided by `0.01 * sqrt(days)`; its omitted return-unit Jacobian is
parameter-independent. No checkpoint is chosen using hedge P&L.

The saved artifact is plain JSON with fitted preprocessing, weights, timestamps,
configuration and training history. Inference uses NumPy only; PyTorch is an
optional training dependency. Reproducibility is tested within the recorded
software environment, not promised bit-for-bit across every platform/version.

## Data, features and information boundaries

The existing `market_path` simulator generates three zero-drift GBMs from prices
100, 80, 120, volatilities 20%, 22.5%, 25%, and pairwise correlation 0.35. These
are toy assumptions. Neither true volatilities nor correlations are features
or inputs to model fitting. They generate training observations and serve as
the separate oracle benchmark.

Each path has 20 toy business-day intervals, with observations at 09:00 and
16:00 UTC, weekdays only. Each close before expiry supplies one training row.
The label is the joint log return to the next business-day close: 24 hours
normally, 72 hours over a weekend. Features use only the released prefix:

- Three current log prices.
- The latest three joint open/close log-return bars, each divided by the square
  root of its actual elapsed calendar days: nine values.
- Three presence masks and three log bar durations. Missing inception history
  is zero-filled with a zero mask; it is never backfilled from future prices.
- Log forward horizon in calendar days.

The default split is by entire independent paths, with distinct seed streams
and simulated dates:

| Split | Paths | Labeled decisions | First simulated close | Use |
|---|---:|---:|---|---|
| Train | 1,024 | 20,480 | 2024-01-08 | Fit weights and preprocessing |
| Validation | 256 | 5,120 | 2025-01-06 | Select epoch by likelihood |
| Test | 200 | 4,000 per optimized policy | 2026-01-05 | Frozen financial evaluation |

The model becomes available on 2025-12-31, after all validation targets.
These dates describe independent synthetic paths, not historical observations.
Master seed 20260920 and namespaced SHA256-derived 128-bit seeds separate train,
validation, financial-test and scenario streams. The manifests record every
path seed and a digest of training features, labels and row metadata. No final
test observations are generated until training and statistical validation finish.

At a decision, both providers use the same declared next-rebalance horizon,
capped by a relevant option expiry and strictly after earliest execution
eligibility. They never inspect a future quote or replay iterator. Both use
identical base normal draws per decision and scenario index, and the same inner
pricing parameters and draws. Sample-count changes preserve the scenario prefix.
Learned inference uses current spots and released history; a model not yet
available at the decision is rejected.

Shared `prepare_scenarios` and `reprice_scenarios` preserve the Stage 1 validation
and sign conventions:

```text
delta_v = -sum(fixed_quantity * multiplier * (future_mark - current_mark))
delta_h[j] = multiplier[j] * (future_mark[j] - current_mark[j])
```

Hypothetical expiry scenarios use exact contractual payoffs. They never become
official ledger fixings. The provider returns arrays, not orders, and cannot
write to the portfolio or bypass risk checks. Constituent options can use the
same repricing path; this financial comparison uses constituent stocks only.

To use a frozen model with an existing case and its pricing configuration:

```python
from basket_hedging.learned_model import ConditionalGaussianModel
from basket_hedging.learned_scenarios import LearnedScenarios
from basket_hedging.scenario_hedge import ScenarioHedge

model = ConditionalGaussianModel.load("reports/stage2/model.json")
source = LearnedScenarios(model, pricing, samples=256, seed=9173)
policy = ScenarioHedge(source, hedge_ids, cost_weights, alpha=0.1)
result = engine.run(case.events(), policy)
```

`pricing`, `hedge_ids`, `cost_weights` and `engine` come from the existing case
setup; the generator does not manufacture instruments or alter trading rules.

## Validation and experiment

Before hedging, compare 20,000 generated draws with 20,000 independent simulator
draws in each of eight cases: normal, recent positive, recent negative and
cold-start history, at both weekday 24-hour and Friday-to-Monday 72-hour horizons.
Histories use the same open/close cadence as training. Fixed-volatility GBM has
independent future increments, so past positive/negative moves should not
produce a true change in future volatility. There is no higher-volatility state
to validate in this environment.

`generator_validation.json` contains simple-return means, sample variances,
full covariance/correlation matrices, 1/5/50/95/99% marginal quantiles and
absolute/signed/relative errors. Fixed-share basket returns have the same
checks and empirical one-dimensional Wasserstein distances. These comparisons
include simulation noise; they are not exact valuation or distributional
certificates. Relative mean errors near a zero reference are unstable; use
absolute return errors when interpreting them.

The financial evaluation runs unhedged, scheduled delta, true-simulator
optimizer and learned-generator optimizer on each saved test replay through
the normal case loader. Both optimizers use alpha 0.1, cost weight 0.02 per
share, 256 scenarios and 1,024 inner pricing samples. The common accounting
replay uses 8,192 pricing samples and pricing seed 731. The engine's capital,
fees, spreads, funding, borrow, research margin and timing rules are unchanged.
These choices were fixed before examining test results.

Reports retain every status. Terminal comparisons use only paths completed by
all four policies, with failure frequencies calculated on all paths. Per-path
records also retain the latest common valid pretrade checkpoint. Stopped runs
receive no invented terminal P&L. Unexpected software/input errors fail the
experiment visibly rather than being silently excluded.

Gross residual hedge error is `-(liability P&L + hedge P&L)` before actual
trading/financing costs. Net P&L includes those costs. SD uses `ddof=1`; RMSE
is the square root of mean squared residual. CVaR95 is mean loss among the
worst `ceil(5% * completed matched paths)` terminal outcomes: ten observations
in a fully completed 200-path run. Turnover is gross two-sided currency traded;
fills include exits and exclude origination/settlement. No statistical claim
is made from a single path.

## Run and artifacts

```powershell
python -m pip install -e ".[test,learn]"
python -B -m pytest -p no:cacheprovider -q
python -B -u -m basket_hedging stage2 --output reports/stage2 --test-paths 200 --workers 6
git diff --check
```

Use fewer workers to reduce CPU use; seeds and financial results do not depend
on worker completion order. Small counts in the Python runner exist for
software smoke tests, not as substitutes for the 200-path validation.

The output contains the frozen model, experiment/split manifest, training loss
history, conditional validation, 800 per-policy path records, aggregate hedge
metrics and test replay hashes. Temporary saved replay fixtures are removed
after evaluation; seeds and configuration reproduce them. Large training or
scenario arrays and full traces for all 800 runs are not bundled.

## Limits and next step

This evaluates learning and integration inside the same simple model family.
Gaussian innovations exclude heavy tails, jumps and volatility clustering.
The learned model can introduce spurious history dependence. Pricing is still
fixed-parameter risk-neutral GBM, and Monte Carlo pricing noise remains.
Delayed fills mean the decision-to-next-review objective is an approximation;
the unchanged optimizer does not separate exposure before and after each fill.
Empirical tails from 200 paths are imprecise. None of this establishes real
market profitability, broker compliance or readiness for live trading.

A next stage involving real constituent option surfaces would need released
spot/option data with quote timestamps and conventions, a defined joint surface
state and conditional generator, surface validity checks, consistent repricing,
chronological independent evaluation and actual funding/borrow/margin inputs.
It would also need Polykern's actual liability and hedge-universe requirements.
Those are separate requirements; this change supplies no new market connector,
surface model or trading strategy.
