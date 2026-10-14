# Read-only pre-change VolGAN diagnosis

The probe evaluates training rows through 2017 and the 227 development targets
from 2018-01-03 through 2018-12-31 only. No 2019 confirmation outcomes or
2020–2025 hedge results were evaluated. Original checkpoint/data hashes are
unchanged. Probe command: `python -B probe.py` (9.68 seconds here).

## A. Demonstrated implementation defects

None found in the inspected alignment, transforms, output slices or training
update mechanics.

- All 1,592 training rows reproduce the frozen input hash
  `338649efb84806a4da0b71c4d3b3b434826e3e3d9dede8765b4d0b33d74f2c7e`.
- Independent direct formulas check every retained training and 2018 row.
  Maximum feature error is 1.11e-16, label error zero, price round-trip error
  2.27e-13 and IV round-trip error 5.55e-17. Current and historical observations
  form features; the next-session label is distinct. Asset blocks and maturity-
  major surface ordering agree at input, output and inverse transformation.
- One joint row shuffle preserves asset/date alignment. There is one shared
  latent vector and one joint discriminator. Return output rows are distinct,
  with no broadcasting/duplicated slices.
- Every asset reacts to latent noise. At the first development state, return
  latent-Jacobian norms are 0.03096/0.03577/0.03435 in scaled-output units;
  log-IV block norms are 0.10262/0.10390/0.09827. All three return-Jacobian rows
  and final-layer return weight rows differ.
- Production updates detach fake observations for the discriminator, update D,
  then backpropagate the generator loss through D. Both optimizers are reset
  after calibration. D gradients accumulated during the G pass are cleared
  before the next D update and do not change the G gradient or D parameters.
- One isolated update on deep copies, a training batch and fresh optimizers
  changes all six G parameter tensors and all four D tensors with finite
  gradients. These copies were discarded; this is not a trained candidate.

## B. Demonstrated training/model weaknesses

The frozen final model has weak dispersion and nearly one-dimensional joint
noise responses for QQQ/IWM. On the 227 development states with 512 shared
draws per state (seed 4402), central 90% coverage is 45.81%/59.03%/67.84%.
This differs from the existing combined-2018/2019 diagnostic because both
the evaluated segment and reproducible draws differ.

Across states, conditional QQQ/IWM correlation ranges from 0.99596 to 0.99976,
with median 0.99945. Conditional QQQ-minus-IWM log-return standard deviation
has median 0.000487, ranging from 0.000297 to 0.001361. Observed spread SD
over dates is 0.007447. These are explicitly different conditional and
unconditional quantities; their difference is not an equality constraint.

Generated conditional means also vary by date. The numerical total-covariance
identity (mean conditional covariance plus covariance of conditional means)
reconciles with pooled generated covariance to 4.91e-20. Matching the mean
conditional covariance directly to observed unconditional covariance would
be incorrect.

The fixed final-epoch selection is an avoidable limitation: training BCE
near equilibrium does not identify the best predictive joint distribution.
The current checkpoint stores 103 loss snapshots but no intermediate weight
or scenario checkpoints. The epoch-100 through epoch-10000 losses alone do
not reveal whether the weak return distribution appeared early or late.

## C. Unproven causal hypotheses

Small hidden width, adversarial equilibrium and smoothness trade-offs could
matter, but none is established as the cause by these checks. Do not call
the network size a bug or force a covariance/correlation target.

For one fixed training batch, the frozen calibrated moneyness/maturity
weights are 7.75944e-6 and 1.71637e-6. Their weighted gradient norms relative
to the adversarial gradient are 4.34/4.27 at seeded initialization and
0.364/0.0915 at the final checkpoint. Both networks have finite, nonzero
gradients. This is a local diagnostic, not proof of the balance throughout
the historical trajectory or proof that smoothness caused weak returns.

## Smallest justified correction

Keep architecture, transformations, losses and all training settings. Add
a small predetermined within-budget checkpoint schedule, selecting on a
proper joint return score over 2018 with explicit calibration/surface
guardrails. Repeat at two predetermined seeds including 1729, freeze the
choice, then inspect 2019 once. This tests the demonstrated selection
weakness without asserting an unsupported architectural explanation.

Save scheduled dispersion and dependence checks to distinguish early and
late weakness during the bounded reruns. Do not infer successful calibration
from BCE, nominal coverage alone or hedge-test outcomes.
