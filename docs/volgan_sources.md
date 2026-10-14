# VolGAN source and implementation record

This is a **multi-asset VolGAN adaptation**, independently written in PyTorch.
It is not an exact reproduction of the single-underlying SPX experiment.
One shared conditional generator and discriminator process the whole dated
SPY/QQQ/IWM state. Joint rows remain aligned; asset-specific shuffling and
independent generation would remove dependence and are not used.

## Sources and permission

- Vuletić and Cont, *VolGAN: A Generative Model for Arbitrage-Free Implied
  Volatility Surfaces*, DOI [10.1080/1350486X.2025.2471317](https://doi.org/10.1080/1350486X.2025.2471317),
  sections 3.1–3.4, equations 9–14. Published article at
  [Oxford Research Archive](https://ora.ox.ac.uk/objects/uuid%3A001da2f7-8b4d-45b4-bb1c-cf456af000e3/files/r5m60qt79n).
  Direct retrieval returned HTTP 403; the primary publisher/Oxford indexed text
  supplied the relevant equations and numerical settings.
- Cont and Vuletić, *Data-driven hedging with generative models*,
  [section 4](https://link.springer.com/article/10.1007/s10479-025-06867-3):
  the hedging experiment uses raw generator output without scenario reweighting.
- Authors' [public implementation](https://github.com/milenavuletic/VolGAN/tree/a93f63362f2540954743ceded4f11ca1f14a9e6b),
  pinned commit `a93f63362f2540954743ceded4f11ca1f14a9e6b`.
  Inspected `VolGAN.py`, `VolGAN-example.py`, and the repository tree.
  No LICENSE file exists at that commit; GitHub's license metadata is null.
  No upstream source code is copied or vendored. PyTorch is already this
  repository's ML framework and is also used upstream; no framework port was made.

## Verified choices

The published generator has Softplus hidden layers of widths 16 and 32 and an
affine output. The discriminator has a width-16 Softplus layer and sigmoid
output. Latent noise is 32 independent standard normals shared by all assets.
Inputs contain two recent scaled returns, 21-session realized volatility and
log IV. Outputs contain scaled next-session returns and log-IV increments.
The public implementation's square-root-of-252 return scale is inverted at
inference. No fitted standardization is applied.

The paper's log-surface finite-difference penalties operate separately along
moneyness and maturity. Here they are summed across assets. BCE gradient-norm
matching calibrates both coefficients over 25 epochs. Networks and optimizers
are restarted from the original initialization, then trained for 10,000 epochs
with RMSprop, learning rate 0.0001 and batches of 100. Every minibatch gets one
discriminator and one generator update. The fixed final checkpoint is retained.

## Explicit discrepancies and adaptations

1. The published grid has 11 moneyness nodes and 9 maturity nodes; the public
   example uses an older 10-by-8 grid. The published grid is the default here.
2. Public calibration exponentiates log IV before penalizing it, but its later
   training loop penalizes log IV. Both stages here use the paper's log-IV
   equations. Grid-axis finite differences are explicit, avoiding the public
   flattened-index/axis inconsistencies.
3. The public wrapper continues training after calibration. The paper specifies
   a restart; this implementation restores network weights and recreates both
   optimizers. Its generator BCE is multiplied by one half as in equation 13;
   the public implementation omits that factor.
4. Realized volatility includes the latest available return in the trailing
   21-session window, consistent with the paper's condition. The public loader's
   index slices omit that most recent return.
5. The user-authorized multi-asset change expands input/output dimensions and
   sums smoothness across assets. Hidden widths and shared latent dimension
   remain unchanged. This permits joint dependence; it does not guarantee that
   a trained model accurately reproduces correlations or tails.
6. The ETF source, chronology and surface preparation differ from OptionMetrics.
   Vendor ETF IV is a market-state input; this is not a claim that American ETF
   option quotes are European prices. No option trades are introduced.

Sampling represents one consecutive observed exchange session. A weekend does
not multiply GAN output by a Brownian square-root-of-time factor. Rates, costs,
the regression intercept and margin do not enter neural training. There is no
arbitrage penalty in training and no post-generation reweighting; raw scenarios
need not be arbitrage-free. Model files explicitly store the model family,
grid, asset order, source commit, training input hash, dates, settings and both
networks. A request for VolGAN rejects Gaussian checkpoints.

Tiny artificial arrays in tests establish transform, chronology and training
mechanics only. They establish neither historical calibration nor hedging
performance. Actual training and benchmark status belongs in the run manifest.
