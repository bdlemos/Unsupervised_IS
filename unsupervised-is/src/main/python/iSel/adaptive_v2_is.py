"""
Adaptive V2 Unsupervised Instance Selection (AdaptiveV2-IS)
============================================================

Refactoring of the threshold-estimation strategy from Adaptive-IS using a
purely non-parametric approach based on **quartile density**.

Motivation
----------

The GMM-boundary strategy in Adaptive-IS suffers from two failure modes:

1. **Variance collapse** — when the base scorer (e.g. Autoencoder) produces
   a degenerate spike near zero, the GMM cannot separate meaningful components.

2. **Homogeneous redundancy treatment** — knee-based detection treats the
   entire low-score mass as a single block with one removal rate, ignoring
   that scores may concentrate differently in Q1 vs. Q2.

Core Insight: Quartile Width ≡ Inverse Density
-----------------------------------------------

Each quartile (Q1, Q2, Q3, Q4) contains exactly **25% of the data mass**.
Therefore, the *width* of a quartile on the score axis is inversely
proportional to the local density of that region:

    density(Qi) ∝ 1 / width(Qi)

A **narrow quartile** → 25% of data squeezed into a tiny score interval
→ extremely high density → high redundancy → aggressive pruning.

A **wide quartile** → 25% of data spread over a large score interval
→ low density, more diverse instances → conservative pruning.

Zone Definitions
----------------

* **Q1 (0–25%)** : Primary redundancy zone. Highest density region.
  Pruned with rate ``beta_q1``.
* **Q2 (25–50%)**: Secondary redundancy zone. May still be dense.
  Pruned with rate ``beta_q2`` (capped lower than ``beta_q1``).
* **Q3 (50–75%)**: Informative / transition zone. **No pruning.**
* **Q4 (75–100%)**: Noise zone. Pruned with rate ``theta``, but
  modulated by an imbalance proxy to protect minority-class instances
  that appear as outliers.

Mathematical Formulation
------------------------

Let ``total_range = P100 - P0`` (with floor of ``1e-9`` for stability).

    width_q1 = P25 - P0
    width_q2 = P50 - P25

    # Fraction of total range occupied by each quartile
    frac_q1 = width_q1 / total_range          ∈ [0, 1]
    frac_q2 = width_q2 / total_range          ∈ [0, 1]

    # Removal rates are INVERSELY proportional to width:
    #   narrow (frac → 0) → high density → beta → clip_max
    #   wide   (frac → 1) → low  density → beta → clip_min
    #
    # We use (1 - frac) as the raw signal and rescale into the desired clip range.
    #
    beta_q1 = clip(1 - frac_q1,  0.10, 0.90)
    beta_q2 = clip(1 - frac_q2,  0.05, 0.60)

    # Imbalance proxy (label-free):
    #   large P95/P5 ratio → heavy right tail → likely imbalanced
    imbalance_proxy = tanh( P95 / (|P5| + 1e-9) / 10.0 )

    # Noise removal rate (Q4), shrunk toward 0 when imbalance is detected:
    theta = 0.30 × (1 - imbalance_proxy)

Uniform vs. Weighted Sampling
------------------------------

* **Q1 and Q2** use **uniform random sampling** within each zone —
  there is no strong a-priori reason to prefer removing one redundant
  instance over another within the same quartile.
* **Q4** uses **score-proportional weighted sampling** — within the noise
  band, instances with higher scores are more likely to be true outliers
  and are therefore given a higher removal probability.

Usage
-----

>>> from src.main.python.iSel.adaptive_v2_is import AdaptiveV2IS
>>> selector = AdaptiveV2IS(base_method='autoencoder', n_epochs=10)
>>> selector.fit(X, y)
>>> print(selector.beta_q1_, selector.beta_q2_, selector.theta_)
>>> print(f"Reduction: {selector.reduction_:.2%}")
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.utils.validation import check_X_y

from src.main.python.iSel.base import InstanceSelectionMixin
from src.main.python.iSel import autoencoder_is, gmm_is, perplexity_is


# ---------------------------------------------------------------------------
# A. Quartile-density parameter estimation
# ---------------------------------------------------------------------------

def _estimate_params_quartiles(scores: np.ndarray) -> dict:
    """Estimate instance-selection parameters from the quartile structure
    of the score distribution.

    This is a fully non-parametric approach — no distribution is fitted.
    It is robust to degenerate score distributions (spikes, heavy tails)
    that cause GMM-based estimators to fail.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples,)
        1-D array of per-instance scores.  Lower scores indicate redundant
        instances; higher scores indicate potentially noisy / rare instances.

    Returns
    -------
    dict with keys:
        p0, p25, p50, p75, p100 : float
            Quartile boundary values.
        beta_q1 : float in [0.10, 0.90]
            Removal rate for Q1 (scores ≤ P25).
        beta_q2 : float in [0.05, 0.60]
            Removal rate for Q2 (P25 < scores ≤ P50).
        theta : float in [0.0, 0.30]
            Removal rate for Q4 (scores > P75).
        imbalance_proxy : float in [0, 1)
            Proxy for dataset imbalance derived from the tail ratio.
            Values near 1 indicate likely class imbalance.

    Notes
    -----
    Mathematical derivation of beta_q1 and beta_q2
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Each quartile contains exactly 25% of the data.  Its *width* on the
    score axis reflects local density:

        frac_qi = width_qi / total_range

    Since density ∝ 1/width, a small frac → high density → more redundancy
    → higher removal rate.  We map this via:

        beta_qi = clip(1 - frac_qi, lo, hi)

    When frac_qi → 0 (all 25% squeezed into a point), beta → clip_max.
    When frac_qi → 1 (25% spread over the full range), beta → clip_min.
    The clip bounds encode domain knowledge: we never remove more than 90%
    of Q1 or more than 60% of Q2, and we always remove at least a small
    fraction of each zone.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()

    # ------------------------------------------------------------------ #
    # 1. Quartile boundaries                                               #
    # ------------------------------------------------------------------ #
    p0, p25, p50, p75, p100 = np.percentile(scores, [0, 25, 50, 75, 100])

    total_range = p100 - p0 + 1e-9   # floor prevents division by zero

    # ------------------------------------------------------------------ #
    # 2. Quartile widths as fractions of the total range                   #
    # ------------------------------------------------------------------ #
    width_q1 = p25 - p0
    width_q2 = p50 - p25

    frac_q1 = width_q1 / total_range   # ∈ [0, 1]
    frac_q2 = width_q2 / total_range   # ∈ [0, 1]

    # ------------------------------------------------------------------ #
    # 3. Removal rates — inversely proportional to quartile width          #
    #                                                                      #
    # Intuition:                                                           #
    #   • frac_q1 ≈ 0.02  (Q1 very narrow) → 1 - 0.02 = 0.98 → clip 0.90 #
    #   • frac_q1 ≈ 0.40  (Q1 moderate)    → 1 - 0.40 = 0.60 → beta=0.60 #
    #   • frac_q1 ≈ 0.90  (Q1 very wide)   → 1 - 0.90 = 0.10 → clip 0.10 #
    # ------------------------------------------------------------------ #
    beta_q1 = float(np.clip(1.0 - frac_q1, 0.10, 0.70))
    beta_q2 = float(np.clip(1.0 - frac_q2, 0.05, 0.60))

    # ------------------------------------------------------------------ #
    # 4. Imbalance proxy — label-free tail-ratio heuristic                 #
    #                                                                      #
    #   imbalance_proxy = tanh( P95 / (|P5| + ε) / 10 )                  #
    #                                                                      #
    # A large P95/P5 ratio signals a heavy right tail, which is typical   #
    # when minority-class instances concentrate at high scores.            #
    # tanh compresses the ratio to [0, 1).                                 #
    # ------------------------------------------------------------------ #
    p5  = float(np.percentile(scores, 5))
    p95 = float(np.percentile(scores, 95))

    tail_ratio      = p95 / (abs(p5) + 1e-9)
    imbalance_proxy = float(np.tanh(tail_ratio / 10.0))

    # ------------------------------------------------------------------ #
    # 5. Theta — noise removal rate, shrunk by imbalance                   #
    #                                                                      #
    #   theta = 0.30 × (1 - imbalance_proxy)                              #
    #                                                                      #
    # When imbalance ≈ 1 (high imbalance risk) → theta ≈ 0.              #
    # When imbalance ≈ 0 (balanced distribution) → theta ≈ 0.30.          #
    # ------------------------------------------------------------------ #
    theta = float(0.30 * (1.0 - imbalance_proxy))

    return dict(
        p0=float(p0),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p100=float(p100),
        beta_q1=beta_q1,
        beta_q2=beta_q2,
        theta=theta,
        imbalance_proxy=imbalance_proxy,
        # diagnostics
        frac_q1=float(frac_q1),
        frac_q2=float(frac_q2),
        total_range=float(total_range - 1e-9),  # original range for display
    )


# ---------------------------------------------------------------------------
# B. Quartile-based selection mask
# ---------------------------------------------------------------------------

def _apply_selection_quartiles(
    scores: np.ndarray,
    beta_q1: float,
    beta_q2: float,
    theta: float,
    p25: float,
    p50: float,
    p75: float,
    random_state: int = 0,
) -> np.ndarray:
    """Apply probabilistic instance removal using the quartile-density strategy.

    Zones
    -----
    * **Q1** (scores ≤ p25)       : removed uniformly at rate ``beta_q1``.
    * **Q2** (p25 < scores ≤ p50) : removed uniformly at rate ``beta_q2``.
    * **Q3** (p50 < scores ≤ p75) : **untouched** (informative zone).
    * **Q4** (scores > p75)       : removed with score-proportional weights
                                    at rate ``theta``.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples,)
        Per-instance scores.
    beta_q1 : float
        Fraction of Q1 instances to remove (uniform sampling).
    beta_q2 : float
        Fraction of Q2 instances to remove (uniform sampling).
    theta : float
        Fraction of Q4 instances to remove (weighted sampling).
    p25, p50, p75 : float
        Quartile boundaries from ``_estimate_params_quartiles``.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    mask : np.ndarray of bool, shape (n_samples,)
        ``True`` → instance is **kept**; ``False`` → instance is **removed**.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n      = len(scores)
    mask   = np.ones(n, dtype=bool)

    rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------ #
    # Q1 — uniform removal                                                 #
    # ------------------------------------------------------------------ #
    q1_idx = np.where(scores <= p25)[0]
    n_remove_q1 = max(0, int(len(q1_idx) * beta_q1))

    if n_remove_q1 > 0 and len(q1_idx) > 0:
        to_remove = rng.choice(q1_idx, size=n_remove_q1, replace=False)
        mask[to_remove] = False
        print(
            f"[AdaptiveV2-IS] Q1 ({len(q1_idx)} instances): "
            f"removing {n_remove_q1} (beta_q1={beta_q1:.3f}, uniform)"
        )

    # ------------------------------------------------------------------ #
    # Q2 — uniform removal                                                 #
    # ------------------------------------------------------------------ #
    q2_idx = np.where((scores > p25) & (scores <= p50))[0]
    n_remove_q2 = max(0, int(len(q2_idx) * beta_q2))

    if n_remove_q2 > 0 and len(q2_idx) > 0:
        to_remove = rng.choice(q2_idx, size=n_remove_q2, replace=False)
        mask[to_remove] = False
        print(
            f"[AdaptiveV2-IS] Q2 ({len(q2_idx)} instances): "
            f"removing {n_remove_q2} (beta_q2={beta_q2:.3f}, uniform)"
        )

    # ------------------------------------------------------------------ #
    # Q3 — fully preserved (no code needed)                               #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Q4 — score-proportional weighted removal                             #
    # ------------------------------------------------------------------ #
    q4_idx = np.where(scores > p75)[0]
    n_remove_q4 = max(0, int(len(q4_idx) * theta))

    if n_remove_q4 > 0 and len(q4_idx) > 0:
        # Weight ∝ (score - p75): higher excess above p75 → higher removal prob.
        raw_weights = scores[q4_idx] - p75
        raw_weights = np.clip(raw_weights, 0.0, None) + 1e-9
        weights     = raw_weights / raw_weights.sum()

        rng_q4    = np.random.RandomState(random_state + 1)
        to_remove = rng_q4.choice(q4_idx, size=n_remove_q4, replace=False, p=weights)
        mask[to_remove] = False
        print(
            f"[AdaptiveV2-IS] Q4 ({len(q4_idx)} instances): "
            f"removing {n_remove_q4} (theta={theta:.3f}, weighted)"
        )

    return mask


# ---------------------------------------------------------------------------
# C. AdaptiveV2IS selector (InstanceSelectionMixin-compatible)
# ---------------------------------------------------------------------------

class AdaptiveV2IS(InstanceSelectionMixin):
    """Adaptive V2 Unsupervised Instance Selection (AdaptiveV2-IS).

    Uses quartile-density estimation to automatically derive removal rates
    for redundant (Q1, Q2) and potentially noisy (Q4) instances — without
    requiring class labels.

    Parameters
    ----------
    base_method : {'autoencoder', 'gmm', 'perplexity'}, default='autoencoder'
        Scorer used to compute per-instance scores.

    random_state : int, default=0
        Random seed for reproducibility.

    **base_kwargs
        Extra keyword arguments forwarded to the base scorer constructor
        (e.g. ``n_epochs=10``, ``batch_size=64`` for the autoencoder).

    Attributes
    ----------
    beta_q1_ : float
        Estimated removal rate for Q1 (primary redundancy).
    beta_q2_ : float
        Estimated removal rate for Q2 (secondary redundancy).
    theta_ : float
        Estimated removal rate for Q4 (noise), imbalance-adjusted.
    imbalance_proxy_ : float
        Score-derived imbalance proxy in [0, 1).
    scores_ : ndarray of shape (n_samples,)
        Raw scores from the base scorer.
    mask : ndarray of bool, shape (n_samples,)
        Boolean selection mask.
    sample_indices_ : ndarray
        Indices of kept instances.
    reduction_ : float
        Fraction of instances removed.
    """

    def __init__(
        self,
        base_method: Literal['autoencoder', 'gmm', 'perplexity'] = 'autoencoder',
        random_state: int = 0,
        **base_kwargs,
    ) -> None:
        self.base_method     = base_method
        self.random_state    = random_state
        self.base_kwargs     = base_kwargs
        self.sample_indices_ = []

    # ------------------------------------------------------------------
    # Internal: build base scorer with neutral thresholds (no removal)
    # ------------------------------------------------------------------

    def _build_base_scorer(self):
        kwargs = dict(self.base_kwargs)
        kwargs.update(low_percentile=0.0, high_percentile=100.0, beta=0.0, theta=0.0)

        if self.base_method == 'autoencoder':
            kwargs.setdefault('n_epochs', 10)
            kwargs.setdefault('batch_size', 64)
            kwargs.setdefault('bottleneck_ratio', 0.05)
            return autoencoder_is.AutoencoderIS(random_state=self.random_state, **kwargs)

        elif self.base_method == 'gmm':
            return gmm_is.GMMIS(random_state=self.random_state, **kwargs)

        elif self.base_method == 'perplexity':
            kwargs.setdefault('n_topics', 10)
            return perplexity_is.PerplexityIS(random_state=self.random_state, **kwargs)

        raise ValueError(
            f"Unknown base_method='{self.base_method}'. "
            "Choose from 'autoencoder', 'gmm', 'perplexity'."
        )

    def _extract_scores(self, scorer) -> np.ndarray:
        if self.base_method == 'autoencoder':
            return scorer.reconstruction_errors_
        elif self.base_method == 'gmm':
            return scorer.gmm_scores_
        elif self.base_method == 'perplexity':
            return scorer.perplexities_
        raise ValueError(f"Cannot extract scores for base_method='{self.base_method}'")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def select_data(self, X, y):
        """Run the full AdaptiveV2-IS pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Labels — **not used** for selection; kept for API compatibility.

        Returns
        -------
        X_ : ndarray — selected instances.
        y_ : ndarray — corresponding labels.
        """
        X, y = check_X_y(X, y, accept_sparse="csr")
        n_samples = len(y)

        # Step 1 — score computation
        print(f"[AdaptiveV2-IS] Step 1 — scoring via '{self.base_method}' …")
        scorer = self._build_base_scorer()
        scorer.fit(X, y)
        scores = self._extract_scores(scorer)
        self.scores_       = scores.copy()
        self.base_scorer_  = scorer

        print(
            f"[AdaptiveV2-IS] Score stats — "
            f"min: {scores.min():.4f}, "
            f"p25: {np.percentile(scores, 25):.4f}, "
            f"median: {np.median(scores):.4f}, "
            f"p75: {np.percentile(scores, 75):.4f}, "
            f"max: {scores.max():.4f}"
        )

        # Step 2 — quartile-density parameter estimation
        print("[AdaptiveV2-IS] Step 2 — estimating params via quartile density …")
        params = _estimate_params_quartiles(scores)

        self.beta_q1_        = params['beta_q1']
        self.beta_q2_        = params['beta_q2']
        self.theta_          = params['theta']
        self.imbalance_proxy_ = params['imbalance_proxy']
        self.p25_            = params['p25']
        self.p50_            = params['p50']
        self.p75_            = params['p75']

        # Expose as plain attrs for run_generateSplit.py logging compatibility
        self.beta  = (self.beta_q1_ + self.beta_q2_) / 2.0  # summary
        self.theta = self.theta_
        self.low_percentile  = 25.0   # Q1+Q2 = bottom 50%
        self.high_percentile = 75.0   # Q4 = top 25%

        print(
            f"[AdaptiveV2-IS] Quartile widths: "
            f"Q1_frac={params['frac_q1']:.3f}, "
            f"Q2_frac={params['frac_q2']:.3f}, "
            f"total_range={params['total_range']:.4f}"
        )
        print(
            f"[AdaptiveV2-IS] Params: "
            f"beta_q1={self.beta_q1_:.3f}, "
            f"beta_q2={self.beta_q2_:.3f}, "
            f"theta={self.theta_:.3f}, "
            f"imbalance_proxy={self.imbalance_proxy_:.3f}"
        )

        # Step 3 — apply quartile-based selection
        print("[AdaptiveV2-IS] Step 3 — applying quartile selection …")
        self.mask = _apply_selection_quartiles(
            scores=scores,
            beta_q1=self.beta_q1_,
            beta_q2=self.beta_q2_,
            theta=self.theta_,
            p25=self.p25_,
            p50=self.p50_,
            p75=self.p75_,
            random_state=self.random_state,
        )

        self.X_              = np.asarray(X[self.mask])
        self.y_              = np.asarray(y[self.mask])
        self.sample_indices_ = np.where(self.mask)[0]
        self.reduction_      = 1.0 - float(len(self.y_)) / n_samples

        n_removed = n_samples - len(self.y_)
        print(
            f"[AdaptiveV2-IS] Done — removed {n_removed} / {n_samples} "
            f"(reduction = {self.reduction_:.2%})"
        )

        return self.X_, self.y_
