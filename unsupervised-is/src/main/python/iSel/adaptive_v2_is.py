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
    beta_q2 = clip(1 - frac_q2,  0.05, 0.40)

    # Imbalance proxy (label-free):
    #   large P95/P5 ratio → heavy right tail → likely imbalanced
    imbalance_proxy = tanh( P95 / (|P5| + 1e-9) / 10.0 )

    # Noise removal rate (Q4), shrunk toward 0 when imbalance is detected:
    theta = 0.30 × (1 - imbalance_proxy)

Uniform vs. Weighted Sampling
------------------------------

* **Q1** uses **score-proportional weighted sampling** within the primary
  redundancy zone, favoring instances with the lowest scores (most redundant).
* **Q2** uses **score-proportional weighted sampling** within the secondary
  redundancy zone, giving slightly more probability to instances closer to
  the lower edge of the quartile (i.e. more redundant examples).
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
    scores = np.asarray(scores, dtype=np.float64).ravel()
    eps = 1e-9

    p0, p25, p50, p75, p100 = np.percentile(scores, [0, 25, 50, 75, 100])
    total_range = p100 - p0 + eps

    # Densidade de cada quartil: 25% da massa / largura do intervalo
    w1 = (p25 - p0)  + eps
    w2 = (p50 - p25) + eps
    w3 = (p75 - p50) + eps
    w4 = (p100 - p75) + eps

    d1, d2, d3, d4 = 0.25/w1, 0.25/w2, 0.25/w3, 0.25/w4

    # Fallback: Q3 degenerado (largura < 1% do range total)
    Q3_DEGEN_THRESH = 0.01 * total_range
    q3_degenerate = w3 < Q3_DEGEN_THRESH

    if q3_degenerate:
        # Sem referência confiável → normaliza pelo máximo absoluto
        d_max = max(d1, d2, d3, d4)
        beta_q1 = float(np.clip(d1 / d_max, 0.10, 0.50))
        beta_q2 = float(np.clip(d2 / d_max, 0.05, 0.30))
        offset  = 0.0
        fallback_used = True
    else:
        # Offset adaptativo: mediana dos log-ratios de todos os quartis vs Q3
        log_ratios = np.array([np.log(d1/d3), np.log(d2/d3), np.log(d4/d3)])
        offset = float(np.median(log_ratios))

        # sigmoid(log(di/d3) - offset): > 0.5 quando qi mais denso que a mediana
        beta_q1 = float(np.clip(1/(1+np.exp(-(np.log(d1/d3) - offset))), 0.10, 0.90))
        beta_q2 = float(np.clip(1/(1+np.exp(-(np.log(d2/d3) - offset))), 0.05, 0.40))
        fallback_used = False

    # Theta (imbalance proxy inalterado)
    p5_val  = float(np.percentile(scores, 75))
    p95_val = float(np.percentile(scores, 100))
    imbalance_proxy = float(np.tanh((p95_val / (abs(p5_val) + eps)) / 10.0))
    theta = float(0.8 * (1.0 - imbalance_proxy))

    return dict(
        p0=float(p0), p25=float(p25), p50=float(p50),
        p75=float(p75), p100=float(p100),
        beta_q1=beta_q1, beta_q2=beta_q2, theta=theta,
        imbalance_proxy=imbalance_proxy,
        offset=offset, fallback_used=fallback_used,
        frac_q1=float(w1/total_range), frac_q2=float(w2/total_range),
        total_range=float(total_range - eps),
        d1=float(d1), d2=float(d2), d3=float(d3), d4=float(d4),
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
            kwargs.setdefault('n_epochs', 50)
            kwargs.setdefault('batch_size', 512)
            kwargs.setdefault('bottleneck_ratio', 0.25)
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
