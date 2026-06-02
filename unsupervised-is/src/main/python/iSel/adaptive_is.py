"""
Adaptive Unsupervised Instance Selection (Adaptive-IS)
=======================================================

A meta-wrapper that **automatically estimates** the hyperparameters
``low_percentile``, ``high_percentile``, ``beta`` and ``theta`` from
the *intrinsic structure* of the score distribution produced by any base
scorer (Autoencoder, GMM, Perplexity-IS).

Motivation
----------

Fixed percentile thresholds ignore dataset-specific properties:

* A heavily **imbalanced** dataset produces score distributions with
  heavy tails — naively removing high-score instances destroys the
  minority class.
* A **balanced** but noisy dataset may need aggressive noise removal.

The adaptive approach uses two complementary strategies to estimate
thresholds without labels:

1. **GMM boundary** (default): fits a 1-D Gaussian Mixture Model on the
   scores to discover natural groups (redundant / informative / noisy).
   The cluster boundaries become the thresholds.

2. **Distribution statistics** (fallback): infers thresholds from
   skewness, excess kurtosis, and the tail ratio of the score
   distribution.

In both cases the **imbalance proxy** — derived from the tail ratio of
the score distribution — modulates ``theta`` downward: the more
asymmetric the score distribution, the less aggressive the noise
removal, protecting minority-class instances.

Usage
-----

>>> from src.main.python.iSel.adaptive_is import AdaptiveIS
>>> selector = AdaptiveIS(base_method='autoencoder')
>>> selector.fit(X, y)
>>> print(selector.low_percentile_, selector.high_percentile_)
>>> print(selector.beta_, selector.theta_)
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import scipy
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_X_y

from src.main.python.iSel.base import InstanceSelectionMixin
from src.main.python.iSel import autoencoder_is, gmm_is, perplexity_is


# ---------------------------------------------------------------------------
# Helper: 1-D knee detector (max distance from chord)
# ---------------------------------------------------------------------------

def _find_knee(values: np.ndarray) -> int:
    """Return the index of the 'elbow' / 'knee' in a sorted 1-D curve.

    Uses the perpendicular-distance method: find the point maximally
    distant from the straight line connecting the first and last points.

    Parameters
    ----------
    values : 1-D array, assumed sorted ascending.

    Returns
    -------
    int – index of the knee point.
    """
    n = len(values)
    if n < 3:
        return n // 2

    x = np.linspace(0, 1, n)
    y = (values - values.min()) / (values.ptp() + 1e-12)

    # Direction vector of the chord (start → end)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    norm = np.hypot(dx, dy) + 1e-12

    # Perpendicular distance from each point to the chord
    distances = np.abs(dy * x - dx * y + x[-1] * y[0] - y[-1] * x[0]) / norm
    return int(np.argmax(distances))


# ---------------------------------------------------------------------------
# Helper: score-based imbalance proxy
# ---------------------------------------------------------------------------

def _imbalance_proxy(scores: np.ndarray) -> float:
    """Estimate how imbalanced the dataset might be from the score distribution.

    Uses the ratio of the 95th to 5th percentile of the scores.  A large
    ratio indicates a heavy-tailed / highly asymmetric distribution which
    typically occurs when rare classes produce very different scores from
    the majority.

    Returns
    -------
    float in [0, 1) — the higher, the more likely imbalanced.
    """
    p5  = np.percentile(scores, 5)
    p95 = np.percentile(scores, 95)
    tail_ratio = (p95 + 1e-9) / (np.abs(p5) + 1e-9)
    return float(np.tanh(tail_ratio / 10.0))


# ---------------------------------------------------------------------------
# Core adaptive parameter estimation
# ---------------------------------------------------------------------------

def _estimate_params_gmm(
    scores: np.ndarray,
    n_components: int = 3,
    random_state: int = 0,
) -> dict:
    """Estimate thresholds using a 1-D GMM fitted on the scores.

    Fits a GMM with ``n_components`` Gaussians on the 1-D score vector.
    Components are ordered by their mean.  The boundary between the
    *lowest-mean* component (redundant) and the middle becomes
    ``low_threshold_``, and the boundary between the middle and the
    *highest-mean* component (noise) becomes ``high_threshold_``.

    Parameters
    ----------
    scores : 1-D array of instance scores.
    n_components : number of Gaussian components (2 or 3).
    random_state : random seed.

    Returns
    -------
    dict with keys: low_percentile, high_percentile, beta, theta,
                    low_threshold_, high_threshold_, component_means_
    """
    n_components = max(2, min(n_components, len(scores) // 10))

    gm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        n_init=3,
        max_iter=200,
    )
    gm.fit(scores.reshape(-1, 1))

    # Sort components by mean
    order   = np.argsort(gm.means_.ravel())
    means   = gm.means_.ravel()[order]
    weights = gm.weights_[order]

    # Classify scores into components
    labels = gm.predict(scores.reshape(-1, 1))
    # Re-map labels to sorted order
    label_map = {old: new for new, old in enumerate(order)}
    labels_sorted = np.array([label_map[l] for l in labels])

    # Boundary between component 0 (redundant) and component 1
    low_threshold = float(means[0] + 2.0 * np.sqrt(gm.covariances_.ravel()[order[0]]))

    # Boundary between last two components
    if n_components >= 3:
        high_threshold = float(means[-2] + 2.0 * np.sqrt(gm.covariances_.ravel()[order[-2]]))
    else:
        # With 2 components use the midpoint between the two means
        high_threshold = float((means[0] + means[-1]) / 2.0)

    # Translate thresholds to percentiles
    low_percentile  = float(np.mean(scores <= low_threshold)  * 100.0)
    high_percentile = float(np.mean(scores <= high_threshold) * 100.0)

    # Clamp to [5, 70] and [70, 99]
    low_percentile  = float(np.clip(low_percentile, 5.0, 70.0))
    high_percentile = float(np.clip(high_percentile, max(low_percentile + 5.0, 70.0), 99.0))

    # Derive beta / theta from component weights
    # beta: weight of the redundant component (more redundant instances → be more aggressive)
    beta = float(np.clip(weights[0] * 1.5, 0.05, 0.60))

    # theta: modulated by imbalance proxy (heavy tail → protect high-score instances)
    imbalance = _imbalance_proxy(scores)
    base_theta = float(np.clip(weights[-1] * 1.5, 0.0, 0.40))
    theta = float(base_theta * (1.0 - imbalance))

    return dict(
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        beta=beta,
        theta=theta,
        low_threshold_=low_threshold,
        high_threshold_=high_threshold,
        component_means_=means.tolist(),
        imbalance_proxy_=imbalance,
    )


def _estimate_params_stats(scores: np.ndarray) -> dict:
    """Estimate thresholds from summary statistics (fallback strategy).

    Uses skewness, excess kurtosis, and the knee of the sorted CDF to
    derive thresholds when the GMM approach is not preferred.

    Returns
    -------
    dict with keys: low_percentile, high_percentile, beta, theta.
    """
    from scipy.stats import skew, kurtosis

    n = len(scores)
    sorted_scores = np.sort(scores)

    # --- low_percentile via knee detection on sorted CDF ---
    knee_idx = _find_knee(sorted_scores)
    low_percentile = float(knee_idx / n * 100.0)
    low_percentile = float(np.clip(low_percentile, 5.0, 60.0))

    # high_percentile: symmetric about the median (conservative)
    high_percentile = float(np.clip(100.0 - low_percentile * 0.5, 70.0, 99.0))

    # --- beta from skewness ---
    sk = float(skew(scores))
    # Positive skew → many low scores (large dense region) → be more aggressive
    beta = float(np.clip(0.10 + 0.25 * np.tanh(sk - 0.5), 0.05, 0.50))

    # --- theta from excess kurtosis + imbalance proxy ---
    ek = float(kurtosis(scores))  # excess kurtosis
    imbalance = _imbalance_proxy(scores)
    # Heavy tails (high kurtosis) + imbalanced → theta near 0
    base_theta = float(np.clip(0.20 * (1.0 - np.tanh(ek / 3.0)), 0.0, 0.30))
    theta = float(base_theta * (1.0 - imbalance))

    return dict(
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        beta=beta,
        theta=theta,
        imbalance_proxy_=imbalance,
    )


# ---------------------------------------------------------------------------
# AdaptiveIS selector
# ---------------------------------------------------------------------------

class AdaptiveIS(InstanceSelectionMixin):
    """Adaptive Unsupervised Instance Selection (Adaptive-IS)

    Automatically estimates ``low_percentile``, ``high_percentile``,
    ``beta`` and ``theta`` from the score distribution produced by a
    chosen base scorer, then applies the standard selection pipeline.

    The method is completely unsupervised: no class labels are used at
    any point.

    Parameters
    ----------
    base_method : {'autoencoder', 'gmm', 'perplexity'}
        The underlying scorer used to compute per-instance scores.
        * ``'autoencoder'`` — reconstruction error (AE-IS).
        * ``'gmm'``         — negative log-likelihood (GMM-IS).
        * ``'perplexity'``  — LDA perplexity (Perplexity-IS).

    adaptive_strategy : {'gmm_boundary', 'distribution_stats'}
        Strategy used to estimate thresholds from the scores.
        * ``'gmm_boundary'``       — fit a 1-D GMM on the scores
          (recommended, data-driven).
        * ``'distribution_stats'`` — use skewness/kurtosis/knee
          (faster, less parameters).

    n_gmm_components : int, default=3
        Number of 1-D Gaussian components for ``'gmm_boundary'``
        strategy.  Ignored for ``'distribution_stats'``.

    random_state : int or None, default=0
        Random seed passed to all internal estimators.

    **base_kwargs
        Additional keyword arguments forwarded to the base scorer
        constructor (e.g. ``n_epochs=10`` for autoencoder,
        ``n_topics=12`` for perplexity).

    Attributes
    ----------
    low_percentile_ : float
        Estimated redundancy percentile threshold.
    high_percentile_ : float
        Estimated noise percentile threshold.
    beta_ : float
        Estimated redundancy removal rate.
    theta_ : float
        Estimated noise removal rate.
    imbalance_proxy_ : float
        Estimated imbalance proxy in [0, 1).  Values close to 1 indicate
        likely class imbalance.
    scores_ : ndarray of shape (n_samples,)
        Raw per-instance scores from the base scorer.
    mask : ndarray of shape (n_samples,)
        Boolean mask of selected instances.
    sample_indices_ : ndarray
        Indices of the selected instances in the original dataset.
    reduction_ : float
        Fraction of instances removed.
    """

    def __init__(
        self,
        base_method: Literal['autoencoder', 'gmm', 'perplexity'] = 'autoencoder',
        adaptive_strategy: Literal['gmm_boundary', 'distribution_stats'] = 'gmm_boundary',
        n_gmm_components: int = 3,
        random_state: int = 0,
        **base_kwargs,
    ) -> None:
        self.base_method        = base_method
        self.adaptive_strategy  = adaptive_strategy
        self.n_gmm_components   = n_gmm_components
        self.random_state       = random_state
        self.base_kwargs        = base_kwargs
        self.sample_indices_    = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_base_scorer(self, dummy_low, dummy_high):
        """Instantiate the base scorer with neutral thresholds (no removal).

        We use percentile=0/100 and beta=theta=0 so that the base scorer
        runs fully but removes *nothing* — we only want the raw scores.
        """
        kwargs = dict(self.base_kwargs)
        kwargs.update(
            low_percentile=0.0,
            high_percentile=100.0,
            beta=0.0,
            theta=0.0,
        )
        if self.base_method == 'autoencoder':
            kwargs.setdefault('n_epochs', 10)
            kwargs.setdefault('batch_size', 64)
            kwargs.setdefault('bottleneck_ratio', 0.05)
            return autoencoder_is.AutoencoderIS(
                random_state=self.random_state, **kwargs
            )
        elif self.base_method == 'gmm':
            return gmm_is.GMMIS(
                random_state=self.random_state, **kwargs
            )
        elif self.base_method == 'perplexity':
            kwargs.setdefault('n_topics', 'auto')
            return perplexity_is.PerplexityIS(
                random_state=self.random_state, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown base_method='{self.base_method}'. "
                "Choose from 'autoencoder', 'gmm', 'perplexity'."
            )

    def _extract_scores(self, scorer) -> np.ndarray:
        """Retrieve raw per-instance scores from a fitted base scorer."""
        if self.base_method == 'autoencoder':
            return scorer.reconstruction_errors_
        elif self.base_method == 'gmm':
            return scorer.gmm_scores_
        elif self.base_method == 'perplexity':
            return scorer.perplexities_
        raise ValueError(f"Cannot extract scores for base_method='{self.base_method}'")

    def _estimate_params(self, scores: np.ndarray) -> dict:
        """Dispatch to the configured adaptive strategy."""
        if self.adaptive_strategy == 'gmm_boundary':
            try:
                return _estimate_params_gmm(
                    scores,
                    n_components=self.n_gmm_components,
                    random_state=self.random_state,
                )
            except Exception as exc:
                warnings.warn(
                    f"[Adaptive-IS] GMM boundary estimation failed ({exc}). "
                    "Falling back to distribution_stats.",
                    UserWarning,
                )
                return _estimate_params_stats(scores)

        elif self.adaptive_strategy == 'distribution_stats':
            return _estimate_params_stats(scores)

        raise ValueError(
            f"Unknown adaptive_strategy='{self.adaptive_strategy}'. "
            "Choose 'gmm_boundary' or 'distribution_stats'."
        )

    def _apply_selection(
        self,
        scores: np.ndarray,
        y: np.ndarray,
        low_percentile: float,
        high_percentile: float,
        beta: float,
        theta: float,
    ):
        """Apply thresholding logic to the scores and build the mask.

        Mirrors the logic in AutoencoderIS / GMMIS / PerplexityIS so that
        the AdaptiveIS result is directly comparable.
        """
        n = len(scores)
        mask = np.ones(n, dtype=bool)
        rng  = np.random.RandomState(self.random_state)

        # --- Redundancy (low score band) ---
        low_thr = np.percentile(scores, low_percentile)
        redundant_mask = scores < low_thr
        redundant_idx  = np.where(redundant_mask)[0]

        if beta > 0 and len(redundant_idx) > 0:
            weights = 1.0 / (scores[redundant_idx] + 1e-12)
            weights /= weights.sum()
            n_remove = max(0, int(len(redundant_idx) * beta))
            print(
                f"[Adaptive-IS] Redundant candidates: {len(redundant_idx)}, "
                f"removing {n_remove} (beta={beta:.3f})"
            )
            if n_remove > 0:
                to_remove = rng.choice(
                    redundant_idx, size=n_remove, replace=False, p=weights
                )
                mask[to_remove] = False

        # --- Noise (high score band) ---
        high_thr = np.percentile(scores, high_percentile)
        noise_mask = scores > high_thr
        noise_idx  = np.where(noise_mask)[0]

        if theta > 0 and len(noise_idx) > 0:
            weights = scores[noise_idx] - scores[noise_idx].min() + 1e-12
            weights /= weights.sum()
            n_remove = max(0, int(len(noise_idx) * theta))
            print(
                f"[Adaptive-IS] Noise candidates: {len(noise_idx)}, "
                f"removing {n_remove} (theta={theta:.3f})"
            )
            if n_remove > 0:
                rng2 = np.random.RandomState(self.random_state + 1)
                to_remove = rng2.choice(
                    noise_idx, size=n_remove, replace=False, p=weights
                )
                mask[to_remove] = False

        return mask, low_thr, high_thr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_data(self, X, y):
        """Run the full adaptive instance selection pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances (sparse or dense).
        y : array-like of shape (n_samples,)
            Labels — **not used** for selection; passed through for API
            compatibility.

        Returns
        -------
        X_ : ndarray — selected instances.
        y_ : ndarray — corresponding labels.
        """
        X, y = check_X_y(X, y, accept_sparse="csr")
        n_samples = len(y)

        print(f"[Adaptive-IS] Step 1 — computing scores via '{self.base_method}' …")
        scorer = self._build_base_scorer(dummy_low=0.0, dummy_high=100.0)
        scorer.fit(X, y)
        scores = self._extract_scores(scorer)
        self.scores_ = scores.copy()
        self.base_scorer_ = scorer

        print(
            f"[Adaptive-IS] Score stats — "
            f"min: {scores.min():.4f}, "
            f"median: {np.median(scores):.4f}, "
            f"max: {scores.max():.4f}"
        )

        print(
            f"[Adaptive-IS] Step 2 — estimating thresholds via "
            f"'{self.adaptive_strategy}' …"
        )
        params = self._estimate_params(scores)

        # Persist estimated hyperparameters as public attributes
        self.low_percentile_   = params['low_percentile']
        self.high_percentile_  = params['high_percentile']
        self.beta_             = params['beta']
        self.theta_            = params['theta']
        self.imbalance_proxy_  = params.get('imbalance_proxy_', float('nan'))

        # Also expose as instance attributes for run_generateSplit.py logging
        self.low_percentile    = self.low_percentile_
        self.high_percentile   = self.high_percentile_
        self.beta              = self.beta_
        self.theta             = self.theta_

        print(
            f"[Adaptive-IS] Estimated params — "
            f"low_percentile={self.low_percentile_:.1f}, "
            f"high_percentile={self.high_percentile_:.1f}, "
            f"beta={self.beta_:.3f}, "
            f"theta={self.theta_:.3f}, "
            f"imbalance_proxy={self.imbalance_proxy_:.3f}"
        )

        if 'component_means_' in params:
            print(f"[Adaptive-IS] GMM component means: {params['component_means_']}")

        print("[Adaptive-IS] Step 3 — applying selection …")
        self.mask, self.low_threshold_, self.high_threshold_ = self._apply_selection(
            scores=scores,
            y=y,
            low_percentile=self.low_percentile_,
            high_percentile=self.high_percentile_,
            beta=self.beta_,
            theta=self.theta_,
        )

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = np.where(self.mask)[0]
        self.reduction_ = 1.0 - float(len(self.y_)) / n_samples

        n_removed = n_samples - len(self.y_)
        print(
            f"[Adaptive-IS] Removed {n_removed} instances — "
            f"kept {len(self.y_)} / {n_samples} "
            f"(reduction = {self.reduction_:.2%})"
        )

        return self.X_, self.y_
