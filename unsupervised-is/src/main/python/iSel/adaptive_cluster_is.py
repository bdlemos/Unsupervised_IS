from __future__ import annotations

from typing import Literal

import numpy as np
import scipy
from sklearn.utils.validation import check_X_y
from src.main.python.iSel import autoencoder_is, gmm_is, perplexity_is
from src.main.python.iSel.base import InstanceSelectionMixin


def _cluster_removal_rates(
    X: np.ndarray,
    n_clusters: int = 30,
    rate_min: float = 0.02,
    rate_max: float = 0.75,
    random_state: int = 0,
) -> tuple:
    """Derive a removal rate for each cluster from its relative size.

    Larger cluster → denser region → more redundancy → higher removal rate.

        rate(c) = rate_min + (rate_max - rate_min) * sigmoid(log(size_c / median_size))

    sigmoid(0) = 0.5, so a median-size cluster gets (rate_min + rate_max) / 2 ≈ 0.38.
    """
    from sklearn.cluster import MiniBatchKMeans

    # guard: need at least 5 instances per cluster on average
    n_clusters = max(2, min(n_clusters, len(X) // 5))

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
    labels = km.fit_predict(X)

    sizes = np.bincount(labels, minlength=n_clusters).astype(float)
    median_size = float(np.median(sizes[sizes > 0]))

    log_ratios = np.log(np.maximum(sizes, 1) / median_size)
    removal_rates = rate_min + (rate_max - rate_min) / (1.0 + np.exp(-log_ratios))

    # distance of each instance to its own cluster centroid
    # km.transform returns (n_samples, n_clusters); we pick each instance's own cluster column
    centroid_dist = km.transform(X)[np.arange(len(labels)), labels]

    return labels, removal_rates, sizes, median_size, centroid_dist


def _apply_selection_clusters(
    scores: np.ndarray,
    labels: np.ndarray,
    removal_rates: np.ndarray,
    centroid_dist: np.ndarray,
    sparse_threshold: float = 0.85,
    random_state: int = 0,
) -> np.ndarray:
    """Within each cluster, sample instances for removal with probability inversely
    proportional to their score — lower score = more redundant = higher removal probability.

    Instances in the top (1 - sparse_threshold) fraction by distance to their cluster
    centroid are excluded from removal: they are atypical within the cluster and likely
    belong to a minority class that was merged into a majority cluster.
    """
    mask = np.ones(len(scores), dtype=bool)
    rng = np.random.RandomState(random_state)

    for c, rate in enumerate(removal_rates):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        # protect the most centroid-distant instances — they don't truly belong here
        cutoff = np.quantile(centroid_dist[idx], sparse_threshold)
        eligible = idx[centroid_dist[idx] <= cutoff]

        n_remove = min(int(len(idx) * rate), len(eligible))
        if n_remove == 0:
            continue

        cluster_scores = scores[eligible]
        # weight ∝ (max - score): lowest score gets the highest removal probability
        weights = cluster_scores.max() - cluster_scores + 1e-9
        weights /= weights.sum()

        to_remove = rng.choice(eligible, size=n_remove, replace=False, p=weights)
        mask[to_remove] = False

    return mask


class AdaptiveClusterIS(InstanceSelectionMixin):
    """Cluster-size proportional unsupervised instance selection.

    Replaces the fixed Q1–Q4 quartile zones with mini-batch k-means clusters.
    Each cluster's removal rate is derived from its relative size, so dense
    (large) clusters are pruned aggressively and rare (small) clusters are
    mostly preserved — without any label information.

    Parameters
    ----------
    base_method : {'autoencoder', 'gmm', 'perplexity'}
    n_clusters : int
        Target number of clusters. Clamped to max(2, n_samples // 5).
    rate_min, rate_max : float
        Bounds on the per-cluster removal rate. A median-size cluster gets
        roughly (rate_min + rate_max) / 2.
    sparse_threshold : float
        Within each cluster, instances beyond this percentile of centroid distance
        are protected from removal. They are atypical within the cluster — likely
        minority-class instances absorbed into a majority cluster.
    local_normalization : bool
        Divide each score by the median score of its 20 nearest neighbors
        before clustering, removing bias against minority-class instances.
    random_state : int
    **base_kwargs
        Forwarded to the base scorer constructor.

    Attributes
    ----------
    labels_ : ndarray of int, shape (n_samples,)
        Cluster assignment for every instance.
    removal_rates_ : ndarray, shape (n_clusters,)
        Per-cluster removal rate.
    cluster_sizes_ : ndarray, shape (n_clusters,)
    scores_ : ndarray, shape (n_samples,)
        Locally-normalized (or raw) scores used for ranking within clusters.
    mask : ndarray of bool
    sample_indices_, reduction_ : ndarray, float
    """

    def __init__(
        self,
        base_method: Literal["autoencoder", "gmm", "perplexity"] = "autoencoder",
        n_clusters: int = 30,
        rate_min: float = 0.02,
        rate_max: float = 0.75,
        sparse_threshold: float = 0.85,
        local_normalization: bool = True,
        random_state: int = 0,
        **base_kwargs,
    ) -> None:
        self.base_method         = base_method
        self.n_clusters          = n_clusters
        self.rate_min            = rate_min
        self.rate_max            = rate_max
        self.sparse_threshold    = sparse_threshold
        self.local_normalization = local_normalization
        self.random_state        = random_state
        self.base_kwargs         = base_kwargs
        self.sample_indices_     = []

    def _build_base_scorer(self):
        # neutral thresholds so the base scorer computes scores without removing anything
        kwargs = dict(self.base_kwargs)
        kwargs.update(low_percentile=0.0, high_percentile=100.0, beta=0.0, theta=0.0)

        if self.base_method == "autoencoder":
            kwargs.setdefault("n_epochs", 50)
            kwargs.setdefault("batch_size", 512)
            kwargs.setdefault("bottleneck_ratio", 0.25)
            return autoencoder_is.AutoencoderIS(random_state=self.random_state, **kwargs)

        elif self.base_method == "gmm":
            return gmm_is.GMMIS(random_state=self.random_state, **kwargs)

        elif self.base_method == "perplexity":
            kwargs.setdefault("n_topics", 10)
            return perplexity_is.PerplexityIS(random_state=self.random_state, **kwargs)

        raise ValueError(f"Unknown base_method='{self.base_method}'")

    def _extract_scores(self, scorer) -> np.ndarray:
        if self.base_method == "autoencoder":
            return scorer.reconstruction_errors_
        elif self.base_method == "gmm":
            return scorer.gmm_scores_
        elif self.base_method == "perplexity":
            return scorer.perplexities_
        raise ValueError(f"Cannot extract scores for base_method='{self.base_method}'")
    
    def _to_dense(self, X):
        """Convert sparse matrix to dense if necessary."""
        if scipy.sparse.issparse(X):
            return X.toarray()
        return np.asarray(X, dtype=np.float64)

    def select_data(self, X, y):
        """Run the full AdaptiveClusterIS pipeline. Labels are not used for selection."""
        X = self._to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        # Step 1 — scoring
        print(f"[AdaptiveClusterIS] Step 1 — scoring via '{self.base_method}' …")
        scorer = self._build_base_scorer()
        scorer.fit(X, y)
        raw_scores        = self._extract_scores(scorer)
        self.raw_scores_  = raw_scores.copy()
        self.base_scorer_ = scorer

        scores = raw_scores.copy()

        if self.local_normalization:
            print("[AdaptiveClusterIS] Step 1.5 — Local Normalization (KNN) …")
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=21, metric="cosine", n_jobs=-1)
            nn.fit(X)
            _, indices = nn.kneighbors(X)

            neighbor_scores     = raw_scores[indices[:, 1:]]
            local_medians       = np.median(neighbor_scores, axis=1)
            scores              = raw_scores / (local_medians + 1e-8)
            self.local_medians_ = local_medians

        self.scores_ = scores.copy()

        # Step 2 — cluster and estimate per-cluster removal rates
        print(f"[AdaptiveClusterIS] Step 2 — clustering into {self.n_clusters} groups …")
        labels, removal_rates, sizes, median_size, centroid_dist = _cluster_removal_rates(
            X=X,
            n_clusters=self.n_clusters,
            rate_min=self.rate_min,
            rate_max=self.rate_max,
            random_state=self.random_state,
        )
        self.labels_              = labels
        self.removal_rates_       = removal_rates
        self.cluster_sizes_       = sizes
        self.median_cluster_size_ = median_size
        self.centroid_dist_       = centroid_dist

        print(
            f"[AdaptiveClusterIS] Cluster sizes: "
            f"min={int(sizes.min())}, median={int(median_size)}, max={int(sizes.max())}",
        )
        print(
            f"[AdaptiveClusterIS] Removal rates: "
            f"min={removal_rates.min():.3f}, "
            f"median={np.median(removal_rates):.3f}, "
            f"max={removal_rates.max():.3f}",
        )

        # Step 3 — apply per-cluster selection
        print("[AdaptiveClusterIS] Step 3 — applying cluster selection …")
        self.mask = _apply_selection_clusters(
            scores=scores,
            labels=labels,
            removal_rates=removal_rates,
            centroid_dist=centroid_dist,
            sparse_threshold=self.sparse_threshold,
            random_state=self.random_state,
        )

        self.X_              = np.asarray(X[self.mask])
        self.y_              = np.asarray(y[self.mask])
        self.sample_indices_ = np.where(self.mask)[0]
        self.reduction_      = 1.0 - float(len(self.y_)) / n_samples

        # compat attrs expected by run_generateSplit.py
        self.beta            = float(np.mean(removal_rates))
        self.theta           = 0.0
        self.low_percentile  = 0.0
        self.high_percentile = 100.0

        n_removed = n_samples - len(self.y_)
        print(
            f"[AdaptiveClusterIS] Done — removed {n_removed} / {n_samples} "
            f"(reduction = {self.reduction_:.2%})",
        )

        return self.X_, self.y_
