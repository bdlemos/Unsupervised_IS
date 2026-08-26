"""
Entropy-Adaptive Sublinear Autoencoder Instance Selection (EntropySublinearAEIS)
================================================================================

Extension of SublinearAEIS that automatically determines the optimal reduction
rate based on the normalized entropy of the micro-cluster size distribution.

Instead of requiring a fixed `target_reduction` parameter, the method computes:

    H = -sum(p_c * log(p_c)) / log(K)

where p_c = S_c / N is the fraction of instances in cluster c.

Then:
    target_reduction = r_max * (1 - H^alpha)

- High entropy (H ≈ 1.0) → clusters are uniform → dataset is balanced →
  conservative reduction (few redundant pockets).
- Low entropy (H ≪ 1.0) → few clusters dominate → heavy imbalance →
  aggressive reduction (large redundant pockets to exploit).

This makes the method **zero-parameter** for practical use.

Parameters
----------
r_max : float, default=0.50
    Maximum allowed reduction rate.
alpha : float, default=15.0
    Entropy sensitivity exponent. Higher = more conservative.
n_clusters : int, default=200
    Number of micro-clusters for spatial tessellation.
gamma : float, default=0.5
    The sublinear scaling factor (0.5 = square root).
ae_epochs : int, default=50
    Number of epochs to train the autoencoder.
random_state : int, default=0
"""

from __future__ import annotations

import numpy as np
import scipy
from sklearn.cluster import MiniBatchKMeans, Birch, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_X_y

from src.main.python.iSel.base import InstanceSelectionMixin
from src.main.python.iSel.autoencoder_is import AutoencoderIS


def _find_lambda(sizes: np.ndarray, gamma: float, target_keep_count: int) -> float:
    """Find the scaling factor lambda to match the target keep count."""
    low = 0.0
    high = float(target_keep_count)
    best_lam = high

    for _ in range(50):
        mid = (low + high) / 2.0
        kept = sum(min(s, int(np.ceil(mid * (s ** gamma)))) for s in sizes)

        if kept >= target_keep_count:
            best_lam = mid
            high = mid
        else:
            low = mid

    return best_lam


def _compute_normalized_entropy(sizes: np.ndarray) -> float:
    """Compute the normalized entropy of the cluster size distribution."""
    total = np.sum(sizes)
    if total == 0:
        return 1.0

    # Filter out empty clusters
    nonzero = sizes[sizes > 0]
    k = len(nonzero)

    if k <= 1:
        return 1.0

    probs = nonzero / total
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(k)

    return entropy / max_entropy  # Normalized to [0, 1]


class EntropySublinearAEIS(InstanceSelectionMixin):

    def __init__(
        self,
        r_max: float = 0.50,
        alpha: float = 15.0,
        n_clusters: int = 200,
        gamma: float = 0.5,
        ae_epochs: int = 50,
        random_state: int = 0,
        clustering_method: str = 'minibatchkmeans',
        **ae_kwargs
    ) -> None:
        self.r_max = r_max
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.ae_epochs = ae_epochs
        self.random_state = random_state
        self.clustering_method = clustering_method.lower()
        self.ae_kwargs = ae_kwargs
        self.sample_indices_ = []

    def _to_dense(self, X):
        if scipy.sparse.issparse(X):
            return X.toarray()
        return np.asarray(X, dtype=np.float64)

    def select_data(self, X, y):
        X = self._to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        # ------------------------------------------------------------------
        # Step 1: Autoencoder Scoring
        # ------------------------------------------------------------------
        print(f"[EntropySublinearAEIS] Step 1 — Training Autoencoder for {self.ae_epochs} epochs...")
        ae = AutoencoderIS(
            n_epochs=self.ae_epochs,
            low_percentile=0.0,
            high_percentile=100.0,
            beta=0.0,
            theta=0.0,
            random_state=self.random_state,
            **self.ae_kwargs
        )
        ae.fit(X, y)
        scores = ae.reconstruction_errors_
        self.scores_ = scores

        print(f"[EntropySublinearAEIS] AE scores: min={scores.min():.4f}, "
              f"median={np.median(scores):.4f}, max={scores.max():.4f}")

        # ------------------------------------------------------------------
        # Step 2: Micro-clustering (Tessellation)
        # ------------------------------------------------------------------
        # n_clusters = max(2, min(self.n_clusters, n_samples // 10))
        n_clusters = np.sqrt(n_samples).astype(int)  # Use sqrt(N) for micro-clusters
        print(f"[EntropySublinearAEIS] Step 2 — Tessellating into {n_clusters} micro-clusters using {self.clustering_method}...")

        if self.clustering_method == 'minibatchkmeans':
            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=3)
            labels = km.fit_predict(X)
        elif self.clustering_method == 'gmm':
            gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
            labels = gmm.fit_predict(X)
        elif self.clustering_method == 'birch':
            birch = Birch(n_clusters=n_clusters)
            labels = birch.fit_predict(X)
        elif self.clustering_method == 'dbscan':
            # DBSCAN doesn't take n_clusters, we use eps and min_samples.
            # Using reasonable defaults for normalized dense embeddings (like jina-v5).
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(X)
            # DBSCAN assigns -1 to noise. We can treat noise as a separate cluster
            # or handle it gracefully.
            # We'll shift labels by 1 so noise becomes cluster 0 and other clusters are 1, 2, ...
            labels = labels + 1
            n_clusters = len(np.unique(labels))
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        sizes = np.bincount(labels, minlength=n_clusters)
        self.labels_ = labels
        self.cluster_sizes_ = sizes

        # ------------------------------------------------------------------
        # Step 3: Entropy-Based Adaptive Reduction
        # ------------------------------------------------------------------
        # K-Means usually produces high entropy (0.9 to 1.0).
        # Using a higher alpha (e.g. 15) makes the function sensitive in this narrow band.
        H = _compute_normalized_entropy(sizes)
        target_reduction = self.r_max * (1.0 - H ** self.alpha)
        target_reduction = max(0.05, min(target_reduction, self.r_max))  # Clamp to [5%, r_max]

        self.entropy_ = H
        self.target_reduction = target_reduction

        target_keep_count = int(n_samples * (1.0 - target_reduction))

        print(f"[EntropySublinearAEIS] Step 3 — Entropy H={H:.4f}, "
              f"adaptive reduction={target_reduction:.2%} "
              f"(Keeping {target_keep_count} / {n_samples})")

        # ------------------------------------------------------------------
        # Step 4: Sublinear Target Distribution
        # ------------------------------------------------------------------
        lam = _find_lambda(sizes, self.gamma, target_keep_count)
        print(f"[EntropySublinearAEIS] Step 4 — Sublinear sampling (gamma={self.gamma}). lambda = {lam:.4f}")

        # ------------------------------------------------------------------
        # Step 5: Probabilistic Redundancy Removal via AE Scores
        # ------------------------------------------------------------------
        np.random.seed(self.random_state)
        mask = np.ones(n_samples, dtype=bool)

        for c in range(n_clusters):
            cluster_idx = np.where(labels == c)[0]
            s_c = len(cluster_idx)

            if s_c == 0:
                continue

            keep_count = min(s_c, int(np.ceil(lam * (s_c ** self.gamma))))
            remove_count = s_c - keep_count

            if remove_count > 0:
                c_scores = scores[cluster_idx]

                # Inverse scores: low error → high removal probability
                inv_scores = 1.0 / (c_scores + 1e-8)
                p_remove = inv_scores / np.sum(inv_scores)

                to_remove_local = np.random.choice(
                    len(cluster_idx),
                    size=remove_count,
                    replace=False,
                    p=p_remove
                )

                to_remove = cluster_idx[to_remove_local]
                mask[to_remove] = False

        # ------------------------------------------------------------------
        # Build outputs
        # ------------------------------------------------------------------
        self.mask = mask
        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = np.where(self.mask)[0]
        self.reduction_ = 1.0 - float(len(self.y_)) / n_samples

        # Compatibility attributes
        self.beta = self.target_reduction
        self.theta = 0.0
        self.low_percentile = 0.0
        self.high_percentile = 100.0

        print(f"[EntropySublinearAEIS] Done — removed {n_samples - len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_
