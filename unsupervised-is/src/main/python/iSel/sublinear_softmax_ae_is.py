"""
Sublinear Softmax Autoencoder Instance Selection (SublinearSoftmaxAEIS)
========================================================================
An advanced unsupervised instance selection method that fuses Spatial Sublinear
Micro-Clustering with Deep Autoencoder Reconstruction Scoring and Intra-Cluster
Z-Score Softmax Sampling with Temperature tau.

Key Methodological Principles:
1. Spatial Sublinear Tessellation:
   Divides the feature space into K micro-clusters (default K = sqrt(N)).
   Applies a sublinear retention quota K_c = ceil(lambda * S_c^gamma) with gamma=0.5,
   flattening cluster size disparity and protecting minority/rare regions.

2. Autoencoder Reconstruction Scoring:
   Measures representation typicality via Autoencoder reconstruction error (RE).

3. Intra-Cluster Z-Score Normalization:
   Normalizes reconstruction error within each micro-cluster:
   z = (score - mean) / std.
   Eliminates intra-cluster scale collapse and decouples decisions from global scale.

4. Softmax Sampling with Temperature (tau):
   Computes removal probabilities via softmax on scaled z-scores:
   - "cleaning" mode (default): logits = + z / tau
     Higher reconstruction error -> higher removal probability.
     Retains dense, low-error prototypes while pruning noisy/outlier points.
   - "original" mode: logits = - z / tau
     Lower reconstruction error -> higher removal probability.
     Retains rare/boundary points while pruning central redundancies.

5. Decoupled and Deterministic RNG:
   Uses an isolated local RandomState seeded by random_state, preventing
   spurious seed coupling across variants.
"""

from __future__ import annotations

import numpy as np
import scipy
from sklearn.cluster import MiniBatchKMeans
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


class SublinearSoftmaxAEIS(InstanceSelectionMixin):

    def __init__(
        self,
        target_reduction: float = 0.30,
        n_clusters: int = -1,
        gamma: float = 0.5,
        ae_epochs: int = 50,
        random_state: int = 13,
        mode: str = "cleaning",
        tau: float = 1.0,
        **ae_kwargs
    ) -> None:
        self.target_reduction = target_reduction
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.ae_epochs = ae_epochs
        self.random_state = random_state
        self.mode = mode
        self.tau = tau
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

        target_keep_count = int(n_samples * (1.0 - self.target_reduction))
        print(f"[SublinearSoftmaxAEIS] Target reduction: {self.target_reduction:.2%} "
              f"(Keeping {target_keep_count} / {n_samples} instances, mode='{self.mode}', tau={self.tau})")

        # ------------------------------------------------------------------
        # Step 1: Autoencoder Scoring
        # ------------------------------------------------------------------
        print(f"[SublinearSoftmaxAEIS] Step 1 — Training Autoencoder for {self.ae_epochs} epochs...")
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

        print(f"[SublinearSoftmaxAEIS] Autoencoder scores: min={scores.min():.4f}, "
              f"median={np.median(scores):.4f}, max={scores.max():.4f}")

        # ------------------------------------------------------------------
        # Step 2: Micro-clustering (Tessellation)
        # ------------------------------------------------------------------
        if self.n_clusters is not None and self.n_clusters > 0 and self.n_clusters < 99999:
            n_clusters = max(2, min(self.n_clusters, n_samples // 10))
        else:
            n_clusters = int(np.sqrt(n_samples))

        print(f"[SublinearSoftmaxAEIS] Step 2 — Tessellating space into {n_clusters} micro-clusters...")
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=3)
        labels = km.fit_predict(X)

        sizes = np.bincount(labels, minlength=n_clusters)
        self.labels_ = labels
        self.cluster_sizes_ = sizes

        # ------------------------------------------------------------------
        # Step 3: Sublinear Target Distribution
        # ------------------------------------------------------------------
        lam = _find_lambda(sizes, self.gamma, target_keep_count)
        print(f"[SublinearSoftmaxAEIS] Step 3 — Sublinear sampling (gamma={self.gamma}). lambda = {lam:.4f}")

        # ------------------------------------------------------------------
        # Step 4: Redundancy removal via Intra-Cluster Z-Score + Softmax
        # ------------------------------------------------------------------
        mask = np.ones(n_samples, dtype=bool)
        cluster_quotas = np.array([min(s, int(np.ceil(lam * (s ** self.gamma)))) for s in sizes])
        self.cluster_quotas_ = cluster_quotas

        rng = np.random.RandomState(self.random_state)

        for c in range(n_clusters):
            cluster_idx = np.where(labels == c)[0]
            s_c = len(cluster_idx)

            if s_c <= 1:
                continue

            keep_count = cluster_quotas[c]
            remove_count = s_c - keep_count

            if remove_count > 0:
                c_scores = scores[cluster_idx]
                mu = np.mean(c_scores)
                sigma = np.std(c_scores)

                if sigma > 1e-8:
                    z = (c_scores - mu) / sigma
                else:
                    z = np.zeros_like(c_scores)

                if self.mode == "original":
                    # Lower RE -> more redundant -> higher removal probability
                    logits = - z / self.tau
                else:
                    # 'cleaning' (default): Higher RE -> noise/outlier -> higher removal probability
                    logits = + z / self.tau

                # Numerically stable softmax
                logits -= np.max(logits)
                weights = np.exp(logits)
                p_remove = weights / np.sum(weights)

                to_remove_local = rng.choice(
                    s_c,
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

        print(f"[SublinearSoftmaxAEIS] Done — removed {n_samples - len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_
