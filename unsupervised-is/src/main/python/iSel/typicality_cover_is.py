"""
Typicality-Cover Instance Selection (TypicalityCoverIS)

Fuses Spatial Sublinear Tessellation with Density Typicality Scoring.
1. Computes typicality (inverse mean cosine distance to k-nearest neighbors).
2. Tessellates the feature space into K micro-clusters with sublinear quota (K_c = lambda * S_c^gamma).
3. Selects the most typical (dense centroid-like) instances within each micro-cluster up to quota.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_X_y

from src.main.python.iSel.base import InstanceSelectionMixin
from src.main.python.iSel.cluster_random_is import _compute_quotas


def compute_typicality(X, k: int = 20, n_jobs: int = 4) -> np.ndarray:
    """Computes density typicality as inverse mean cosine distance to k-NN."""
    n = X.shape[0]
    k_eff = min(k, n - 1)
    if k_eff <= 0:
        return np.ones(n, dtype=np.float64)

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", algorithm="brute", n_jobs=n_jobs)
    nn.fit(X)
    dist, _ = nn.kneighbors(X)
    # Exclude self distance (index 0)
    mean_dist = dist[:, 1:].mean(axis=1)
    return 1.0 / (mean_dist + 1e-8)


class TypicalityCoverIS(InstanceSelectionMixin):
    """Typicality-Cover Instance Selection."""

    def __init__(
        self,
        target_reduction: float = 0.50,
        gamma: float = 0.5,
        n_clusters: int = -1,
        k_neighbors: int = 20,
        random_state: int = 42,
    ) -> None:
        self.target_reduction = float(target_reduction)
        self.gamma = float(gamma)
        self.n_clusters = int(n_clusters)
        self.k_neighbors = int(k_neighbors)
        self.random_state = random_state
        self.sample_indices_ = []

    def select_data(self, X, y):
        X, y = check_X_y(X, y, accept_sparse="csr")
        n_samples = len(y)
        target_keep = max(int(round(n_samples * (1.0 - self.target_reduction))), 2)

        if target_keep >= n_samples:
            self.sample_indices_ = list(range(n_samples))
            self.reduction_ = 0.0
            self.X_ = X
            self.y_ = y
            return self.X_, self.y_

        # Prepare normalized features for distance and clustering
        if sp.issparse(X):
            if X.shape[1] <= 4096:
                X_norm = normalize(np.asarray(X.toarray(), dtype=np.float32), norm="l2", copy=False)
            else:
                X_norm = X.copy()
                X_norm.indices = X_norm.indices.astype(np.int32)
                X_norm.indptr = X_norm.indptr.astype(np.int32)
                X_norm = normalize(X_norm, norm="l2", copy=False)
        else:
            X_norm = normalize(np.asarray(X, dtype=np.float32), norm="l2", copy=False)

        # 1. Typicality Density Scoring
        typicality = compute_typicality(X_norm, k=self.k_neighbors)
        self.scores_ = typicality

        # 2. Spatial Tessellation into Micro-clusters
        K = self.n_clusters
        if K <= 0:
            K = int(min(500, max(2, n_samples // 2)))
        K = min(K, n_samples)

        km = MiniBatchKMeans(
            n_clusters=K,
            random_state=self.random_state,
            n_init=3,
            batch_size=min(4096, n_samples),
        )
        labels = km.fit_predict(X_norm)
        sizes = np.bincount(labels, minlength=K)

        quotas = _compute_quotas(sizes, self.gamma, target_keep)

        # 3. Quota-based Typicality Selection
        keep_indices = []
        for c in range(K):
            c_idx = np.where(labels == c)[0]
            s_c = len(c_idx)
            q_c = quotas[c]
            if s_c == 0 or q_c == 0:
                continue
            if q_c >= s_c:
                keep_indices.extend(c_idx)
            else:
                top_c = c_idx[np.argsort(-typicality[c_idx])[:q_c]]
                keep_indices.extend(top_c)

        keep_indices = list(sorted(keep_indices))
        self.sample_indices_ = keep_indices
        self.labels_ = labels
        self.cluster_sizes_ = sizes
        self.cluster_quotas_ = quotas
        self.reduction_ = 1.0 - float(len(keep_indices)) / n_samples
        self.X_ = X[keep_indices]
        self.y_ = y[keep_indices]

        return self.X_, self.y_
