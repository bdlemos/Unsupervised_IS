"""
Cluster-Random Instance Selection (ClusterRandomIS)

Partitions the feature space into K micro-clusters and applies a sublinear
retention quota (K_c = lambda * S_c^gamma). Within each micro-cluster, instances
are sampled uniformly at random.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_X_y

from src.main.python.iSel.base import InstanceSelectionMixin


def _find_lambda(sizes: np.ndarray, gamma: float, target_keep: int) -> float:
    low = 0.0
    high = float(target_keep)
    best_lam = high
    for _ in range(50):
        mid = (low + high) / 2.0
        kept = sum(min(s, int(np.ceil(mid * (s ** gamma)))) for s in sizes)
        if kept >= target_keep:
            best_lam = mid
            high = mid
        else:
            low = mid
    return best_lam


def _compute_quotas(sizes: np.ndarray, gamma: float, target_keep: int) -> np.ndarray:
    if target_keep <= 0:
        return np.zeros_like(sizes, dtype=int)
    lam = _find_lambda(sizes.astype(float), gamma, target_keep)
    q = np.minimum(sizes, np.ceil(lam * (sizes ** gamma))).astype(int)
    diff = int(q.sum() - target_keep)
    if diff > 0:
        order = np.argsort(-q)
        for c in order:
            if diff == 0:
                break
            if q[c] > 1:
                take = min(diff, q[c] - 1)
                q[c] -= take
                diff -= take
        if diff > 0:
            for c in order:
                if diff == 0:
                    break
                if q[c] > 0:
                    q[c] -= 1
                    diff -= 1
    elif diff < 0:
        room = sizes - q
        order = np.argsort(-room)
        for c in order:
            if diff == 0:
                break
            if room[c] > 0:
                q[c] += 1
                diff += 1
    return q


class ClusterRandomIS(InstanceSelectionMixin):
    """Cluster-Random Instance Selection with sublinear cluster quotas."""

    def __init__(
        self,
        target_reduction: float = 0.50,
        gamma: float = 0.5,
        n_clusters: int = -1,
        random_state: int = 42,
    ) -> None:
        self.target_reduction = float(target_reduction)
        self.gamma = float(gamma)
        self.n_clusters = int(n_clusters)
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

        # Prepare features for clustering
        if sp.issparse(X):
            if X.shape[1] <= 4096:
                X_clust = normalize(np.asarray(X.toarray(), dtype=np.float32), norm="l2", copy=False)
            else:
                X_clust = X.copy()
                X_clust.indices = X_clust.indices.astype(np.int32)
                X_clust.indptr = X_clust.indptr.astype(np.int32)
                X_clust = normalize(X_clust, norm="l2", copy=False)
        else:
            X_clust = normalize(np.asarray(X, dtype=np.float32), norm="l2", copy=False)

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
        labels = km.fit_predict(X_clust)
        sizes = np.bincount(labels, minlength=K)

        quotas = _compute_quotas(sizes, self.gamma, target_keep)
        rng = np.random.RandomState(self.random_state)

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
                sel = rng.choice(c_idx, size=q_c, replace=False)
                keep_indices.extend(sel)

        keep_indices = list(sorted(keep_indices))
        self.sample_indices_ = keep_indices
        self.labels_ = labels
        self.cluster_sizes_ = sizes
        self.cluster_quotas_ = quotas
        self.reduction_ = 1.0 - float(len(keep_indices)) / n_samples
        self.X_ = X[keep_indices]
        self.y_ = y[keep_indices]

        return self.X_, self.y_
