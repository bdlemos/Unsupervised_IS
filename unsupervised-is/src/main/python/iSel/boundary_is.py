"""
Boundary-Preserving Sublinear Instance Selection (BoundaryIS)
=============================================================

An unsupervised instance selection method designed to protect minority classes
and preserve class boundaries by using sublinear spatial sampling.

Core Idea
---------
Standard unsupervised methods fail when minority classes are dense (like in SST1)
because they mistakenly prune them. To fix this, we need a method that corrects
imbalance based on spatial population without assuming minority = sparse.

1. **Micro-clustering (Tessellation)**: We divide the feature space into a large
   number of micro-clusters (e.g., K=100) using K-Means. This creates a Voronoi
   tessellation of the space.
2. **Sublinear Sampling**: Dense regions (typically majority classes) will form
   micro-clusters with many instances. We determine the number of instances to
   keep from each cluster $c$ using a sublinear function of its size $S_c$:
       keep_count = lambda * (S_c ^ gamma)
   Where `gamma` (e.g., 0.5) strongly flattens the distribution, penalizing
   overpopulated regions while leaving sparsely populated micro-clusters nearly
   intact. `lambda` is solved to match the exact global target reduction.
3. **Boundary Preservation**: Within each micro-cluster, we must choose *which*
   instances to remove. We remove the instances *closest to the centroid*
   (the highly redundant core of the micro-cluster) and explicitly KEEP the
   instances *furthest from the centroid* (the boundaries, which often contain
   informative support vectors and minority overlaps).

Parameters
----------
target_reduction : float, default=0.30
    The overall fraction of the dataset to remove (e.g., 0.30 = 30%).
n_clusters : int, default=100
    Number of micro-clusters for spatial tessellation.
gamma : float, default=0.5
    The sublinear scaling factor. 
    1.0 = Proportional (preserves imbalance).
    0.5 = Square-root (corrects imbalance smoothly).
    0.0 = Uniform (flattens completely, every cluster gets same amount).
random_state : int, default=0

Attributes
----------
mask, X_, y_, sample_indices_, reduction_
labels_, cluster_sizes_
"""

from __future__ import annotations

import numpy as np
import scipy
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.validation import check_X_y

from src.main.python.iSel.base import InstanceSelectionMixin


def _find_lambda(sizes: np.ndarray, gamma: float, target_keep_count: int) -> float:
    """Find the scaling factor lambda such that sum(min(S_c, lambda * S_c^gamma)) 
    equals the target_keep_count.
    """
    # If gamma is 0, we just want uniform cap per cluster
    def get_kept(lam):
        return sum(min(s, int(np.ceil(lam * (s ** gamma)))) for s in sizes)

    # Binary search for lambda
    low = 0.0
    high = float(target_keep_count)  # Max possible lambda is keeping everything
    best_lam = high

    for _ in range(50):  # 50 iterations is more than enough for precision
        mid = (low + high) / 2.0
        kept = get_kept(mid)
        
        if kept >= target_keep_count:
            best_lam = mid
            high = mid
        else:
            low = mid

    return best_lam


class BoundaryIS(InstanceSelectionMixin):
    
    def __init__(
        self,
        target_reduction: float = 0.30,
        n_clusters: int = 100,
        gamma: float = 0.5,
        random_state: int = 0,
    ) -> None:
        self.target_reduction = target_reduction
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
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
        print(f"[BoundaryIS] Target reduction: {self.target_reduction:.2%} "
              f"(Keeping {target_keep_count} / {n_samples} instances)")

        # ------------------------------------------------------------------
        # Step 1: Micro-clustering (Tessellation)
        # ------------------------------------------------------------------
        n_clusters = max(2, min(self.n_clusters, n_samples // 10))
        print(f"[BoundaryIS] Step 1 — Tessellating space into {n_clusters} micro-clusters …")
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=3)
        labels = km.fit_predict(X)
        
        sizes = np.bincount(labels, minlength=n_clusters)
        self.labels_ = labels
        self.cluster_sizes_ = sizes
        
        # Distance of each instance to its own centroid
        centroid_dist = km.transform(X)[np.arange(len(labels)), labels]

        print(f"[BoundaryIS] Micro-cluster sizes: min={sizes.min()}, median={np.median(sizes):.1f}, max={sizes.max()}")

        # ------------------------------------------------------------------
        # Step 2: Sublinear target distribution
        # ------------------------------------------------------------------
        lam = _find_lambda(sizes, self.gamma, target_keep_count)
        print(f"[BoundaryIS] Step 2 — Sublinear sampling (gamma={self.gamma}). Found lambda = {lam:.4f}")

        # ------------------------------------------------------------------
        # Step 3: Boundary preservation (remove from the core)
        # ------------------------------------------------------------------
        mask = np.ones(n_samples, dtype=bool)
        
        for c in range(n_clusters):
            cluster_idx = np.where(labels == c)[0]
            s_c = len(cluster_idx)
            
            if s_c == 0:
                continue
                
            keep_count = min(s_c, int(np.ceil(lam * (s_c ** self.gamma))))
            remove_count = s_c - keep_count
            
            if remove_count > 0:
                # Sort by distance to centroid (ascending: closest first)
                dists = centroid_dist[cluster_idx]
                closest_indices = cluster_idx[np.argsort(dists)]
                
                # Remove the most redundant (closest to centroid, core instances)
                # This PRESERVES the instances furthest from the centroid (boundaries)
                to_remove = closest_indices[:remove_count]
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

        print(f"[BoundaryIS] Done — removed {n_samples - len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_
