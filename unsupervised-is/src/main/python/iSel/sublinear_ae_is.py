"""
Sublinear Autoencoder Instance Selection (SublinearAEIS)
Sublinear Spatial Autoencoding for Instance Selection
========================================================
"Beyond Sparsity: Sublinear Spatial Autoencoding for Unsupervised Instance 
Selection in Imbalanced Text Classification" ou 
"Challenging the Sparsity Assumption: A Sublinear Boundary-Preserving Framework 
for Unsupervised Instance Selection"


An advanced unsupervised instance selection method that fuses Spatial Sublinear
Sampling with Deep Autoencoder Reconstruction Scoring to achieve extreme 
robustness against both imbalance and overlapping distributions.

Core Idea
---------
1. **Autoencoder Scoring (The "What to Remove")**:
   We train an Autoencoder to reconstruct the feature space. Instances that are 
   highly redundant and typical of the majority classes will have a very LOW 
   reconstruction error. Rare, minority, or boundary instances will have HIGH 
   reconstruction errors.

2. **Micro-clustering & Sublinear Sampling (The "How Much to Remove")**:
   We tessellate the space into K micro-clusters. We cap the retention of each 
   cluster using a sublinear function of its size: $K_c = \\lambda \\cdot S_c^\\gamma$.
   This strongly flattens the distribution, forcing large dense clusters (majority)
   to lose many instances while protecting sparse/small clusters.

3. **Synergy (The "Selection")**:
   Within each micro-cluster, instead of blindly removing instances close to the 
   centroid (which fails if an ultra-minority instance happens to be in the center), 
   we sort instances by their Autoencoder Reconstruction Error. We remove the 
   instances with the **lowest** errors (the most redundant) and keep those with 
   the **highest** errors (the rarest/most boundary-like) - probabilistically 
   based on the inverse of the reconstruction error.

This ensures that even if an extreme long-tail class (e.g., 2 instances) falls 
into a massive majority micro-cluster, its Autoencoder error will be extremely 
high, protecting it from deletion.

Parameters
----------
target_reduction : float, default=0.30
    The overall fraction of the dataset to remove.
n_clusters : int, default=100
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


class SublinearAEIS(InstanceSelectionMixin):
    
    def __init__(
        self,
        target_reduction: float = 0.30,
        n_clusters: int = 100,
        gamma: float = 0.5,
        ae_epochs: int = 50,
        random_state: int = 0,
        **ae_kwargs
    ) -> None:
        self.target_reduction = target_reduction
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.ae_epochs = ae_epochs
        self.random_state = random_state
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
        print(f"[SublinearAEIS] Target reduction: {self.target_reduction:.2%} "
              f"(Keeping {target_keep_count} / {n_samples} instances)")

        # ------------------------------------------------------------------
        # Step 1: Autoencoder Scoring
        # ------------------------------------------------------------------
        print(f"[SublinearAEIS] Step 1 — Training Autoencoder for {self.ae_epochs} epochs...")
        ae = AutoencoderIS(
            n_epochs=self.ae_epochs,
            low_percentile=0.0,   # We won't use the built-in removal
            high_percentile=100.0,
            beta=0.0,
            theta=0.0,
            random_state=self.random_state,
            **self.ae_kwargs
        )
        # Train AE and extract scores
        ae.fit(X, y)
        scores = ae.reconstruction_errors_
        self.scores_ = scores

        print(f"[SublinearAEIS] Autoencoder scores: min={scores.min():.4f}, "
              f"median={np.median(scores):.4f}, max={scores.max():.4f}")

        # ------------------------------------------------------------------
        # Step 2: Micro-clustering (Tessellation)
        # ------------------------------------------------------------------
        n_clusters = max(2, min(self.n_clusters, n_samples // 10))
        print(f"[SublinearAEIS] Step 2 — Tessellating space into {n_clusters} micro-clusters...")
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=3)
        labels = km.fit_predict(X)
        
        sizes = np.bincount(labels, minlength=n_clusters)
        self.labels_ = labels
        self.cluster_sizes_ = sizes

        # ------------------------------------------------------------------
        # Step 3: Sublinear Target Distribution
        # ------------------------------------------------------------------
        lam = _find_lambda(sizes, self.gamma, target_keep_count)
        print(f"[SublinearAEIS] Step 3 — Sublinear sampling (gamma={self.gamma}). lambda = {lam:.4f}")

        # ------------------------------------------------------------------
        # Step 4: Redundancy removal via AE Scores
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
                c_scores = scores[cluster_idx]
                
                # Invert the scores so that LOW errors have HIGH probabilities of removal
                # We add a small epsilon to avoid division by zero
                inv_scores = 1.0 / (c_scores + 1e-8)
                
                # Normalize to create a probability distribution
                p_remove = inv_scores / np.sum(inv_scores)
                
                # Sample the instances to remove probabilistically without replacement
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

        print(f"[SublinearAEIS] Done — removed {n_samples - len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_
