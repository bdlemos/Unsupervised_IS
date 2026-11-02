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
mode : {"original", "cleaning"}, default="original"
    "original" removes low-error instances first (as described above);
    "cleaning" removes high-error instances first.
beta : float, default=1.0
    Exponent applied to the removal weights.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.utils.validation import check_X_y

from src.iSel import esae_core
from src.iSel.base import InstanceSelectionMixin
from src.iSel.autoencoder_is import AutoencoderIS


class SublinearAEIS(InstanceSelectionMixin):

    def __init__(
        self,
        target_reduction: float = 0.30,
        n_clusters: int = 100,
        gamma: float = 0.5,
        ae_epochs: int = 50,
        random_state: int = 0,
        mode: str = "original",
        beta: float = 1.0,
        **ae_kwargs
    ) -> None:
        self.target_reduction = target_reduction
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.ae_epochs = ae_epochs
        self.random_state = random_state
        self.mode = mode
        self.beta = beta
        self.ae_kwargs = ae_kwargs
        self.sample_indices_ = []

    def _reconstruction_errors(self, X, y) -> np.ndarray:
        ae = AutoencoderIS(
            n_epochs=self.ae_epochs,
            low_percentile=0.0,   # We won't use the built-in removal
            high_percentile=100.0,
            beta=0.0,
            theta=0.0,
            random_state=self.random_state,
            **self.ae_kwargs
        )
        ae.fit(X, y)
        return ae.reconstruction_errors_

    def select_data(self, X, y):
        X = esae_core.to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        target_keep_count = int(n_samples * (1.0 - self.target_reduction))
        print(f"[SublinearAEIS] Target reduction: {self.target_reduction:.2%} "
              f"(Keeping {target_keep_count} / {n_samples} instances)")

        # Step 1: autoencoder scoring
        print(f"[SublinearAEIS] Step 1 — Training Autoencoder for {self.ae_epochs} epochs...")
        t0 = time.time()
        scores = self._reconstruction_errors(X, y)
        self.scores_ = scores
        t_ae = time.time() - t0
        print(f"[SublinearAEIS] Autoencoder scores: min={scores.min():.4f}, "
              f"median={np.median(scores):.4f}, max={scores.max():.4f}")

        # Step 2: tessellation
        n_clusters = esae_core.resolve_n_clusters(self.n_clusters, n_samples, max_divisor=10)
        print(f"[SublinearAEIS] Step 2 — Tessellating space into {n_clusters} micro-clusters...")
        t0 = time.time()
        labels, sizes = esae_core.tessellate(X, n_clusters, self.random_state)
        self.labels_ = labels
        self.cluster_sizes_ = sizes
        t_tess = time.time() - t0

        # Step 3: sublinear quotas; step 4: score-weighted removal inside each cell
        t0 = time.time()
        lam, quotas = esae_core.sublinear_quotas(sizes, self.gamma, target_keep_count)
        self.cluster_quotas_ = quotas
        print(f"[SublinearAEIS] Step 3 — Sublinear sampling (gamma={self.gamma}). lambda = {lam:.4f}")
        weights = esae_core.removal_weights(scores, self.mode, self.beta)
        mask = esae_core.remove_within_cells(labels, quotas, np.random.RandomState(self.random_state), weights)
        t_sample = time.time() - t0

        self._set_selection(
            X, y, mask,
            n_clusters=n_clusters,
            target_reduction=self.target_reduction,
            target_keep_count=target_keep_count,
            sublinear_lambda=lam,
            **esae_core.cell_summary(sizes, quotas),
            **esae_core.score_summary(scores),
            time_autoencoder=t_ae,
            time_tessellation=t_tess,
            time_sampling=t_sample,
        )

        print(f"[SublinearAEIS] Done — removed {n_samples - len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_
