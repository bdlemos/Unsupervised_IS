"""
Entropy-Adaptive Sublinear Autoencoder Instance Selection (EntropySublinearAEIS)
================================================================================

Extension of SublinearAEIS that automatically determines the reduction rate
from the normalized entropy of the micro-cluster size distribution:

    H = -sum(p_c * log(p_c)) / log(K),   p_c = S_c / N
    target_reduction = r_max * (1 - H^alpha), clamped to [5%, r_max]

- High entropy (H ≈ 1.0) → clusters are uniform → conservative reduction.
- Low entropy (H ≪ 1.0) → few clusters dominate → aggressive reduction.

The rate needs no per-dataset tuning; r_max and alpha are fixed constants.

Parameters
----------
r_max : float, default=0.50
    Maximum allowed reduction rate.
alpha : float, default=15.0
    Entropy sensitivity exponent. Higher = more conservative.
n_clusters : int, default=-1
    Number of micro-clusters. None, <= 0 or >= 99999 means sqrt(N).
gamma : float, default=0.5
    The sublinear scaling factor (0.5 = square root).
ae_epochs : int, default=50
    Number of epochs to train the autoencoder.
random_state : int, default=0
mode : {"original", "cleaning"}, default="original"
    Direction of the reconstruction-error weighting inside each cell.
    "original" (the paper's method) removes low-error instances first.
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


class EntropySublinearAEIS(InstanceSelectionMixin):

    def __init__(
        self,
        r_max: float = 0.50,
        alpha: float = 15.0,
        n_clusters: int = -1,
        gamma: float = 0.5,
        ae_epochs: int = 50,
        random_state: int = 0,
        mode: str = "original",
        beta: float = 1.0,
        **ae_kwargs
    ) -> None:
        self.r_max = r_max
        self.alpha = alpha
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
            low_percentile=0.0,
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

        # Step 1: autoencoder scoring
        print(f"[EntropySublinearAEIS] Step 1 — Training Autoencoder for {self.ae_epochs} epochs...")
        t0 = time.time()
        scores = self._reconstruction_errors(X, y)
        self.scores_ = scores
        t_ae = time.time() - t0
        print(f"[EntropySublinearAEIS] AE scores: min={scores.min():.4f}, "
              f"median={np.median(scores):.4f}, max={scores.max():.4f}")

        # Step 2: tessellation
        n_clusters = esae_core.resolve_n_clusters(self.n_clusters, n_samples)
        print(f"[EntropySublinearAEIS] Step 2 — Tessellating into {n_clusters} micro-clusters using minibatchkmeans...")
        t0 = time.time()
        labels, sizes = esae_core.tessellate(X, n_clusters, self.random_state)
        self.labels_ = labels
        self.cluster_sizes_ = sizes
        t_tess = time.time() - t0

        # Step 3: entropy-adaptive rate
        H = esae_core.normalized_entropy(sizes)
        target_reduction = esae_core.entropy_rate(H, self.r_max, self.alpha)
        self.entropy_ = H
        self.target_reduction_ = target_reduction
        target_keep_count = int(n_samples * (1.0 - target_reduction))
        print(f"[EntropySublinearAEIS] Step 3 — Entropy H={H:.4f}, "
              f"adaptive reduction={target_reduction:.2%} "
              f"(Keeping {target_keep_count} / {n_samples})")

        # Step 4: sublinear quotas and score-weighted removal inside each cell
        t0 = time.time()
        lam, quotas = esae_core.sublinear_quotas(sizes, self.gamma, target_keep_count)
        self.cluster_quotas_ = quotas
        print(f"[EntropySublinearAEIS] Step 4 — Sublinear sampling (gamma={self.gamma}). lambda = {lam:.4f}")
        weights = esae_core.removal_weights(scores, self.mode, self.beta)
        mask = esae_core.remove_within_cells(labels, quotas, np.random.RandomState(self.random_state), weights)
        t_sample = time.time() - t0

        self._set_selection(
            X, y, mask,
            n_clusters=n_clusters,
            entropy=H,
            target_reduction=target_reduction,
            target_keep_count=target_keep_count,
            sublinear_lambda=lam,
            **esae_core.cell_summary(sizes, quotas),
            **esae_core.score_summary(scores),
            time_autoencoder=t_ae,
            time_tessellation=t_tess,
            time_sampling=t_sample,
        )

        print(f"[EntropySublinearAEIS] Done — removed {n_samples - len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_
