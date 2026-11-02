"""
Ablation Variants for ESAE-IS Component Analysis
==================================================

Factors of the 2^k factorial (A = AE ranking, B = clustering, C = sublinear
quotas). Every variant uses the ESAE entropy-adaptive rate, computed on the
same K = sqrt(N) tessellation, so all cells remove the same amount:

1. AblationRandomESAERate           (A=0, B=0)    uniform random over the whole set
2. AblationClusterUniform           (A=0, B=1, C=0) same fraction removed from every cell, uniform
3. AblationClusterSublinearRandom   (A=0, B=1, C=1) sublinear quotas, uniform inside each cell
4. AblationAEESAERate               (A=1, B=0)    AE-weighted removal over the whole set
5. AblationClusterUniformAE         (A=1, B=1, C=0) same fraction per cell, AE-weighted inside

ESAE-IS itself is the (A=1, B=1, C=1) cell.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_X_y

from src.main.python.iSel import esae_core
from src.main.python.iSel.base import InstanceSelectionMixin
from src.main.python.iSel.autoencoder_is import AutoencoderIS


def _compute_esae_adaptive_rate(X, r_max: float, alpha: float, random_state: int) -> dict:
    """Tessellation + entropy rate exactly as in ESAE-IS (K = sqrt(N))."""
    n_clusters = esae_core.resolve_n_clusters(-1, X.shape[0])
    labels, sizes = esae_core.tessellate(X, n_clusters, random_state)
    H = esae_core.normalized_entropy(sizes)
    target_reduction = esae_core.entropy_rate(H, r_max, alpha)
    print(f"  [Ablation] Entropy H={H:.4f}, adaptive reduction={target_reduction:.2%}")
    return {
        "labels": labels,
        "sizes": sizes,
        "n_clusters": n_clusters,
        "entropy": H,
        "target_reduction": target_reduction,
    }


def _ae_scores(X, y, ae_epochs: int, random_state: int) -> np.ndarray:
    ae = AutoencoderIS(
        n_epochs=ae_epochs,
        low_percentile=0.0,
        high_percentile=100.0,
        beta=0.0,
        theta=0.0,
        random_state=random_state,
    )
    ae.fit(X, y)
    return ae.reconstruction_errors_


def _rate_info(rate: dict) -> dict:
    """Decision values shared by every ablation cell."""
    return {k: rate[k] for k in ("n_clusters", "entropy", "target_reduction")}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Random @ ESAE adaptive rate
# ══════════════════════════════════════════════════════════════════════════════

class AblationRandomESAERate(InstanceSelectionMixin):
    """Uniform random sampling over the whole set at the ESAE adaptive rate."""

    def __init__(
        self,
        r_max: float = 0.50,
        alpha: float = 15.0,
        gamma: float = 0.5,
        random_state: int = 13,
    ) -> None:
        self.r_max = r_max
        self.alpha = alpha
        self.gamma = gamma
        self.random_state = random_state
        self.sample_indices_ = []

    def select_data(self, X, y):
        X = esae_core.to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        print(f"[AblationRandomESAERate] Computing adaptive rate...")
        rate = _compute_esae_adaptive_rate(X, self.r_max, self.alpha, self.random_state)
        self.target_reduction = rate["target_reduction"]

        keep_count = int(n_samples * (1.0 - self.target_reduction))
        rng = np.random.RandomState(self.random_state)
        keep_idx = rng.choice(n_samples, size=keep_count, replace=False)
        mask = np.zeros(n_samples, dtype=bool)
        mask[keep_idx] = True

        self._set_selection(X, y, mask, **_rate_info(rate), target_keep_count=keep_count)
        print(f"[AblationRandomESAERate] Done — kept {len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")
        return self.X_, self.y_


# ══════════════════════════════════════════════════════════════════════════════
# 2. Clustering + Uniform (linear) removal, no AE
# ══════════════════════════════════════════════════════════════════════════════

class AblationClusterUniform(InstanceSelectionMixin):
    """Removes the same fraction from every cell, uniformly at random."""

    def __init__(
        self,
        r_max: float = 0.50,
        alpha: float = 15.0,
        random_state: int = 13,
    ) -> None:
        self.r_max = r_max
        self.alpha = alpha
        self.random_state = random_state
        self.sample_indices_ = []

    def select_data(self, X, y):
        X = esae_core.to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        print(f"[AblationClusterUniform] Computing adaptive rate + clustering...")
        rate = _compute_esae_adaptive_rate(X, self.r_max, self.alpha, self.random_state)
        self.target_reduction = rate["target_reduction"]

        quotas = esae_core.linear_quotas(rate["sizes"], self.target_reduction)
        mask = esae_core.remove_within_cells(rate["labels"], quotas, np.random.RandomState(self.random_state))

        self._set_selection(X, y, mask, **_rate_info(rate),
                            **esae_core.cell_summary(rate["sizes"], quotas))
        print(f"[AblationClusterUniform] Done — kept {len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")
        return self.X_, self.y_


# ══════════════════════════════════════════════════════════════════════════════
# 3. Clustering + Sublinear cap + Random removal (no AE)
# ══════════════════════════════════════════════════════════════════════════════

class AblationClusterSublinearRandom(InstanceSelectionMixin):
    """ESAE geometry (sublinear quotas) with uniform removal inside each cell — ESAE without the AE."""

    def __init__(
        self,
        r_max: float = 0.50,
        alpha: float = 15.0,
        gamma: float = 0.5,
        random_state: int = 13,
    ) -> None:
        self.r_max = r_max
        self.alpha = alpha
        self.gamma = gamma
        self.random_state = random_state
        self.sample_indices_ = []

    def select_data(self, X, y):
        X = esae_core.to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        print(f"[AblationClusterSublinearRandom] Computing adaptive rate + clustering...")
        rate = _compute_esae_adaptive_rate(X, self.r_max, self.alpha, self.random_state)
        self.target_reduction = rate["target_reduction"]

        target_keep_count = int(n_samples * (1.0 - self.target_reduction))
        lam, quotas = esae_core.sublinear_quotas(rate["sizes"], self.gamma, target_keep_count)
        print(f"[AblationClusterSublinearRandom] Sublinear lambda={lam:.4f}")
        mask = esae_core.remove_within_cells(rate["labels"], quotas, np.random.RandomState(self.random_state))

        self._set_selection(X, y, mask, **_rate_info(rate),
                            target_keep_count=target_keep_count, sublinear_lambda=lam,
                            **esae_core.cell_summary(rate["sizes"], quotas))
        print(f"[AblationClusterSublinearRandom] Done — kept {len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")
        return self.X_, self.y_


# ══════════════════════════════════════════════════════════════════════════════
# 4. AE ranking @ ESAE adaptive rate, NO clustering (A=1, B=0, C=—)
# ══════════════════════════════════════════════════════════════════════════════

class AblationAEESAERate(InstanceSelectionMixin):
    """AE-weighted removal over the whole set (p_remove ∝ 1/(RE+ε), as ESAE), no tessellation."""

    def __init__(
        self,
        r_max: float = 0.50,
        alpha: float = 15.0,
        gamma: float = 0.5,
        ae_epochs: int = 50,
        random_state: int = 13,
    ) -> None:
        self.r_max = r_max
        self.alpha = alpha
        self.gamma = gamma
        self.ae_epochs = ae_epochs
        self.random_state = random_state
        self.sample_indices_ = []

    def select_data(self, X, y):
        X = esae_core.to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        print(f"[AblationAEESAERate] Computing adaptive rate...")
        rate = _compute_esae_adaptive_rate(X, self.r_max, self.alpha, self.random_state)
        self.target_reduction = rate["target_reduction"]

        print(f"[AblationAEESAERate] Training Autoencoder for {self.ae_epochs} epochs...")
        scores = _ae_scores(X, y, self.ae_epochs, self.random_state)
        self.scores_ = scores

        # One global "cell" holding every instance
        remove_count = int(n_samples * self.target_reduction)
        mask = esae_core.remove_within_cells(
            np.zeros(n_samples, dtype=np.int64),
            np.array([n_samples - remove_count]),
            np.random.RandomState(self.random_state),
            esae_core.removal_weights(scores, "original"),
        )

        self._set_selection(X, y, mask, **_rate_info(rate), **esae_core.score_summary(scores))
        print(f"[AblationAEESAERate] Done — kept {len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")
        return self.X_, self.y_


# ══════════════════════════════════════════════════════════════════════════════
# 5. Clustering + Uniform (linear) removal + AE ranking (A=1, B=1, C=0)
# ══════════════════════════════════════════════════════════════════════════════

class AblationClusterUniformAE(InstanceSelectionMixin):
    """Same fraction removed per cell, AE-weighted inside each cell (ESAE without sublinear quotas)."""

    def __init__(
        self,
        r_max: float = 0.50,
        alpha: float = 15.0,
        ae_epochs: int = 50,
        random_state: int = 13,
    ) -> None:
        self.r_max = r_max
        self.alpha = alpha
        self.ae_epochs = ae_epochs
        self.random_state = random_state
        self.sample_indices_ = []

    def select_data(self, X, y):
        X = esae_core.to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        print(f"[AblationClusterUniformAE] Computing adaptive rate + clustering...")
        rate = _compute_esae_adaptive_rate(X, self.r_max, self.alpha, self.random_state)
        self.target_reduction = rate["target_reduction"]

        print(f"[AblationClusterUniformAE] Training Autoencoder for {self.ae_epochs} epochs...")
        scores = _ae_scores(X, y, self.ae_epochs, self.random_state)
        self.scores_ = scores

        quotas = esae_core.linear_quotas(rate["sizes"], self.target_reduction)
        mask = esae_core.remove_within_cells(
            rate["labels"], quotas, np.random.RandomState(self.random_state),
            esae_core.removal_weights(scores, "original"),
        )

        self._set_selection(X, y, mask, **_rate_info(rate),
                            **esae_core.cell_summary(rate["sizes"], quotas),
                            **esae_core.score_summary(scores))
        print(f"[AblationClusterUniformAE] Done — kept {len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")
        return self.X_, self.y_
