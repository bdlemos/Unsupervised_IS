"""
Ablation Variants for ESAE-IS Component Analysis
==================================================

Three ablation methods to isolate the contribution of each ESAE-IS component:

1. AblationRandomESAERate
   - Computes the adaptive rate via entropy (same as ESAE) but applies RANDOM
     sampling instead of the full AE + sublinear pipeline.
   - Answers: "What does random removal at the same adaptive rate achieve?"

2. AblationClusterUniform
   - Uses MiniBatchKMeans tessellation + LINEAR proportional removal per cluster
     (not sublinear), without AE scores (uniform random within each cluster).
   - Answers: "What does tessellation alone contribute, without sublinear or AE?"

3. AblationClusterSublinearRandom
   - Uses MiniBatchKMeans tessellation + SUBLINEAR cap per cluster, but removes
     instances randomly within each cluster (no AE scoring).
   - Answers: "What does the sublinear cap add over uniform, without AE guidance?"
"""

from __future__ import annotations

import numpy as np
import scipy
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.validation import check_X_y

from src.main.python.iSel.base import InstanceSelectionMixin
from src.main.python.iSel.autoencoder_is import AutoencoderIS


# ── Shared helpers ──────────────────────────────────────────────────────────

def _compute_normalized_entropy(sizes: np.ndarray) -> float:
    """Compute the normalized entropy of the cluster size distribution."""
    total = np.sum(sizes)
    if total == 0:
        return 1.0
    nonzero = sizes[sizes > 0]
    k = len(nonzero)
    if k <= 1:
        return 1.0
    probs = nonzero / total
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(k)
    return entropy / max_entropy


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


def _to_dense(X):
    if scipy.sparse.issparse(X):
        return X.toarray()
    return np.asarray(X, dtype=np.float64)


def _compute_esae_adaptive_rate(
    X, r_max: float, alpha: float, random_state: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run the clustering + entropy computation from ESAE to get the adaptive rate.

    Returns (target_reduction, labels, sizes).
    """
    n_samples = X.shape[0]
    n_clusters = int(np.sqrt(n_samples))

    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
    labels = km.fit_predict(X)
    sizes = np.bincount(labels, minlength=n_clusters)

    H = _compute_normalized_entropy(sizes)
    target_reduction = r_max * (1.0 - H ** alpha)
    target_reduction = max(0.05, min(target_reduction, r_max))

    print(f"  [Ablation] Entropy H={H:.4f}, adaptive reduction={target_reduction:.2%}")

    return target_reduction, labels, sizes


# ══════════════════════════════════════════════════════════════════════════════
# 1. Random @ ESAE adaptive rate
# ══════════════════════════════════════════════════════════════════════════════

class AblationRandomESAERate(InstanceSelectionMixin):
    """Random sampling using the ESAE entropy-adaptive rate.

    Computes the adaptive reduction via clustering + entropy (exactly as ESAE
    does), then applies pure random sampling at that rate.  This isolates
    the contribution of the AE scoring + sublinear geometry.
    """

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
        X = _to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        print(f"[AblationRandomESAERate] Computing adaptive rate...")
        target_reduction, _, _ = _compute_esae_adaptive_rate(
            X, self.r_max, self.alpha, self.random_state
        )

        # Pure random sampling at the adaptive rate
        keep_count = int(n_samples * (1.0 - target_reduction))
        rng = np.random.RandomState(self.random_state)
        keep_idx = rng.choice(n_samples, size=keep_count, replace=False)
        keep_idx.sort()

        self.X_ = np.asarray(X[keep_idx])
        self.y_ = np.asarray(y[keep_idx])
        self.sample_indices_ = keep_idx
        self.reduction_ = 1.0 - float(len(self.y_)) / n_samples
        self.target_reduction = target_reduction

        # Compatibility attributes
        self.beta = self.target_reduction
        self.theta = 0.0
        self.low_percentile = 0.0
        self.high_percentile = 100.0

        print(f"[AblationRandomESAERate] Done — kept {len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_


# ══════════════════════════════════════════════════════════════════════════════
# 2. Clustering + Uniform (linear) removal
# ══════════════════════════════════════════════════════════════════════════════

class AblationClusterUniform(InstanceSelectionMixin):
    """Tessellation with uniform (linear) proportional removal per cluster.

    Uses MiniBatchKMeans to create micro-clusters, then removes a FIXED
    fraction of instances from EACH cluster uniformly at random.  No AE
    scores, no sublinear cap.  The global reduction rate is computed via
    the same entropy-adaptive formula as ESAE for a fair comparison.
    """

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
        X = _to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        print(f"[AblationClusterUniform] Computing adaptive rate + clustering...")
        target_reduction, labels, sizes = _compute_esae_adaptive_rate(
            X, self.r_max, self.alpha, self.random_state
        )

        n_clusters = len(sizes)
        rng = np.random.RandomState(self.random_state)
        mask = np.ones(n_samples, dtype=bool)

        # Linear proportional removal: remove the same fraction from every cluster
        for c in range(n_clusters):
            cluster_idx = np.where(labels == c)[0]
            s_c = len(cluster_idx)
            if s_c == 0:
                continue

            remove_count = int(s_c * target_reduction)
            if remove_count > 0:
                to_remove = rng.choice(cluster_idx, size=remove_count, replace=False)
                mask[to_remove] = False

        self.X_ = np.asarray(X[mask])
        self.y_ = np.asarray(y[mask])
        self.sample_indices_ = np.where(mask)[0]
        self.reduction_ = 1.0 - float(len(self.y_)) / n_samples
        self.target_reduction = target_reduction

        # Compatibility attributes
        self.beta = self.target_reduction
        self.theta = 0.0
        self.low_percentile = 0.0
        self.high_percentile = 100.0

        print(f"[AblationClusterUniform] Done — kept {len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_


# ══════════════════════════════════════════════════════════════════════════════
# 3. Clustering + Sublinear cap + Random removal (no AE)
# ══════════════════════════════════════════════════════════════════════════════

class AblationClusterSublinearRandom(InstanceSelectionMixin):
    """Tessellation with sublinear cap and random removal within clusters.

    Same geometry as ESAE (MiniBatchKMeans + sublinear cap on cluster size),
    but instances are removed UNIFORMLY at random within each cluster — no
    AE reconstruction error guidance.  This isolates the AE contribution.
    """

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
        X = _to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        print(f"[AblationClusterSublinearRandom] Computing adaptive rate + clustering...")
        target_reduction, labels, sizes = _compute_esae_adaptive_rate(
            X, self.r_max, self.alpha, self.random_state
        )

        n_clusters = len(sizes)
        target_keep_count = int(n_samples * (1.0 - target_reduction))
        lam = _find_lambda(sizes, self.gamma, target_keep_count)

        print(f"[AblationClusterSublinearRandom] Sublinear lambda={lam:.4f}")

        rng = np.random.RandomState(self.random_state)
        mask = np.ones(n_samples, dtype=bool)

        for c in range(n_clusters):
            cluster_idx = np.where(labels == c)[0]
            s_c = len(cluster_idx)
            if s_c == 0:
                continue

            keep_count = min(s_c, int(np.ceil(lam * (s_c ** self.gamma))))
            remove_count = s_c - keep_count

            if remove_count > 0:
                # Random removal (NO AE scores)
                to_remove = rng.choice(cluster_idx, size=remove_count, replace=False)
                mask[to_remove] = False

        self.X_ = np.asarray(X[mask])
        self.y_ = np.asarray(y[mask])
        self.sample_indices_ = np.where(mask)[0]
        self.reduction_ = 1.0 - float(len(self.y_)) / n_samples
        self.target_reduction = target_reduction

        # Compatibility attributes
        self.beta = self.target_reduction
        self.theta = 0.0
        self.low_percentile = 0.0
        self.high_percentile = 100.0

        print(f"[AblationClusterSublinearRandom] Done — kept {len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_


# ══════════════════════════════════════════════════════════════════════════════
# 4. AE ranking @ ESAE adaptive rate, NO clustering (A=1, B=0, C=—)
# ══════════════════════════════════════════════════════════════════════════════

class AblationAEESAERate(InstanceSelectionMixin):
    """AE-guided removal at the ESAE entropy-adaptive rate, without clustering.

    Computes the adaptive reduction via clustering + entropy (exactly as all
    other ablation variants), trains the autoencoder for reconstruction error
    scores, then removes instances from the **global pool** using probabilistic
    AE-guided removal (p_remove ∝ 1/(score+ε), same as ESAE).

    No spatial partitioning is used for the removal step — this isolates the
    AE contribution without any tessellation geometry.
    """

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
        X = _to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        # ── Step 1: Compute adaptive rate (same as all ablation variants) ──
        print(f"[AblationAEESAERate] Computing adaptive rate...")
        target_reduction, _, _ = _compute_esae_adaptive_rate(
            X, self.r_max, self.alpha, self.random_state
        )

        # ── Step 2: Train autoencoder for reconstruction error scores ──────
        print(f"[AblationAEESAERate] Training Autoencoder for {self.ae_epochs} epochs...")
        ae = AutoencoderIS(
            n_epochs=self.ae_epochs,
            low_percentile=0.0,
            high_percentile=100.0,
            beta=0.0,
            theta=0.0,
            random_state=self.random_state,
        )
        ae.fit(X, y)
        scores = ae.reconstruction_errors_

        print(f"[AblationAEESAERate] AE scores: min={scores.min():.4f}, "
              f"median={np.median(scores):.4f}, max={scores.max():.4f}")

        # ── Step 3: Probabilistic AE-guided removal from global pool ───────
        remove_count = int(n_samples * target_reduction)

        if remove_count > 0:
            # Same weighting as ESAE: low error → high removal probability
            inv_scores = 1.0 / (scores + 1e-8)
            p_remove = inv_scores / np.sum(inv_scores)

            rng = np.random.RandomState(self.random_state)
            to_remove = rng.choice(
                n_samples, size=remove_count, replace=False, p=p_remove
            )
            mask = np.ones(n_samples, dtype=bool)
            mask[to_remove] = False
        else:
            mask = np.ones(n_samples, dtype=bool)

        self.X_ = np.asarray(X[mask])
        self.y_ = np.asarray(y[mask])
        self.sample_indices_ = np.where(mask)[0]
        self.reduction_ = 1.0 - float(len(self.y_)) / n_samples
        self.target_reduction = target_reduction

        # Compatibility attributes
        self.beta = self.target_reduction
        self.theta = 0.0
        self.low_percentile = 0.0
        self.high_percentile = 100.0

        print(f"[AblationAEESAERate] Done — kept {len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_


# ══════════════════════════════════════════════════════════════════════════════
# 5. Clustering + Uniform (linear) removal + AE ranking (A=1, B=1, C=0)
# ══════════════════════════════════════════════════════════════════════════════

class AblationClusterUniformAE(InstanceSelectionMixin):
    """Tessellation with linear proportional removal, AE-guided within clusters.

    Same structure as AblationClusterUniform (linear proportional removal per
    cluster, no sublinear cap), but within each cluster instances are removed
    using probabilistic AE-guided selection (p_remove ∝ 1/(score+ε), same as
    ESAE) instead of uniform random.

    This isolates the sublinear cap contribution: comparing this cell against
    the full ESAE shows what the sublinear geometry adds on top of
    AE+clustering.
    """

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
        X = _to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)

        # ── Step 1: Compute adaptive rate + clustering ─────────────────────
        print(f"[AblationClusterUniformAE] Computing adaptive rate + clustering...")
        target_reduction, labels, sizes = _compute_esae_adaptive_rate(
            X, self.r_max, self.alpha, self.random_state
        )

        # ── Step 2: Train autoencoder for reconstruction error scores ──────
        print(f"[AblationClusterUniformAE] Training Autoencoder for {self.ae_epochs} epochs...")
        ae = AutoencoderIS(
            n_epochs=self.ae_epochs,
            low_percentile=0.0,
            high_percentile=100.0,
            beta=0.0,
            theta=0.0,
            random_state=self.random_state,
        )
        ae.fit(X, y)
        scores = ae.reconstruction_errors_

        print(f"[AblationClusterUniformAE] AE scores: min={scores.min():.4f}, "
              f"median={np.median(scores):.4f}, max={scores.max():.4f}")

        # ── Step 3: Linear proportional removal with AE guidance ───────────
        n_clusters = len(sizes)
        np.random.seed(self.random_state)
        mask = np.ones(n_samples, dtype=bool)

        for c in range(n_clusters):
            cluster_idx = np.where(labels == c)[0]
            s_c = len(cluster_idx)
            if s_c == 0:
                continue

            remove_count = int(s_c * target_reduction)
            if remove_count > 0:
                # AE-guided removal within cluster (same as ESAE)
                c_scores = scores[cluster_idx]
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

        self.X_ = np.asarray(X[mask])
        self.y_ = np.asarray(y[mask])
        self.sample_indices_ = np.where(mask)[0]
        self.reduction_ = 1.0 - float(len(self.y_)) / n_samples
        self.target_reduction = target_reduction

        # Compatibility attributes
        self.beta = self.target_reduction
        self.theta = 0.0
        self.low_percentile = 0.0
        self.high_percentile = 100.0

        print(f"[AblationClusterUniformAE] Done — kept {len(self.y_)} / {n_samples} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_
