"""
Shared building blocks of the ESAE-IS family
============================================

Every selector built on the ESAE geometry (ESAE-IS, SAE-IS and the ablation
variants) composes the same steps, implemented once here so the variants
cannot drift apart:

1. tessellate the space into K micro-clusters (K = sqrt(N) by default);
2. pick a global reduction rate (fixed, or entropy-adaptive);
3. give each cell a keep quota (sublinear lambda * |C|^gamma, or linear);
4. remove instances inside each cell, uniformly or weighted by a score.

All randomness goes through an explicit ``np.random.RandomState`` built from
the selector's ``random_state``; nothing touches the global numpy RNG.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
from sklearn.cluster import MiniBatchKMeans

# n_clusters values meaning "use sqrt(N)": None, any value <= 0, or this sentinel and above.
SQRT_N_SENTINEL = 99999

REMOVAL_MODES = ("original", "cleaning")


def to_dense(X) -> np.ndarray:
    if scipy.sparse.issparse(X):
        return X.toarray()
    return np.asarray(X, dtype=np.float64)


def resolve_n_clusters(n_clusters, n_samples: int, max_divisor: int = 2) -> int:
    """Effective K. Explicit values are clamped to [2, n_samples // max_divisor]."""
    if n_clusters is None or n_clusters <= 0 or n_clusters >= SQRT_N_SENTINEL:
        return max(2, int(np.sqrt(n_samples)))
    return max(2, min(int(n_clusters), n_samples // max_divisor))


def tessellate(X: np.ndarray, n_clusters: int, random_state) -> tuple[np.ndarray, np.ndarray]:
    """MiniBatchKMeans micro-clusters. Returns (labels, sizes)."""
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
    labels = km.fit_predict(X)
    sizes = np.bincount(labels, minlength=n_clusters)
    return labels, sizes


def normalized_entropy(sizes: np.ndarray) -> float:
    """Entropy of the cell occupancy, normalized by log(#non-empty cells)."""
    total = np.sum(sizes)
    if total == 0:
        return 1.0
    nonzero = sizes[sizes > 0]
    k = len(nonzero)
    if k <= 1:
        return 1.0
    probs = nonzero / total
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(k))


def entropy_rate(H: float, r_max: float, alpha: float, r_min: float = 0.05) -> float:
    """ESAE rule r = r_max * (1 - H^alpha), clamped to [r_min, r_max]."""
    rate = r_max * (1.0 - H ** alpha)
    return float(max(r_min, min(rate, r_max)))


def find_lambda(sizes: np.ndarray, gamma: float, target_keep_count: int) -> float:
    """Smallest lambda (by bisection) with sum_c min(|C_c|, ceil(lambda |C_c|^gamma)) >= target."""
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


def sublinear_quotas(sizes: np.ndarray, gamma: float, target_keep_count: int) -> tuple[float, np.ndarray]:
    """Keep quota per cell, min(|C_c|, ceil(lambda |C_c|^gamma)). Returns (lambda, quotas)."""
    lam = find_lambda(sizes, gamma, target_keep_count)
    quotas = np.array([min(s, int(np.ceil(lam * (s ** gamma)))) for s in sizes], dtype=np.int64)
    return lam, quotas


def linear_quotas(sizes: np.ndarray, rate: float) -> np.ndarray:
    """Keep quota per cell when every cell loses the same fraction ``rate``."""
    return np.array([s - int(s * rate) for s in sizes], dtype=np.int64)


def removal_weights(scores: np.ndarray, mode: str = "original", beta: float = 1.0) -> np.ndarray:
    """Unnormalized removal weights from reconstruction errors.

    original: low error is removed first (keeps hard instances), as in the paper.
    cleaning: high error is removed first (keeps prototypes).
    """
    if mode == "original":
        return (1.0 / (scores + 1e-8)) ** beta
    if mode == "cleaning":
        return (scores + 1e-8) ** beta
    raise ValueError(f"Unknown removal mode {mode!r}; expected one of {REMOVAL_MODES}")


def remove_within_cells(
    labels: np.ndarray,
    quotas: np.ndarray,
    rng: np.random.RandomState,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Keep mask after removing |C_c| - quota_c instances from each cell.

    Removal is uniform when ``weights`` is None, otherwise proportional to the
    weights of the instances in the cell (sampling without replacement).
    """
    mask = np.ones(len(labels), dtype=bool)
    for c in range(len(quotas)):
        cluster_idx = np.where(labels == c)[0]
        remove_count = len(cluster_idx) - int(quotas[c])
        if remove_count <= 0:
            continue
        if weights is None:
            to_remove = rng.choice(cluster_idx, size=remove_count, replace=False)
        else:
            w = weights[cluster_idx]
            local = rng.choice(len(cluster_idx), size=remove_count, replace=False, p=w / np.sum(w))
            to_remove = cluster_idx[local]
        mask[to_remove] = False
    return mask


def cell_summary(sizes: np.ndarray, quotas: np.ndarray) -> dict:
    """Descriptive stats of a tessellation and its quotas, for the selection record."""
    nonempty = sizes > 0
    pruned = quotas < sizes
    return {
        "n_cells": int(len(sizes)),
        "n_cells_nonempty": int(nonempty.sum()),
        "n_cells_pruned": int(pruned.sum()),
        "cell_size_min": int(sizes[nonempty].min()) if nonempty.any() else 0,
        "cell_size_median": float(np.median(sizes[nonempty])) if nonempty.any() else 0.0,
        "cell_size_max": int(sizes.max()) if len(sizes) else 0,
        "frac_points_in_untouched_cells": float(sizes[~pruned].sum() / max(1, sizes.sum())),
    }


def score_summary(scores: np.ndarray) -> dict:
    return {
        "re_min": float(np.min(scores)),
        "re_median": float(np.median(scores)),
        "re_max": float(np.max(scores)),
    }
