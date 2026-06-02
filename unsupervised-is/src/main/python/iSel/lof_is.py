"""Local Outlier Factor based Unsupervised Instance Selection (LOF-IS)."""

from __future__ import annotations

from typing import Optional

import copy

import numpy as np
import scipy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

from src.main.python.iSel.base import InstanceSelectionMixin


class LOFIS(InstanceSelectionMixin):
    """LOF-based unsupervised instance selection.

    Higher LOF normality scores mean the instance is more representative.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        low_percentile: float = 10.0,
        high_percentile: float = 90.0,
        beta: float = 0.0,
        theta: float = 0.0,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.beta = beta
        self.theta = theta

        self.sample_indices_ = []

    def _to_dense(self, X: np.ndarray) -> np.ndarray:
        if scipy.sparse.issparse(X):
            return X.toarray()
        return np.asarray(X, dtype=np.float64)

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        X_dense = self._to_dense(X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_dense)

        n_neighbors = max(2, min(self.n_neighbors, len(X_scaled) - 1))
        self.lof_ = LocalOutlierFactor(n_neighbors=n_neighbors)
        self.lof_.fit_predict(X_scaled)

        # negative_outlier_factor_: more negative = more anomalous.
        scores = -self.lof_.negative_outlier_factor_
        self.lof_scores_ = copy.copy(scores)
        return scores

    def _identify_redundant(self, scores: np.ndarray) -> np.ndarray:
        self.low_threshold_ = np.percentile(scores, self.low_percentile)
        redundant_idx = np.where(scores < self.low_threshold_)[0]

        if self.beta > 0.0 and len(redundant_idx) > 0:
            weights = 1.0 / (scores[redundant_idx] + 1e-12)
            weights /= weights.sum()
            n_to_remove = max(1, int(len(redundant_idx) * (1.0 - self.beta)))
            rng = np.random.RandomState(0)
            redundant_idx = rng.choice(
                redundant_idx, size=n_to_remove, replace=False, p=weights
            )

        return redundant_idx

    def _identify_noise(self, scores: np.ndarray) -> np.ndarray:
        self.high_threshold_ = np.percentile(scores, self.high_percentile)
        noise_idx = np.where(scores > self.high_threshold_)[0]

        if self.theta > 0.0 and len(noise_idx) > 0:
            weights = scores[noise_idx] - scores[noise_idx].min() + 1e-12
            weights /= weights.sum()
            n_to_remove = max(1, int(len(noise_idx) * (1.0 - self.theta)))
            rng = np.random.RandomState(1)
            noise_idx = rng.choice(
                noise_idx, size=n_to_remove, replace=False, p=weights
            )

        return noise_idx

    def select_data(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)
        self.mask = np.ones(y.size, dtype=bool)

        scores = self._compute_scores(X)

        idx_redundant = self._identify_redundant(scores)
        self._idx_redundant = idx_redundant
        self.mask[idx_redundant] = False

        idx_noise = self._identify_noise(scores)
        self._idx_noise = idx_noise
        self.mask[idx_noise] = False

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = np.asarray(range(len(y)))[self.mask]
        self.reduction_ = 1.0 - float(len(self.y_)) / len_original_y

        return self.X_, self.y_
