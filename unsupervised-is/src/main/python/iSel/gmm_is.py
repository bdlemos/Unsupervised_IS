"""Gaussian Mixture based Unsupervised Instance Selection (GMM-IS)."""

from __future__ import annotations

from typing import Optional

import copy

import numpy as np
import scipy
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

from src.main.python.iSel.base import InstanceSelectionMixin


class GMMIS(InstanceSelectionMixin):
    """GMM-based unsupervised instance selection.

    The selector keeps the middle band of the log-likelihood distribution:
    low likelihood is treated as noise and very high likelihood as redundant.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        low_percentile: float = 10.0,
        high_percentile: float = 90.0,
        beta: float = 0.0,
        theta: float = 0.0,
        covariance_type: str = "diag",
        random_state: int = 0,
    ) -> None:
        """
        initialize the GMMIS instance selector.
        :param n_components: Number of GMM components. If None, it will be determined based on the number of samples (between 2 and 20).
        :param low_percentile: Percentile for redundancy threshold (lower bound of log-likelihood).
        :param high_percentile: Percentile for noise threshold (upper bound of log-likelihood).
        :param beta: Additional partial redundancy removal rate applied *within* the low error band. 0 means no additional removal, 1 means remove all below low_percentile. Values in (0, 1) allow partial removal via probability-weighted sampling.
        :param theta: Additional partial noise removal rate applied *within* the high error band. 0 means no additional removal, 1 means remove all above high_percentile. Values in (0, 1) allow partial removal via probability-weighted sampling.
        :param covariance_type: Covariance type for GMM ('full', 'tied', 'diag', 'spherical').
        :param random_state: Random seed for reproducibility.

        """
        self.n_components = n_components
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.beta = beta
        self.theta = theta
        self.covariance_type = covariance_type
        self.random_state = random_state

        self.sample_indices_ = []

    def _to_dense(self, X: np.ndarray) -> np.ndarray:
        if scipy.sparse.issparse(X):
            return X.toarray()
        return np.asarray(X, dtype=np.float64)

    def _resolve_n_components(self, n_samples: int) -> int:
        if self.n_components is not None:
            return max(1, min(int(self.n_components), n_samples))
        estimate = int(np.sqrt(n_samples))
        return max(2, min(estimate, 20))

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        X_dense = self._to_dense(X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_dense)

        n_components = self._resolve_n_components(X_scaled.shape[0])
        self.gmm_ = GaussianMixture(
            n_components=n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        self.gmm_.fit(X_scaled)

        scores = self.gmm_.score_samples(X_scaled)
        # Negate scores so that lower values mean redundant and higher values mean noise,
        # aligning with Perplexity-IS and Autoencoder-IS.
        scores = -scores
        self.gmm_scores_ = copy.copy(scores)
        return scores

    def _identify_redundant(self, scores: np.ndarray) -> np.ndarray:
        self.low_threshold_ = np.percentile(scores, self.low_percentile)
        redundant_idx = np.where(scores < self.low_threshold_)[0]

        if self.beta < 1.0 and len(redundant_idx) > 0:
            weights = 1.0 / (np.abs(scores[redundant_idx]) + 1e-12)
            weights /= weights.sum()
            n_to_remove = max(0, int(len(redundant_idx) * self.beta))
            print(f"[GMM] Redundant instances: {len(redundant_idx)}, removing {n_to_remove} with beta={self.beta}")
            rng = np.random.RandomState(self.random_state)
            redundant_idx = rng.choice(
                redundant_idx, size=n_to_remove, replace=False, p=weights
            )

        return redundant_idx

    def _identify_noise(self, scores):
        self.high_threshold_ = np.percentile(scores, self.high_percentile)
        noise_idx = np.where(scores > self.high_threshold_)[0]

        if self.theta < 1.0 and len(noise_idx) > 0:
            weights = scores[noise_idx] - scores[noise_idx].min() + 1e-12
            weights /= weights.sum()
            n_to_remove = max(0, int(len(noise_idx) * self.theta))
            print(f"[GMM] Noise instances: {len(noise_idx)}, removing {n_to_remove} with theta={self.theta}")
            rng = np.random.RandomState(
                self.random_state + 1 if self.random_state is not None else None
            )
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

        print(f"[GMM] Removed {len(idx_redundant)} redundant + "
              f"{len(idx_noise)} noisy = "
              f"{len(idx_redundant) + len(idx_noise)} total instances")
        print(f"[GMM] Kept {len(self.y_)} / {len_original_y} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_
