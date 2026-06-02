"""Isolation Forest based Unsupervised Instance Selection (IForest-IS)."""

from __future__ import annotations

from typing import Optional, Tuple

import copy

import numpy as np
import scipy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

from src.main.python.iSel.base import InstanceSelectionMixin


class IForestIS(InstanceSelectionMixin):
    """Isolation-Forest based unsupervised instance selection.

    IsolationForest.score_samples() returns scores in [-1, 0]:
        - Close to  0  → most NORMAL   (hard to isolate)
        - Close to -1  → most ANOMALOUS (easy to isolate)

    This selector removes both extremes of the score distribution:
        - Bottom percentile (most anomalous scores, closest to -1) → treated as NOISE
        - Top percentile   (most normal scores,    closest to  0) → treated as REDUNDANT

    Only the middle band is kept as the selected instances.

    Score axis illustration:
        -1.0 ----[NOISE]----|----- kept -----|----[REDUNDANT]---- 0.0
              (anomalous)   ^                ^   (normal/dense)
                        low_threshold   high_threshold
    """

    def __init__(
        self,
        low_percentile: float = 10.0,
        high_percentile: float = 90.0,
        beta: float = 0.0,
        theta: float = 0.0,
        n_estimators: int = 200,
        max_samples: str = "auto",
        contamination: str = "auto",
        random_state: int = 0,
    ) -> None:
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.beta = beta
        self.theta = theta
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.sample_indices_ = []

    def _to_dense(self, X: np.ndarray) -> np.ndarray:
        if scipy.sparse.issparse(X):
            return X.toarray()
        return np.asarray(X, dtype=np.float64)

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        X_dense = self._to_dense(X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_dense)

        self.iforest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.iforest_.fit(X_scaled)

        # Scores in [-1, 0]: 0 = most normal, -1 = most anomalous.
        scores = self.iforest_.score_samples(X_scaled)
        self.anomaly_scores_ = copy.copy(scores)
        return scores

    def _identify_noise(self, scores: np.ndarray) -> np.ndarray:
        """Remove the bottom percentile: most anomalous scores (closest to -1).

        These are outliers/noise that are easy to isolate and unlikely
        to represent the true underlying distribution.

        beta > 0 softens the removal: instead of discarding all candidates,
        only (1 - beta) fraction is removed, weighted toward the most anomalous.
            beta=0.0 → remove all below threshold (hard cut)
            beta=0.5 → remove 50% of candidates, prioritising the worst scores
            beta=1.0 → remove nothing
        """
        self.low_threshold_ = np.percentile(scores, self.low_percentile)
        noise_idx = np.where(scores < self.low_threshold_)[0]

        if self.beta > 0.0 and len(noise_idx) > 0:
            # Higher weight → more anomalous (smaller/more negative score)
            weights = 1.0 / (np.abs(scores[noise_idx]) + 1e-12)
            weights /= weights.sum()
            n_to_remove = max(1, int(len(noise_idx) * (1.0 - self.beta)))
            rng = np.random.RandomState(self.random_state)
            noise_idx = rng.choice(
                noise_idx, size=n_to_remove, replace=False, p=weights
            )

        return noise_idx

    def _identify_redundant(self, scores: np.ndarray) -> np.ndarray:
        """Remove the top percentile: most normal scores (closest to 0).

        These are overly dense/typical points that add little information
        and can safely be pruned to reduce dataset size.

        theta > 0 softens the removal: instead of discarding all candidates,
        only (1 - theta) fraction is removed, weighted toward the most normal.
            theta=0.0 → remove all above threshold (hard cut)
            theta=0.5 → remove 50% of candidates, prioritising the densest scores
            theta=1.0 → remove nothing
        """
        self.high_threshold_ = np.percentile(scores, self.high_percentile)
        redundant_idx = np.where(scores > self.high_threshold_)[0]

        if self.theta > 0.0 and len(redundant_idx) > 0:
            # Higher weight → more normal (score closer to 0, i.e. larger value)
            weights = scores[redundant_idx] - scores[redundant_idx].min() + 1e-12
            weights /= weights.sum()
            n_to_remove = max(1, int(len(redundant_idx) * (1.0 - self.theta)))
            rng = np.random.RandomState(
                self.random_state + 1 if self.random_state is not None else None
            )
            redundant_idx = rng.choice(
                redundant_idx, size=n_to_remove, replace=False, p=weights
            )

        return redundant_idx

    def select_data(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)
        self.mask = np.ones(y.size, dtype=bool)

        scores = self._compute_scores(X)

        idx_noise = self._identify_noise(scores)
        self._idx_noise = idx_noise
        self.mask[idx_noise] = False

        idx_redundant = self._identify_redundant(scores)
        self._idx_redundant = idx_redundant
        self.mask[idx_redundant] = False

        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = np.asarray(range(len(y)))[self.mask]
        self.reduction_ = 1.0 - float(len(self.y_)) / len_original_y

        return self.X_, self.y_