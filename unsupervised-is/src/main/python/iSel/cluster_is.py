"""Cluster-Oriented Instance Selection (Cluster-IS)."""

from __future__ import annotations

from typing import Optional

import copy

import numpy as np
import scipy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y

from src.main.python.iSel.base import InstanceSelectionMixin


class ClusterIS(InstanceSelectionMixin):
    """Cluster-Oriented unsupervised instance selection.

    The selector uses K-Means clustering to identify representative and
    boundary instances:
    - **Center instances** (nearest to centroid): capture cluster essence.
    - **Border instances** (farthest from centroid): protect cluster boundaries.
    - **Middle instances** (in between): removed as redundant.

    This "dual sampling" strategy ensures balanced representation while
    removing the redundant middle band.

    References
    ==========
    Saha, S., Chatterjee, B., & Bandyopadhyay, S. (2022).
    Cluster-Oriented Instance Selection for Classification Problems.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        selection_rate: float = 0.5,
        center_ratio: float = 0.5,
        border_ratio: float = 0.5,
        random_state: int = 0,
    ) -> None:
        """
        Parameters
        ==========
        n_clusters : int or None, default=None
            Number of clusters for K-Means. If None, estimated as sqrt(n_samples).

        selection_rate : float, default=0.5
            Overall selection rate (0 < rate <= 1.0).
            E.g., 0.5 means keep 50% of instances.

        center_ratio : float, default=0.5
            Fraction of selection_rate to allocate to center instances.
            E.g., if selection_rate=0.5 and center_ratio=0.5,
            then 25% of original data comes from centers.

        border_ratio : float, default=0.5
            Fraction of selection_rate to allocate to border instances.
            E.g., if selection_rate=0.5 and border_ratio=0.5,
            then 25% of original data comes from borders.

        random_state : int, default=0
            Random state for reproducibility.
        """
        self.n_clusters = n_clusters
        self.selection_rate = selection_rate
        self.center_ratio = center_ratio
        self.border_ratio = border_ratio
        self.random_state = random_state

        self.sample_indices_ = []

    def _to_dense(self, X: np.ndarray) -> np.ndarray:
        if scipy.sparse.issparse(X):
            return X.toarray()
        return np.asarray(X, dtype=np.float64)

    def _resolve_n_clusters(self, n_samples: int) -> int:
        if self.n_clusters is not None:
            return max(2, min(int(self.n_clusters), n_samples))
        estimate = max(2, int(np.sqrt(n_samples)))
        return min(estimate, 50)

    def _fit_kmeans(self, X: np.ndarray) -> tuple:
        """Fit K-Means and compute per-instance distances to centroid."""
        n_samples = X.shape[0]
        n_clusters = self._resolve_n_clusters(n_samples)

        X_dense = self._to_dense(X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_dense)

        self.kmeans_ = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.kmeans_.fit_predict(X_scaled)
        centroids = self.kmeans_.cluster_centers_

        # Compute distance from each instance to its cluster centroid
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            cluster_id = labels[i]
            distances[i] = np.linalg.norm(
                X_scaled[i] - centroids[cluster_id]
            )

        return labels, distances

    def _select_from_clusters(
        self, labels: np.ndarray, distances: np.ndarray
    ) -> np.ndarray:
        """Select center and border instances from each cluster."""
        n_samples = len(labels)
        n_clusters = len(np.unique(labels))

        # Allocate budget
        total_to_select = max(1, int(n_samples * self.selection_rate))
        center_budget = max(1, int(total_to_select * self.center_ratio))
        border_budget = max(1, int(total_to_select * self.border_ratio))

        selected_indices = []

        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_distances = distances[cluster_indices]

            if len(cluster_indices) == 0:
                continue

            # Sort by distance
            sorted_idx = np.argsort(cluster_distances)
            sorted_indices = cluster_indices[sorted_idx]

            # Select centers (nearest to centroid)
            n_centers = max(1, int(len(cluster_indices) * self.center_ratio))
            center_indices = sorted_indices[:n_centers]
            selected_indices.extend(center_indices)

            # Select borders (farthest from centroid, but avoid extreme tail)
            n_borders = max(1, int(len(cluster_indices) * self.border_ratio))
            # Take from the tail, but skip the very last (extreme outliers)
            border_start = max(n_centers + 1, len(sorted_indices) - n_borders)
            border_indices = sorted_indices[border_start:]
            selected_indices.extend(border_indices)

        # Remove duplicates and sort
        selected_indices = np.unique(np.array(selected_indices))

        return selected_indices

    def select_data(self, X: np.ndarray, y: np.ndarray):
        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)
        self.mask = np.ones(y.size, dtype=bool)

        # Step 1: Fit K-Means and compute distances
        labels, distances = self._fit_kmeans(X)

        # Step 2: Select center and border instances
        selected_idx = self._select_from_clusters(labels, distances)

        # Step 3: Mark non-selected as False
        self.mask[:] = False
        self.mask[selected_idx] = True

        # Build outputs
        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])
        self.sample_indices_ = np.asarray(range(len(y)))[self.mask]
        self.reduction_ = 1.0 - float(len(self.y_)) / len_original_y

        return self.X_, self.y_
