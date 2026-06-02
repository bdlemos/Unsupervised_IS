"""
Perplexity-based Unsupervised Instance Selection (Perplexity-IS)

An unsupervised instance selection framework that leverages perplexity from
Latent Dirichlet Allocation (LDA) topic modeling to simultaneously remove
redundant and noisy instances — without requiring class labels.
"""

import numpy as np
import copy
import warnings

from sklearn.utils.validation import check_X_y
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
import scipy

from src.main.python.iSel.base import InstanceSelectionMixin


class PerplexityIS(InstanceSelectionMixin):
    """Perplexity-based Unsupervised Instance Selection (Perplexity-IS)

    Description
    ===========

    A fully unsupervised instance selection method that uses perplexity from
    topic modeling to identify and remove both redundant and noisy instances.

    Perplexity measures how well a probabilistic model "explains" a given
    observation. In the context of instance selection:

    * **Low perplexity** → the instance is highly predictable within the
      discovered latent structure and is likely *redundant*.
    * **Medium-high perplexity** → the instance is informative, sitting at
      the boundary between clusters — helpful for generalization.
    * **Very high perplexity** → the instance is inconsistent with all
      discovered clusters and is likely *noise* or an error.

    Mathematically, perplexity is the exponential of the entropy of the
    posterior topic distribution for each document/instance:

        perplexity(x) = exp( H(p(z|x)) )

    where H is Shannon entropy and p(z|x) is the topic distribution for
    instance x.

    The method works in three steps:
        1. Fit an LDA topic model to discover latent structure.
        2. Compute per-instance perplexity from the topic distributions.
        3. Remove instances falling below the ``low_percentile`` threshold
           (redundant) or above the ``high_percentile`` threshold (noisy),
           keeping the informative middle band.

    Parameters
    ==========

    n_topics : int or str, default='auto'
        Number of topics (clusters) for LDA. When ``'auto'``, it is
        estimated as ``sqrt(n_samples)``, clamped to ``[5, 100]``.

    low_percentile : float, default=10.0
        Percentile below which instances are considered redundant and
        removed (low perplexity → too predictable).

    high_percentile : float, default=90.0
        Percentile above which instances are considered noise and removed
        (high perplexity → too surprising).

    beta : float, default=0.0
        Additional flat redundancy removal rate applied *within* the low
        perplexity band. When 1.0, all instances below ``low_percentile``
        are removed. Values in (0, 1) allow partial removal via
        probability-weighted sampling.

    theta : float, default=0.0
        Additional flat noise removal rate applied *within* the high
        perplexity band. When 1.0, all instances above ``high_percentile``
        are removed. Values in (0, 1) allow partial removal via
        probability-weighted sampling.

    max_iter_lda : int, default=30
        Maximum number of EM iterations for LDA fitting.

    random_state : int or None, default=0
        Random state for reproducibility.

    Attributes
    ==========

    mask : ndarray of shape (n_samples,)
        Boolean array indicating the selected instances.

    X_ : ndarray
        Instances in the reduced training set.

    y_ : ndarray
        Labels in the reduced training set (pass-through; not used for
        selection since this is unsupervised).

    sample_indices_ : ndarray
        Indices of the selected samples in the original dataset.

    reduction_ : float
        Reduction ratio R = (|T| - |S|) / |T|.

    perplexities_ : ndarray of shape (n_samples,)
        Per-instance perplexity values computed during fitting.

    topic_distributions_ : ndarray of shape (n_samples, n_topics)
        Per-instance topic probability distributions from LDA.

    low_threshold_ : float
        The computed perplexity threshold for redundancy removal.

    high_threshold_ : float
        The computed perplexity threshold for noise removal.

    lda_model_ : LatentDirichletAllocation
        The fitted LDA model.

    References
    ==========

    The approach is inspired by the biO-IS framework [1], extending the
    entropy/perplexity rationale to a fully unsupervised setting using
    topic modeling.

    [1] Washington Cunha, Alejandro Moreo, Andrea Esuli, Fabrizio
        Sebastiani, Leonardo Rocha, and Marcos A. Gonçalves. A Noise-
        Oriented and Redundancy-Aware Instance Selection Framework.
        ACM Transactions on Information Systems (TOIS).

    Example
    =======

    >>> import numpy as np
    >>> from src.main.python.iSel.perplexity_is import PerplexityIS
    >>> X = np.random.rand(500, 50)
    >>> y = np.zeros(500)  # labels not used for selection
    >>> selector = PerplexityIS(n_topics=10, low_percentile=15, high_percentile=85)
    >>> selector.fit(X, y)
    >>> idx = selector.sample_indices_
    >>> print(f"Selected {len(idx)} out of {len(y)} instances")
    >>> print(f"Reduction: {selector.reduction_:.2%}")
    """

    def __init__(self,
                 n_topics='auto',
                 low_percentile=10.0,
                 high_percentile=90.0,
                 beta=0.0,
                 theta=0.0,
                 max_iter_lda=30,
                 random_state=0):

        self.n_topics = n_topics
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.beta = beta
        self.theta = theta
        self.max_iter_lda = max_iter_lda
        self.random_state = random_state

        self.sample_indices_ = []


    def _resolve_n_topics(self, n_samples):
        """
        Determine the number of topics when n_topics='auto'.

        :param n_samples: Number of samples in the dataset.
        :return: Number of topics.
        """
        if self.n_topics == 'auto':
            k = int(np.sqrt(n_samples))
            k = max(5, min(k, 100))
            print(f"[PerplexityIS] auto n_topics = {k}")
            return k
        return int(self.n_topics)

    def _ensure_non_negative(self, X):
        """
        Shift features so that the minimum value is 0 (LDA requires
        non-negative input).  Returns a copy.
        
        :param X: Input features.
        :return: Shifted features.
        """
        X_shifted = X.copy()
        col_mins = X_shifted.min(axis=0)
        negative_cols = col_mins < 0

        if not isinstance(X_shifted, scipy.sparse.csr.csr_matrix) and np.any(negative_cols):
            warnings.warn(
                "[PerplexityIS] Input contains negative values. "
                "Shifting features to non-negative range for LDA.",
                UserWarning
            )
            X_shifted[:, negative_cols] -= col_mins[negative_cols]
        return X_shifted

    # ------------------------------------------------------------------
    # Step 1: Fit LDA and compute topic distributions
    # ------------------------------------------------------------------

    def _fit_topic_model(self, X):
        """Fit LDA and return per-instance topic distributions."""
        n_samples = X.shape[0]
        n_topics = self._resolve_n_topics(n_samples)

        X_nn = self._ensure_non_negative(X)

        self.lda_model_ = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=self.max_iter_lda,
            learning_method='batch',
            random_state=self.random_state,
            n_jobs=-1,
        )

        print(f"[PerplexityIS] Fitting LDA with {n_topics} topics on "
              f"{n_samples} instances …")
        # transform returns the normalised topic distribution per instance
        # each instance has t features, where t is the number of topics and
        # val in pos i is the probability of instance belonging to topic i
        topic_dist = self.lda_model_.fit_transform(X_nn)

        self.topic_distributions_ = copy.copy(topic_dist)
        return topic_dist

    # ------------------------------------------------------------------
    # Step 2: Compute per-instance perplexity
    # ------------------------------------------------------------------

    def _compute_perplexities(self, topic_distributions):
        """
        Vectorized per-instance perplexity calculation.

        :param topic_distributions: shape (n_samples, n_topics)
        :return: shape (n_samples,)
        """

        # Avoid log(0)
        p = np.clip(topic_distributions, 1e-12, 1.0)

        # Entropy for each row
        entropy = -(p * np.log(p)).sum(axis=1)

        # Perplexity
        perplexities = np.exp(entropy)

        self.perplexities_ = perplexities.copy()

        print(
            f"[PerplexityIS] Perplexity stats — "
            f"min: {perplexities.min():.4f}, "
            f"median: {np.median(perplexities):.4f}, "
            f"max: {perplexities.max():.4f}"
        )

        return perplexities

    # ------------------------------------------------------------------
    # Step 3a: Identify redundant instances (low perplexity)
    # ------------------------------------------------------------------

    def _identify_redundant(self, perplexities):
        """Identify instances with perplexity below low_percentile.

        Returns indices to remove.
        """
        self.low_threshold_ = np.percentile(perplexities, self.low_percentile)
        redundant_mask = perplexities < self.low_threshold_
        redundant_idx = np.where(redundant_mask)[0]

        print(f"[PerplexityIS] Redundancy threshold (p{self.low_percentile}): "
              f"{self.low_threshold_:.4f} → {len(redundant_idx)} candidates")

        if self.beta < 1.0 and len(redundant_idx) > 0:
            # Partial removal: probability-weighted sampling within band
            # Lower perplexity → higher removal probability (more redundant)
            weights = 1.0 / (perplexities[redundant_idx] + 1e-12)
            weights /= weights.sum()
            n_to_remove = max(0, int(len(redundant_idx) * self.beta))
            print(f"[PerplexityIS] Redundant instances: {len(redundant_idx)}, removing {n_to_remove} with beta={self.beta}")

            rng = np.random.RandomState(self.random_state)
            redundant_idx = rng.choice(
                redundant_idx, size=n_to_remove, replace=False, p=weights
            )

        return redundant_idx

    # ------------------------------------------------------------------
    # Step 3b: Identify noisy instances (high perplexity)
    # ------------------------------------------------------------------

    def _identify_noise(self, perplexities):
        """Identify instances with perplexity above high_percentile.

        Returns indices to remove.
        """
        self.high_threshold_ = np.percentile(perplexities, self.high_percentile)
        noise_mask = perplexities > self.high_threshold_
        noise_idx = np.where(noise_mask)[0]

        print(f"[PerplexityIS] Noise threshold (p{self.high_percentile}): "
              f"{self.high_threshold_:.4f} → {len(noise_idx)} candidates")

        if self.theta < 1.0 and len(noise_idx) > 0:
            # Partial removal: higher perplexity → higher removal probability
            weights = perplexities[noise_idx]
            weights /= weights.sum()
            n_to_remove = max(0, int(len(noise_idx) * self.theta))
            print(f"[PerplexityIS] Noise instances: {len(noise_idx)}, removing {n_to_remove} with theta={self.theta}")

            rng = np.random.RandomState(self.random_state + 1
                                         if self.random_state is not None
                                         else None)
            noise_idx = rng.choice(
                noise_idx, size=n_to_remove, replace=False, p=weights
            )

        return noise_idx

    # ------------------------------------------------------------------
    # Main pipeline (follows InstanceSelectionMixin contract)
    # ------------------------------------------------------------------

    def select_data(self, X, y):
        """Run the full unsupervised instance selection pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances.
        y : array-like of shape (n_samples,)
            Labels (passed through but **not** used for selection).

        Returns
        -------
        X_ : ndarray
            Selected instances.
        y_ : ndarray
            Corresponding labels.
        """
        # Validate dimensions (y is kept for API compatibility but unused)
        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original_y = len(y)
        self.mask = np.ones(y.size, dtype=bool)

        # Step 1 — fit topic model
        topic_distributions = self._fit_topic_model(X)

        # Step 2 — compute per-instance perplexity
        perplexities = self._compute_perplexities(topic_distributions)

        # Step 3a — redundancy removal (low perplexity band)
        idx_redundant = self._identify_redundant(perplexities)
        self._idx_redundant = idx_redundant
        self.mask[idx_redundant] = False

        # Step 3b — noise removal (high perplexity band)
        idx_noise = self._identify_noise(perplexities)
        self._idx_noise = idx_noise
        self.mask[idx_noise] = False

        # Build outputs (same pattern as BIOIS)
        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])

        self.sample_indices_ = np.asarray(range(len(y)))[self.mask]
        self.reduction_ = 1.0 - float(len(self.y_)) / len_original_y

        print(f"[PerplexityIS] Removed {len(idx_redundant)} redundant + "
              f"{len(idx_noise)} noisy = "
              f"{len(idx_redundant) + len(idx_noise)} total instances")
        print(f"[PerplexityIS] Kept {len(self.y_)} / {len_original_y} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_
