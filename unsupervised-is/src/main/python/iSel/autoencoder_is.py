"""
Autoencoder Reconstruction-Error Unsupervised Instance Selection (AE-IS)

An unsupervised instance selection framework that leverages reconstruction
error from a PyTorch autoencoder to simultaneously remove redundant and noisy
instances — without requiring class labels.

Strategy
========

The approach relies on the "triplet of constraints" (reduction, efficacy,
and efficiency) to guide instance selection:

1. **Redundancy identification** — Instances with *low* reconstruction
   error are easily compressible because they follow the dominant pattern
   in the data.  They are considered redundant and can be removed without
   harming generalisation.

2. **Information / noise identification** — Instances with *high*
   reconstruction error are atypical.  Selecting them ensures the model
   learns nuances and edge cases.

3. **Unsupervised refinement** — Since there are no labels to distinguish
   "hard but useful instances" from pure noise (e.g., typos), we remove
   the very bottom (redundant) and the very top (likely noise) of the
   reconstruction-error distribution and densely sample the medium-to-high
   error region.

Implementation
==============

The autoencoder is built in PyTorch with GPU acceleration (CUDA) when
available.  It uses a symmetric encoder–bottleneck–decoder architecture
with dropout and batch normalisation for regularisation.
"""

import numpy as np
import copy
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
import scipy

from src.main.python.iSel.base import InstanceSelectionMixin


# ======================================================================
# PyTorch Autoencoder Module
# ======================================================================

class _Autoencoder(nn.Module):
    """Symmetric bottleneck autoencoder.

    Architecture:
        input → encoder_dim → bottleneck → encoder_dim → input
    With BatchNorm + ReLU + Dropout between layers.
    """

    def __init__(self, n_features, bottleneck, encoder_dim, dropout=0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(encoder_dim, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(encoder_dim, n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# ======================================================================
# Instance Selection Class
# ======================================================================

class AutoencoderIS(InstanceSelectionMixin):
    """Autoencoder Reconstruction-Error Instance Selection (AE-IS)

    Description
    ===========

    A fully unsupervised instance selection method that uses reconstruction
    error from a bottleneck autoencoder to identify and remove both
    redundant and noisy instances.

    The autoencoder is trained to compress the input into a lower-
    dimensional bottleneck and then reconstruct the original features.
    The per-instance reconstruction error (MSE) characterises each
    instance:

    * **Low error** → the instance sits in a dense, well-represented
      region of feature space and is likely *redundant*.
    * **Medium-high error** → the instance is informative, capturing
      edge cases and boundary patterns — helpful for generalisation.
    * **Very high error** → the instance is inconsistent with the
      learned manifold and is likely *noise* or an error.

    The reconstruction error for instance *x* is:

        RE(x) = || x - AE(x) ||²  /  d

    where AE(x) is the autoencoder output and d is the number of features.

    The method works in three steps:
        1. Fit a bottleneck autoencoder to learn the data manifold.
        2. Compute per-instance reconstruction error.
        3. Remove instances falling below the ``low_percentile`` threshold
           (redundant) or above the ``high_percentile`` threshold (noisy),
           keeping the informative middle band.

    Parameters
    ==========

    bottleneck_ratio : float, default=0.5
        Ratio of the bottleneck dimension to the input dimension.
        The bottleneck size is ``max(2, int(n_features * bottleneck_ratio))``.

    low_percentile : float, default=10.0
        Percentile below which instances are considered redundant and
        removed (low reconstruction error → too easy to reconstruct).

    high_percentile : float, default=95.0
        Percentile above which instances are considered noise and removed
        (high reconstruction error → too surprising / inconsistent).

    beta : float, default=0.0
        Additional partial redundancy removal rate applied *within* the
        low error band.  When 1.0, all instances below ``low_percentile``
        are removed.  Values in (0, 1) allow partial removal via
        probability-weighted sampling.

    theta : float, default=0.0
        Additional partial noise removal rate applied *within* the high
        error band.  When 1.0, all instances above ``high_percentile``
        are removed.  Values in (0, 1) allow partial removal via
        probability-weighted sampling.

    n_epochs : int, default=50
        Maximum number of training epochs for the autoencoder.

    batch_size : int, default=256
        Mini-batch size for training.

    lr : float, default=1e-3
        Learning rate for the Adam optimiser.

    patience : int, default=10
        Number of epochs with no improvement before early stopping.

    dropout : float, default=0.1
        Dropout rate in encoder and decoder layers.

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

    reconstruction_errors_ : ndarray of shape (n_samples,)
        Per-instance MSE reconstruction error computed during fitting.

    low_threshold_ : float
        The computed error threshold for redundancy removal.

    high_threshold_ : float
        The computed error threshold for noise removal.

    autoencoder_ : _Autoencoder (nn.Module)
        The fitted autoencoder model.

    scaler_ : StandardScaler
        The fitted scaler used to normalise data before autoencoding.

    References
    ==========

    The approach is inspired by the biO-IS framework [1], extending the
    reconstruction-error rationale to a fully unsupervised setting using
    autoencoder neural networks.

    [1] Washington Cunha, Alejandro Moreo, Andrea Esuli, Fabrizio
        Sebastiani, Leonardo Rocha, and Marcos A. Gonçalves. A Noise-
        Oriented and Redundancy-Aware Instance Selection Framework.
        ACM Transactions on Information Systems (TOIS).

    Example
    =======

    >>> import numpy as np
    >>> from src.main.python.iSel.autoencoder_is import AutoencoderIS
    >>> X = np.random.rand(500, 50)
    >>> y = np.zeros(500)  # labels not used for selection
    >>> selector = AutoencoderIS(bottleneck_ratio=0.3,
    ...                          low_percentile=15,
    ...                          high_percentile=90)
    >>> selector.fit(X, y)
    >>> idx = selector.sample_indices_
    >>> print(f"Selected {len(idx)} out of {len(y)} instances")
    >>> print(f"Reduction: {selector.reduction_:.2%}")
    """

    def __init__(self,
                 bottleneck_ratio=0.5,
                 low_percentile=25.0,
                 high_percentile=75.0,
                 beta=0.10,
                 theta=0.10,
                 n_epochs=50,
                 batch_size=256,
                 lr=1e-3,
                 patience=10,
                 dropout=0.1,
                 random_state=0):
        """
        Initialize the AE-IS instance selector with specified parameters.
        
        :param bottleneck_ratio: Ratio of bottleneck dimension to input features.
        :param low_percentile: Percentile for redundancy threshold.
        :param high_percentile: Percentile for noise threshold.
        :param beta: Partial redundancy removal rate within low error band. 0 means no additional removal, 1 means remove all below low_percentile.
        :param theta: Partial noise removal rate within high error band. 0 means no additional removal, 1 means remove all above high_percentile.
        :param n_epochs: Maximum number of training epochs for the autoencoder.
        :param batch_size: Mini-batch size for training.
        :param lr: Learning rate for the Adam optimiser.
        :param patience: Number of epochs with no improvement before early stopping.
        :param dropout: Dropout rate in encoder and decoder layers.
        :param random_state: Random state for reproducibility.
        """

        self.bottleneck_ratio = bottleneck_ratio
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.beta = beta
        self.theta = theta
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.dropout = dropout
        self.random_state = random_state

        self.sample_indices_ = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_device(self):
        """Select CUDA if available, otherwise CPU."""
        if torch.cuda.is_available():
            dev = torch.device('cuda')
            print(f"[AE-IS] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            dev = torch.device('cpu')
            print("[AE-IS] Using CPU")
        return dev

    def _build_architecture(self, n_features):
        """Compute the symmetric hidden-layer topology.

        :param n_features: Number of input features.
        :return: (bottleneck, encoder_dim)
        """
        bottleneck = max(2, int(n_features * self.bottleneck_ratio))
        # bottleneck = 10
        encoder_dim = max(bottleneck + 1, int((n_features + bottleneck) / 2))
        # encoder_dim = 20
        print(f"[AE-IS] Architecture: {n_features} → {encoder_dim} → "
              f"{bottleneck} → {encoder_dim} → {n_features}")
        return bottleneck, encoder_dim

    def _to_dense(self, X):
        """Convert sparse matrix to dense if necessary."""
        if scipy.sparse.issparse(X):
            return X.toarray()
        return np.asarray(X, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 1: Fit autoencoder (PyTorch)
    # ------------------------------------------------------------------

    def _fit_autoencoder(self, X):
        """Scale data, fit the bottleneck autoencoder on GPU/CPU,
        and return per-instance reconstruction errors.

        :param X: Input features (n_samples, n_features).
        :return: Per-instance MSE array of shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # Convert to dense and scale
        X_dense = self._to_dense(X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_dense).astype(np.float32)

        # Setup device and reproducibility
        device = self._get_device()
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        # Build model
        bottleneck, encoder_dim = self._build_architecture(n_features)
        self.autoencoder_ = _Autoencoder(
            n_features, bottleneck, encoder_dim, self.dropout
        ).to(device)

        # DataLoader
        X_tensor = torch.from_numpy(X_scaled)
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, drop_last=False)

        # Optimiser & loss
        optimiser = torch.optim.Adam(self.autoencoder_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Training loop with early stopping
        print(f"[AE-IS] Training autoencoder on {n_samples} instances "
              f"({n_features} features) for up to {self.n_epochs} epochs …")

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, self.n_epochs + 1):
            self.autoencoder_.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, _ in loader:
                x_batch = x_batch.to(device)
                x_hat = self.autoencoder_(x_batch)
                loss = criterion(x_hat, x_batch)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches

            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                best_state = copy.deepcopy(self.autoencoder_.state_dict())
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.n_epochs} — "
                      f"loss: {avg_loss:.6f} (best: {best_loss:.6f})")

            if epochs_no_improve >= self.patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs)")
                break

        # Restore best weights
        self.autoencoder_.load_state_dict(best_state)
        print(f"[AE-IS] Training finished — best loss: {best_loss:.6f}")

        # Compute per-instance reconstruction errors on full dataset
        self.autoencoder_.eval()
        all_errors = []
        eval_loader = DataLoader(
            TensorDataset(X_tensor, X_tensor),
            batch_size=self.batch_size, shuffle=False
        )

        with torch.no_grad():
            for x_batch, _ in eval_loader:
                x_batch = x_batch.to(device)
                x_hat = self.autoencoder_(x_batch)
                # Per-instance MSE (mean over features)
                batch_errors = ((x_batch - x_hat) ** 2).mean(dim=1)
                all_errors.append(batch_errors.cpu().numpy())

        errors = np.concatenate(all_errors)
        return errors

    # ------------------------------------------------------------------
    # Step 2: Compute per-instance reconstruction error
    # ------------------------------------------------------------------

    def _compute_reconstruction_errors(self, errors):
        """Store and report reconstruction error statistics.

        :param errors: Per-instance MSE array of shape (n_samples,).
        :return: The same errors array.
        """
        self.reconstruction_errors_ = errors.copy()

        print(
            f"[AE-IS] Reconstruction error stats — "
            f"min: {errors.min():.6f}, "
            f"median: {np.median(errors):.6f}, "
            f"max: {errors.max():.6f}"
        )

        return errors

    # ------------------------------------------------------------------
    # Step 3a: Identify redundant instances (low reconstruction error)
    # ------------------------------------------------------------------

    def _identify_redundant(self, errors):
        """Identify instances with reconstruction error below
        low_percentile.

        Returns indices to remove.
        """
        self.low_threshold_ = np.percentile(errors, self.low_percentile)
        redundant_mask = errors < self.low_threshold_
        redundant_idx = np.where(redundant_mask)[0]

        # print(f"[AE-IS] Redundancy threshold (p{self.low_percentile}): "
        #       f"{self.low_threshold_:.6f} → {len(redundant_idx)} candidates")

        if self.beta < 1.0 and len(redundant_idx) > 0:
            # Partial removal: lower error → higher removal probability
            # (more redundant)
            weights = 1.0 / (errors[redundant_idx] + 1e-12)
            weights /= weights.sum()
            n_to_remove = max(0, int(len(redundant_idx) * self.beta))
            print(f"[AE-IS] Redundant instances: {len(redundant_idx)}, removing {n_to_remove} with beta={self.beta}")

            rng = np.random.RandomState(self.random_state)
            redundant_idx = rng.choice(
                redundant_idx, size=n_to_remove, replace=False, p=weights
            )

        return redundant_idx

    # ------------------------------------------------------------------
    # Step 3b: Identify noisy instances (high reconstruction error)
    # ------------------------------------------------------------------

    def _identify_noise(self, errors):
        """Identify instances with reconstruction error above
        high_percentile.

        Returns indices to remove.
        """
        self.high_threshold_ = np.percentile(errors, self.high_percentile)
        noise_mask = errors > self.high_threshold_
        noise_idx = np.where(noise_mask)[0]

        # print(f"[AE-IS] Noise threshold (p{self.high_percentile}): "
        #       f"{self.high_threshold_:.6f} → {len(noise_idx)} candidates")

        if self.theta < 1.0 and len(noise_idx) > 0:
            # Partial removal: higher error → higher removal probability
            weights = errors[noise_idx]
            weights /= weights.sum()
            n_to_remove = max(0, int(len(noise_idx) * self.theta))
            print(f"[AE-IS] Noise instances: {len(noise_idx)}, removing {n_to_remove} with theta={self.theta}")

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

        # Step 1 — fit autoencoder and get raw errors
        errors = self._fit_autoencoder(X)

        # Step 2 — store and report error statistics
        errors = self._compute_reconstruction_errors(errors)

        # Step 3a — redundancy removal (low error band)
        idx_redundant = self._identify_redundant(errors)
        self._idx_redundant = idx_redundant
        self.mask[idx_redundant] = False

        # Step 3b — noise removal (high error band)
        idx_noise = self._identify_noise(errors)
        self._idx_noise = idx_noise
        self.mask[idx_noise] = False

        # Build outputs (same pattern as BIOIS / PerplexityIS)
        self.X_ = np.asarray(X[self.mask])
        self.y_ = np.asarray(y[self.mask])

        self.sample_indices_ = np.asarray(range(len(y)))[self.mask]
        self.reduction_ = 1.0 - float(len(self.y_)) / len_original_y

        print(f"[AE-IS] Removed {len(idx_redundant)} redundant + "
              f"{len(idx_noise)} noisy = "
              f"{len(idx_redundant) + len(idx_noise)} total instances")
        print(f"[AE-IS] Kept {len(self.y_)} / {len_original_y} "
              f"(reduction = {self.reduction_:.2%})")

        return self.X_, self.y_
