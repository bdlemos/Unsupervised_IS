"""Deterministic synthetic data and a fake autoencoder score shared by the tests."""
import numpy as np


def make_blobs(n, d, seed):
    """Imbalanced gaussian blobs with 3 labels."""
    rng = np.random.RandomState(seed)
    centers = rng.normal(0, 5, size=(8, d))
    weights = rng.dirichlet(np.ones(8) * 0.7)
    comp = rng.choice(8, size=n, p=weights)
    X = centers[comp] + rng.normal(0, 1 + comp[:, None] * 0.2, size=(n, d))
    return X, comp % 3


def fake_reconstruction_errors(X):
    """Stand-in for the autoencoder: squared distance to the global mean."""
    X = np.asarray(X, dtype=np.float64)
    return ((X - X.mean(0)) ** 2).mean(1) + 1e-3


def fake_ae_fit(self, X, y):
    """Patch for AutoencoderIS.fit. The real AE seeds the global numpy RNG and never consumes it."""
    np.random.seed(self.random_state)
    self.reconstruction_errors_ = fake_reconstruction_errors(X)
    return self
