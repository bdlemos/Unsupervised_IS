"""
No instance selection baseline

Selects all training instances (no selection).
"""
from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
from sklearn.utils.validation import check_X_y


class NoIS(InstanceSelectionMixin):
    def __init__(self):
        super().__init__()
        self.sample_indices_ = []

    def select_data(self, X, y):
        X, y = check_X_y(X, y, accept_sparse="csr")

        n = len(y)
        idxs = list(range(n))

        self.X_ = np.asarray(X[idxs])
        self.y_ = np.asarray(y[idxs])
        self.sample_indices_ = list(sorted(idxs))
        self.reduction_ = 0.0
        return self.X_, self.y_

