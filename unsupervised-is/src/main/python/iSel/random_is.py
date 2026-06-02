"""
Random instance selection baseline

Selects a random subset of training indices according to `selection_rate`.
"""
from src.main.python.iSel.base import InstanceSelectionMixin
import numpy as np
import random
from sklearn.utils.validation import check_X_y


class RandomIS(InstanceSelectionMixin):
    def __init__(self, selection_rate=0.5, random_state=None):
        self.selection_rate = float(selection_rate)
        self.random_state = random_state
        self.sample_indices_ = []

    def select_data(self, X, y):
        X, y = check_X_y(X, y, accept_sparse="csr")

        n = len(y)
        k = int(round(self.selection_rate * n))
        if k <= 0:
            k = 1
        if k >= n:
            # select all
            idxs = list(range(n))
        else:
            rnd = random.Random(self.random_state)
            idxs = rnd.sample(list(range(n)), k)

        self.X_ = np.asarray(X[idxs])
        self.y_ = np.asarray(y[idxs])
        self.sample_indices_ = list(sorted(idxs))
        self.reduction_ = 1.0 - float(len(self.y_)) / n
        return self.X_, self.y_
