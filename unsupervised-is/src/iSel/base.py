"""Base para metodos de redução"""
import warnings
class InstanceReductionWarning(UserWarning):
    pass
warnings.simplefilter("always", InstanceReductionWarning)

import numpy as np
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
import six

class InstanceSelectionBase(six.with_metaclass(ABCMeta, BaseEstimator)):
    @abstractmethod
    def __init__(self):
        pass


class InstanceSelectionMixin(InstanceSelectionBase):

    """Mixin class for all instance reduction techniques"""
    def select_data(self, X, y):
        """
        Procedimento para redução dos dados.
        # Entrada X e y
        # X =
        # y =
        """
        pass

    def fit(self, X, y):
        """
        Call reduce data procedure
        """
        self.X = X
        self.y = y

        self.select_data(X, y)

        return self

    def _set_selection(self, X, y, mask, **info):
        """Store the kept subset and the values that decided it (see selection_record)."""
        self.mask = mask
        self.X_ = np.asarray(X[mask])
        self.y_ = np.asarray(y[mask])
        self.sample_indices_ = np.where(mask)[0]
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)
        self.selection_info_ = {
            "n_samples": int(len(y)),
            "n_kept": int(len(self.y_)),
            "reduction": self.reduction_,
            **info,
        }

    def selection_record(self):
        """Constructor parameters plus the decision values of the last fit."""
        return {
            "params": self.get_params(),
            "selection": getattr(self, "selection_info_", {}),
        }
