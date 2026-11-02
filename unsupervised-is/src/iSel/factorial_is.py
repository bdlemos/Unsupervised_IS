"""
Rate x selection factorial on the ESAE geometry (FactorialIS)
=============================================================

Separates the two things ESAE-IS decides: *how much* to remove (the rate) and
*which* instances to remove (the selection). Each cell of the factorial is one
FactorialIS(rate, selection):

rate
    "entropy"   the ESAE rule r = r_max (1 - H^alpha) on the K = sqrt(N) cells
    float       a fixed rate, the same for every dataset (e.g. 0.20)

selection
    "re"        sublinear cell quotas, removal weighted by 1/RE (ESAE-IS)
    "cells"     same sublinear cell quotas, uniform removal inside each cell
    "global"    uniform removal over the whole set, no cells

Within one (dataset, fold, random_state) every cell shares the tessellation
and the random stream, so cells differ only in the factor being varied.
Equivalences (tested): (entropy, re) == EntropySublinearAEIS,
(entropy, cells) == AblationClusterSublinearRandom,
(entropy, global) == AblationRandomESAERate,
(r, re) == SublinearAEIS(target_reduction=r).

Only the work a cell needs is done (no autoencoder unless selection="re", no
tessellation for a fixed-rate global cell), so per-phase times are comparable.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.utils.validation import check_X_y

from src.iSel import esae_core
from src.iSel.base import InstanceSelectionMixin
from src.iSel.autoencoder_is import AutoencoderIS

SELECTIONS = ("re", "cells", "global")


class FactorialIS(InstanceSelectionMixin):

    def __init__(
        self,
        rate="entropy",
        selection: str = "re",
        r_max: float = 0.50,
        alpha: float = 15.0,
        n_clusters: int = -1,
        gamma: float = 0.5,
        ae_epochs: int = 50,
        random_state: int = 13,
    ) -> None:
        self.rate = rate
        self.selection = selection
        self.r_max = r_max
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.ae_epochs = ae_epochs
        self.random_state = random_state
        self.sample_indices_ = []

    def _check_params(self):
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}, got {self.selection!r}")
        if self.rate != "entropy" and not (isinstance(self.rate, (int, float)) and 0.0 < self.rate < 1.0):
            raise ValueError(f"rate must be 'entropy' or a float in (0, 1), got {self.rate!r}")

    def _reconstruction_errors(self, X, y) -> np.ndarray:
        ae = AutoencoderIS(
            n_epochs=self.ae_epochs,
            low_percentile=0.0,
            high_percentile=100.0,
            beta=0.0,
            theta=0.0,
            random_state=self.random_state,
        )
        ae.fit(X, y)
        return ae.reconstruction_errors_

    def select_data(self, X, y):
        self._check_params()
        X = esae_core.to_dense(X)
        X, y = check_X_y(X, y, accept_sparse=False)
        n_samples = len(y)
        info = {}
        times = {"time_autoencoder": 0.0, "time_tessellation": 0.0, "time_sampling": 0.0}

        scores = None
        if self.selection == "re":
            t0 = time.time()
            scores = self._reconstruction_errors(X, y)
            self.scores_ = scores
            times["time_autoencoder"] = time.time() - t0
            info.update(esae_core.score_summary(scores))

        if self.rate == "entropy" or self.selection != "global":
            t0 = time.time()
            n_clusters = esae_core.resolve_n_clusters(self.n_clusters, n_samples)
            labels, sizes = esae_core.tessellate(X, n_clusters, self.random_state)
            self.labels_ = labels
            self.cluster_sizes_ = sizes
            times["time_tessellation"] = time.time() - t0
            # H is recorded for fixed-rate cells too, as a diagnostic
            H = esae_core.normalized_entropy(sizes)
            self.entropy_ = H
            info.update(n_clusters=n_clusters, entropy=H)

        if self.rate == "entropy":
            target_reduction = esae_core.entropy_rate(self.entropy_, self.r_max, self.alpha)
        else:
            target_reduction = float(self.rate)
        self.target_reduction_ = target_reduction
        target_keep_count = int(n_samples * (1.0 - target_reduction))
        print(f"[FactorialIS] rate={self.rate} selection={self.selection} → "
              f"target reduction {target_reduction:.2%} (keeping {target_keep_count} / {n_samples})")

        t0 = time.time()
        rng = np.random.RandomState(self.random_state)
        if self.selection == "global":
            keep_idx = rng.choice(n_samples, size=target_keep_count, replace=False)
            mask = np.zeros(n_samples, dtype=bool)
            mask[keep_idx] = True
        else:
            lam, quotas = esae_core.sublinear_quotas(self.cluster_sizes_, self.gamma, target_keep_count)
            self.cluster_quotas_ = quotas
            weights = esae_core.removal_weights(scores, "original") if self.selection == "re" else None
            mask = esae_core.remove_within_cells(self.labels_, quotas, rng, weights)
            info.update(sublinear_lambda=lam, **esae_core.cell_summary(self.cluster_sizes_, quotas))
        times["time_sampling"] = time.time() - t0

        self._set_selection(
            X, y, mask,
            rate_rule=self.rate if self.rate == "entropy" else "fixed",
            target_reduction=target_reduction,
            target_keep_count=target_keep_count,
            **info,
            **times,
        )
        print(f"[FactorialIS] Done — kept {len(self.y_)} / {n_samples} (reduction = {self.reduction_:.2%})")
        return self.X_, self.y_
