"""
Shared helpers for the proxy budget-curve study.

Everything here is CPU-only and independent from the main pipeline.
Results of this study live exclusively under results/proxy-budget-curve/.

Selection methods (all unsupervised: labels are never used for selection):
  - random          : uniform sampling without replacement (same as RandomIS).
  - cluster-random  : k-means coverage + sublinear quota per cluster,
                      random choice inside each cluster (ablation: coverage only).
  - typ-cover       : k-means coverage + sublinear quota per cluster,
                      keep the most *typical* points inside each cluster
                      (typicality = inverse mean distance to k nearest neighbours,
                      TypiClust-style).
"""

from __future__ import annotations

import os
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASETS_DIR = os.path.join(REPO_DIR, 'datasets')
OUT_ROOT = os.path.join(REPO_DIR, 'results', 'proxy-budget-curve')
INPUT_REP = 'jina-v5'


def md(df, floatfmt: str = '.4f', index: bool = True) -> str:
    """Render a DataFrame as a markdown table (no `tabulate` dependency)."""
    d = df.reset_index() if index else df
    fmt = lambda v: format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    head = '| ' + ' | '.join(map(str, d.columns)) + ' |'
    sep = '| ' + ' | '.join('---' for _ in d.columns) + ' |'
    body = ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in d.itertuples(index=False)]
    return '\n'.join([head, sep] + body)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_fold(dataset: str, fold: int):
    """Load train/test jina-v5 embeddings for a fold (rows aligned with idxinfold)."""
    d = os.path.join(DATASETS_DIR, dataset, INPUT_REP)
    X_tr, y_tr = load_svmlight_file(os.path.join(d, f'train{fold}.gz'))
    X_te, y_te = load_svmlight_file(os.path.join(d, f'test{fold}.gz'), n_features=X_tr.shape[1])
    X_tr = normalize(np.asarray(X_tr.todense(), dtype=np.float32))
    X_te = normalize(np.asarray(X_te.todense(), dtype=np.float32))
    return X_tr, y_tr.astype(int), X_te, y_te.astype(int)


# ── Proxy classifier ─────────────────────────────────────────────────────────
def proxy_eval(X_tr, y_tr, X_te, y_te, C: float = 10.0):
    """Cheap proxy: multinomial logistic regression on frozen embeddings."""
    clf = LogisticRegression(C=C, max_iter=200, solver='lbfgs')
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    labels = np.unique(y_te)
    return {
        'macro': f1_score(y_te, pred, average='macro', labels=labels, zero_division=0),
        'micro': f1_score(y_te, pred, average='micro'),
        'per_class': f1_score(y_te, pred, average=None, labels=labels, zero_division=0).tolist(),
        'labels': labels.tolist(),
    }


# ── Selection helpers ─────────────────────────────────────────────────────────
def _find_lambda(sizes, gamma, target):
    lo, hi = 0.0, float(target)
    best = hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        kept = np.minimum(sizes, np.ceil(mid * sizes ** gamma)).sum()
        if kept >= target:
            best, hi = mid, mid
        else:
            lo = mid
    return best


def _quotas(sizes, gamma, n_keep):
    if n_keep <= 0:
        return np.zeros_like(sizes, dtype=int)
    lam = _find_lambda(sizes.astype(float), gamma, n_keep)
    q = np.minimum(sizes, np.ceil(lam * sizes ** gamma)).astype(int)
    diff = int(q.sum() - n_keep)
    if diff > 0:
        order = np.argsort(-q)
        for c in order:
            if diff == 0:
                break
            if q[c] > 1:
                take = min(diff, q[c] - 1)
                q[c] -= take
                diff -= take
        if diff > 0:
            for c in order:
                if diff == 0:
                    break
                if q[c] > 0:
                    q[c] -= 1
                    diff -= 1
    elif diff < 0:
        room = sizes - q
        order = np.argsort(-room)
        for c in order:
            if diff == 0:
                break
            if room[c] > 0:
                q[c] += 1
                diff += 1
    return q


def typicality(X, k: int = 20):
    """Inverse mean cosine distance to k nearest neighbours (global)."""
    k = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute', n_jobs=8).fit(X)
    dist, _ = nn.kneighbors(X)
    return 1.0 / (dist[:, 1:].mean(axis=1) + 1e-8)


def select(method: str, X, n_keep: int, seed: int, typ=None,
           gamma: float = 0.5, k_max: int = 1000):
    n = len(X)
    rng = np.random.RandomState(seed)
    if n_keep >= n:
        return np.arange(n)

    if method == 'random':
        return np.sort(rng.choice(n, size=n_keep, replace=False))

    # coverage-based methods
    K = int(min(n_keep, k_max, n // 2))
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=3, batch_size=4096)
    labels = km.fit_predict(X)
    sizes = np.bincount(labels, minlength=K)
    q = _quotas(sizes, gamma, n_keep)

    keep = []
    for c in range(K):
        idx = np.where(labels == c)[0]
        if len(idx) == 0 or q[c] == 0:
            continue
        if method == 'cluster-random':
            keep.append(rng.choice(idx, size=q[c], replace=False))
        elif method == 'typ-cover':
            keep.append(idx[np.argsort(-typ[idx])[:q[c]]])
        else:
            raise ValueError(method)
    return np.sort(np.concatenate(keep))
