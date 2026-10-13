#!/usr/bin/env python3
"""
Evaluate DiffusionRank-IS on the budget curve across 10 folds for webkb, acm, books, dblp.
Saves to results/proxy-budget-curve/diffusion_budget/
"""

import os, sys, argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from proxy_lib import REPO_DIR, OUT_ROOT, load_fold, proxy_eval, md
from diffusion_rank_is import compute_diffusion_weights, lazy_greedy_facility_location, diffusion_cover

OUT = os.path.join(OUT_ROOT, 'diffusion_budget')
DATASETS = ['20ng', 'acm', 'books', 'dblp', 'sst1', 'webkb']
RATES = [90, 95, 98, 99]


def select_from_clusters(labels, pi, n_keep, gamma=0.5):
    from proxy_lib import _quotas
    K = int(np.max(labels)) + 1
    sizes = np.bincount(labels, minlength=K)
    q = _quotas(sizes, gamma, n_keep)
    keep = []
    for c in range(K):
        idx = np.where(labels == c)[0]
        if len(idx) == 0 or q[c] == 0:
            continue
        keep.append(idx[np.argsort(-pi[idx])[:q[c]]])
    return np.sort(np.concatenate(keep))


def run_fold(ds, fold, rates):
    from sklearn.cluster import MiniBatchKMeans
    X_tr, y_tr, X_te, y_te = load_fold(ds, fold)
    n = len(y_tr)
    
    # 1. Manifold diffusion stationary distribution (k=25, alpha=0.85)
    pi = compute_diffusion_weights(X_tr, k=25)
    
    # 2. Tessellate space once per fold into micro-clusters
    K = int(min(500, n // 2))
    km = MiniBatchKMeans(n_clusters=K, random_state=100 + fold, n_init=3, batch_size=4096)
    labels = km.fit_predict(X_tr)
    
    rows = []
    for rate in rates:
        n_keep = max(int(round(n * (1 - rate / 100))), 2)
        idx = select_from_clusters(labels, pi, n_keep, gamma=0.5)
        if len(np.unique(y_tr[idx])) < 2:
            continue
        r = proxy_eval(X_tr[idx], y_tr[idx], X_te, y_te)
        rows.append(dict(dataset=ds, fold=fold, rate=rate, method='diffusion-cover', n_keep=len(idx),
                         n_classes_kept=len(np.unique(y_tr[idx])),
                         macro=r['macro'], micro=r['micro']))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    res = Parallel(n_jobs=24, verbose=5)(
        delayed(run_fold)(ds, f, RATES) for ds in DATASETS for f in range(10))
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(OUT, 'per_fold.csv'), index=False)
    
    summ = df.groupby(['dataset', 'rate', 'method']).macro.agg(['mean', 'std']).reset_index()
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    
    pv = summ.pivot_table(index=['dataset', 'rate'], columns='method', values='mean')
    lines = ['# Avaliação do Novo Paradigma: Diffusion-Cover na Curva de Orçamento', '']
    lines.append('## Macro-F1 médio\n')
    lines.append(md(pv))
    
    open(os.path.join(OUT, 'REPORT.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()


