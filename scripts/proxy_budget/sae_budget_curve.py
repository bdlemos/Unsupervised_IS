#!/usr/bin/env python3
"""
Step C — Budget curve for SAE-IS (Original and Clean variants) with cheap proxy.

Compares:
  - sae-orig (the original mode: inverse scores p ~ (1/s)^beta)
  - sae-clean (the cleaning mode: direct scores p ~ s^beta)

Evaluates on the 3 datasets that have full pre-computed AE diagnostic scores:
  webkb, books, dblp (all 10 folds, rates 50, 80, 90, 95, 98, 99)

Output: results/proxy-budget-curve/sae_budget/
"""
import os, sys, argparse, glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from proxy_lib import REPO_DIR, OUT_ROOT, load_fold, proxy_eval, _quotas, md

DIAG_ROOT = os.path.join(REPO_DIR, 'results', 'diag-aggressive-rates', 'diag')
OUT = os.path.join(OUT_ROOT, 'sae_budget')
DATASETS = ['webkb', 'books', 'dblp', '20ng', 'acm']
RATES = [50, 80, 90, 95, 98, 99]


def load_diag_fold(dataset, fold):
    """Load cell_id and re_score from precomputed diag parquet."""
    fpath = os.path.join(DIAG_ROOT, dataset, 'sae-rate-40', f'fold_{fold}.parquet')
    if not os.path.exists(fpath):
        fpath = os.path.join(DIAG_ROOT, dataset, 'sae-rate-60', f'fold_{fold}.parquet')
    df = pd.read_parquet(fpath)
    return df['cell_id'].values, df['re_score'].values


def select_sae(cell_ids, scores, n_keep, mode, seed, gamma=0.5):
    n = len(cell_ids)
    rng = np.random.RandomState(seed)
    if n_keep >= n:
        return np.arange(n)

    K = int(np.max(cell_ids)) + 1
    sizes = np.bincount(cell_ids, minlength=K)
    q = _quotas(sizes, gamma, n_keep)

    keep = []
    for c in range(K):
        idx = np.where(cell_ids == c)[0]
        s_c = len(idx)
        if s_c == 0 or q[c] == 0:
            continue
        if q[c] >= s_c:
            keep.append(idx)
            continue

        c_scores = scores[idx]
        if mode == 'clean':
            weights = (c_scores + 1e-8)
        else:
            weights = 1.0 / (c_scores + 1e-8)
        
        p = weights / np.sum(weights)
        # In SAE-IS, instances are removed probabilistically with p_remove = weights / sum(weights)
        # so keeping q[c] corresponds to sampling keep_count without replacement using p_keep or removing (s_c - q[c])
        r_c = s_c - q[c]
        to_remove = rng.choice(s_c, size=r_c, replace=False, p=p)
        mask_local = np.ones(s_c, dtype=bool)
        mask_local[to_remove] = False
        keep.append(idx[mask_local])

    return np.sort(np.concatenate(keep))


def run_fold(ds, fold, rates):
    X_tr, y_tr, X_te, y_te = load_fold(ds, fold)
    cell_ids, scores = load_diag_fold(ds, fold)
    n = len(y_tr)
    rows = []
    for rate in rates:
        n_keep = max(int(round(n * (1 - rate / 100))), 2)
        for mode in ['sae-orig', 'sae-clean']:
            idx = select_sae(cell_ids, scores, n_keep, mode=mode.split('-')[1], seed=2000 + fold)
            if len(np.unique(y_tr[idx])) < 2:
                continue
            r = proxy_eval(X_tr[idx], y_tr[idx], X_te, y_te)
            rows.append(dict(dataset=ds, fold=fold, rate=rate, method=mode, n_keep=len(idx),
                             n_classes_kept=len(np.unique(y_tr[idx])),
                             macro=r['macro'], micro=r['micro']))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-jobs', type=int, default=15)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(run_fold)(ds, f, RATES) for ds in DATASETS for f in range(10))
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(OUT, 'per_fold.csv'), index=False)

    # Load existing budget curve results for random and typ-cover for direct comparison
    base_file = os.path.join(OUT_ROOT, 'budget_curve', 'per_fold.csv')
    if os.path.exists(base_file):
        base_df = pd.read_csv(base_file)
        base_sub = base_df[base_df.dataset.isin(DATASETS) & base_df.method.isin(['random', 'typ-cover-g0.5'])].copy()
        combined = pd.concat([base_sub, df], ignore_index=True)
    else:
        combined = df

    summ = combined.groupby(['dataset', 'rate', 'method']).macro.agg(['mean', 'std']).reset_index()
    summ.to_csv(os.path.join(OUT, 'summary_comparison.csv'), index=False)

    pv = summ.pivot_table(index=['dataset', 'rate'], columns='method', values='mean')
    lines = ['# Comparativo SAE-IS (Original vs Clean) vs Random e Typ-Cover na Curva de Orçamento', '']
    lines.append('## Macro-F1 médio (Proxy: Regressão Logística)\n')
    lines.append(md(pv))

    open(os.path.join(OUT, 'REPORT.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
