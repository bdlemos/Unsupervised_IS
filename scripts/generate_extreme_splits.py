#!/usr/bin/env python3
"""
Generates instance selection splits at extreme rates (e.g. 98%, 99%)
for ModernBERT evaluation.

Methods supported:
  - random-rate-<R>
  - typ-cover-rate-<R>
  - diff-cover-rate-<R>
  - sae-clean-rate-<R>

Outputs:
  results/exp-extreme-rates/rate_<R>/instance_selection/selection/<dataset>/
    split_10_<method>-rate-<R>.pkl
    split_10_<method>-rate-<R>_idxinfold.pkl
"""

import os, sys, argparse, pickle
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(REPO_DIR, 'scripts', 'proxy_budget'))

from proxy_lib import load_fold, _quotas, typicality
from diffusion_rank_is import compute_diffusion_weights

BASE_OUT = os.path.join(REPO_DIR, 'results', 'exp-extreme-rates')
DIAG_ROOT = os.path.join(REPO_DIR, 'results', 'diag-aggressive-rates', 'diag')


def get_sae_clean_idx(dataset, fold, n_keep, seed):
    fpath = os.path.join(DIAG_ROOT, dataset, 'sae-rate-40', f'fold_{fold}.parquet')
    if not os.path.exists(fpath):
        fpath = os.path.join(DIAG_ROOT, dataset, 'sae-rate-60', f'fold_{fold}.parquet')
    df = pd.read_parquet(fpath)
    cell_ids = df['cell_id'].values
    scores = df['re_score'].values
    n = len(cell_ids)
    if n_keep >= n:
        return np.arange(n)
        
    K = int(np.max(cell_ids)) + 1
    sizes = np.bincount(cell_ids, minlength=K)
    q = _quotas(sizes, 0.5, n_keep)
    rng = np.random.RandomState(seed)
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
        # Clean mode: lower reconstruction error -> higher probability to keep
        # i.e. sort ascending by error or use weights
        keep.append(idx[np.argsort(c_scores)[:q[c]]])
    return np.sort(np.concatenate(keep))


from joblib import Parallel, delayed


def process_single_fold(dataset, fold, rate, m, orig_splits):
    orig_tr = orig_splits.loc[fold].train_idxs
    orig_te = orig_splits.loc[fold].test_idxs
    n_samples = len(orig_tr)
    n_keep = max(int(round(n_samples * (1 - rate / 100))), 2)
    
    if m == 'random':
        rng = np.random.RandomState(28101994 + fold)
        idx_local = np.sort(rng.choice(n_samples, size=n_keep, replace=False))
    elif m == 'typ-cover':
        X_tr, _, _, _ = load_fold(dataset, fold)
        typ = typicality(X_tr)
        K = int(min(500, n_samples // 2))
        km = MiniBatchKMeans(n_clusters=K, random_state=1000 + fold, n_init=3, batch_size=4096)
        labels = km.fit_predict(X_tr)
        sizes = np.bincount(labels, minlength=K)
        q = _quotas(sizes, 0.5, n_keep)
        keep = []
        for c in range(K):
            c_idx = np.where(labels == c)[0]
            if len(c_idx) == 0 or q[c] == 0:
                continue
            keep.append(c_idx[np.argsort(-typ[c_idx])[:q[c]]])
        idx_local = np.sort(np.concatenate(keep))
    elif m == 'diff-cover':
        X_tr, _, _, _ = load_fold(dataset, fold)
        pi = compute_diffusion_weights(X_tr, k=25)
        K = int(min(500, n_samples // 2))
        km = MiniBatchKMeans(n_clusters=K, random_state=1000 + fold, n_init=3, batch_size=4096)
        labels = km.fit_predict(X_tr)
        sizes = np.bincount(labels, minlength=K)
        q = _quotas(sizes, 0.5, n_keep)
        keep = []
        for c in range(K):
            c_idx = np.where(labels == c)[0]
            if len(c_idx) == 0 or q[c] == 0:
                continue
            keep.append(c_idx[np.argsort(-pi[c_idx])[:q[c]]])
        idx_local = np.sort(np.concatenate(keep))
    elif m == 'sae-clean':
        idx_local = get_sae_clean_idx(dataset, fold, n_keep, seed=2000 + fold)
    else:
        raise ValueError(m)
        
    global_tr = np.array(orig_tr)[idx_local].tolist()
    return fold, global_tr, orig_te, idx_local.tolist()


def generate_splits(dataset, rate=98, methods=None):
    if methods is None:
        methods = ['random', 'typ-cover', 'diff-cover', 'sae-clean']
        
    out_dir = os.path.join(BASE_OUT, f'rate_{rate}', 'instance_selection', 'selection', dataset)
    os.makedirs(out_dir, exist_ok=True)
    
    orig_split_file = os.path.join(REPO_DIR, 'datasets', dataset, 'splits', 'split_10.pkl')
    with open(orig_split_file, 'rb') as f:
        orig_splits = pickle.load(f)
        
    for m in methods:
        m_name = f'{m}-rate-{rate}'
        target_f = os.path.join(out_dir, f'split_10_{m_name}.pkl')
        if os.path.exists(target_f):
            print(f"  -> [CACHED] {target_f} already exists, skipping.")
            continue
        print(f"Generating {dataset} | {m_name} in parallel...")
        
        results = Parallel(n_jobs=10)(
            delayed(process_single_fold)(dataset, fold, rate, m, orig_splits) for fold in range(10)
        )
        # Sort by fold index
        results.sort(key=lambda x: x[0])
        
        train_idxs_global = [r[1] for r in results]
        test_idxs_global = [r[2] for r in results]
        train_idxs_infold = [r[3] for r in results]
            
        # Save split_10.pkl
        df_global = pd.DataFrame({'train_idxs': train_idxs_global, 'test_idxs': test_idxs_global})
        with open(os.path.join(out_dir, f'split_10_{m_name}.pkl'), 'wb') as f:
            pickle.dump(df_global, f)
            
        # Save split_10_idxinfold.pkl
        df_infold = pd.DataFrame({'train_idxs': train_idxs_infold, 'test_idxs': test_idxs_global})
        with open(os.path.join(out_dir, f'split_10_{m_name}_idxinfold.pkl'), 'wb') as f:
            pickle.dump(df_infold, f)
            
        print(f"  -> Saved {out_dir}/split_10_{m_name}.pkl (n_keep={len(train_idxs_global[0])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='books,acm,dblp,20ng,webkb')
    ap.add_argument('--rate', type=int, default=98)
    ap.add_argument('--methods', default='random,typ-cover,diff-cover,sae-clean')
    a = ap.parse_args()
    
    for ds in a.dataset.split(','):
        generate_splits(ds, rate=a.rate, methods=a.methods.split(','))


if __name__ == '__main__':
    main()
