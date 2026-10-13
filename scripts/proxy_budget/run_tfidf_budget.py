#!/usr/bin/env python3
"""
Budget curve benchmark on sparse TF-IDF feature representations.
Evaluates:
  - Random
  - Cluster-Random (gamma=0.5)
  - Typ-Cover (gamma=0.5)
  - Diffusion-Cover (gamma=0.5)

Across 6 datasets: 20ng, acm, books, dblp, sst1, webkb
Across budget rates: 50, 80, 90, 95, 98, 99
Saves to results/proxy-budget-curve/tfidf_budget/
"""

import os, sys, argparse, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from joblib import Parallel, delayed

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

from proxy_lib import _quotas, md

OUT_DIR = os.path.join(REPO_DIR, 'results', 'proxy-budget-curve', 'tfidf_budget')
DATASETS = ['20ng', 'acm', 'books', 'dblp', 'sst1', 'webkb']
RATES = [50, 80, 90, 95, 98, 99]


def load_tfidf_fold(dataset, fold):
    d = os.path.join(REPO_DIR, 'datasets', dataset, 'tfidf')
    X_tr, y_tr = load_svmlight_file(os.path.join(d, f'train{fold}.gz'))
    X_te, y_te = load_svmlight_file(os.path.join(d, f'test{fold}.gz'), n_features=X_tr.shape[1])
    X_tr.indices = X_tr.indices.astype(np.int32)
    X_tr.indptr = X_tr.indptr.astype(np.int32)
    X_te.indices = X_te.indices.astype(np.int32)
    X_te.indptr = X_te.indptr.astype(np.int32)
    X_tr = normalize(X_tr, norm='l2', copy=False)
    X_te = normalize(X_te, norm='l2', copy=False)
    return X_tr, y_tr.astype(int), X_te, y_te.astype(int)


def proxy_eval_tfidf(X_tr_sub, y_tr_sub, X_te, y_te, max_iter=100):
    clf = LogisticRegression(C=1.0, max_iter=max_iter, solver='lbfgs')
    clf.fit(X_tr_sub, y_tr_sub)
    pred = clf.predict(X_te)
    labels = np.unique(y_te)
    return {
        'macro': f1_score(y_te, pred, average='macro', labels=labels, zero_division=0),
        'micro': f1_score(y_te, pred, average='micro')
    }


def compute_geometry_tfidf(X, k=25, alpha=0.85):
    n = X.shape[0]
    k = min(k, n - 1)
    
    # 1. Cosine k-NN on sparse CSR
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute', n_jobs=2).fit(X)
    dist, indices = nn.kneighbors(X)
    
    # Typicality: inverse mean cosine distance
    typ = 1.0 / (dist[:, 1:].mean(axis=1) + 1e-8)
    
    # Adaptive local bandwidth sigma_i
    sigmas = np.median(dist[:, 1:], axis=1, keepdims=True) + 1e-8
    weights = np.exp(- (dist[:, 1:] ** 2) / (sigmas ** 2))
    row_sums = weights.sum(axis=1, keepdims=True) + 1e-12
    norm_weights = weights / row_sums
    
    # Sparse Transition Matrix
    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].flatten()
    data = norm_weights.flatten()
    P = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Power iteration for PageRank diffusion
    pi = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(30):
        pi_next = alpha * P.T.dot(pi) + (1.0 - alpha) * teleport
        if np.linalg.norm(pi_next - pi, 1) < 1e-6:
            break
        pi = pi_next
        
    return typ, pi


def run_tfidf_fold(dataset, fold, rates):
    X_tr, y_tr, X_te, y_te = load_tfidf_fold(dataset, fold)
    n = X_tr.shape[0]
    
    # 1. Geometry: typicality + diffusion stationary distribution
    typ, pi = compute_geometry_tfidf(X_tr, k=25)
    
    # 2. Spatial tessellation into micro-clusters (once per fold)
    K = int(min(500, n // 2))
    km = MiniBatchKMeans(n_clusters=K, random_state=1000 + fold, n_init=3, batch_size=4096)
    labels = km.fit_predict(X_tr)
    sizes = np.bincount(labels, minlength=K)
    
    rows = []
    for rate in rates:
        n_keep = max(int(round(n * (1 - rate / 100))), 2)
        q = _quotas(sizes, 0.5, n_keep)
        rng = np.random.RandomState(2000 + fold * 100 + rate)
        
        # Method 1: Random
        idx_rand = np.sort(rng.choice(n, size=n_keep, replace=False))
        
        # Methods 2, 3, 4: Cluster-based
        clust_rand_keep = []
        typ_cover_keep = []
        diff_cover_keep = []
        
        for c in range(K):
            c_idx = np.where(labels == c)[0]
            s_c = len(c_idx)
            if s_c == 0 or q[c] == 0:
                continue
            if q[c] >= s_c:
                clust_rand_keep.append(c_idx)
                typ_cover_keep.append(c_idx)
                diff_cover_keep.append(c_idx)
                continue
                
            # Cluster-Random
            clust_rand_keep.append(rng.choice(c_idx, size=q[c], replace=False))
            # Typ-Cover (typicality density)
            typ_cover_keep.append(c_idx[np.argsort(-typ[c_idx])[:q[c]]])
            # Diffusion-Cover (PageRank manifold hubs)
            diff_cover_keep.append(c_idx[np.argsort(-pi[c_idx])[:q[c]]])
            
        methods_dict = {
            'random': idx_rand,
            'cluster-random-g0.5': np.sort(np.concatenate(clust_rand_keep)),
            'typ-cover-g0.5': np.sort(np.concatenate(typ_cover_keep)),
            'diffusion-cover': np.sort(np.concatenate(diff_cover_keep)),
        }
        
        for m_name, idx_sel in methods_dict.items():
            if len(np.unique(y_tr[idx_sel])) < 2:
                continue
            res = proxy_eval_tfidf(X_tr[idx_sel], y_tr[idx_sel], X_te, y_te)
            rows.append({
                'dataset': dataset,
                'fold': fold,
                'rate': rate,
                'method': m_name,
                'n_keep': len(idx_sel),
                'n_classes_kept': len(np.unique(y_tr[idx_sel])),
                'macro': res['macro'],
                'micro': res['micro']
            })
            
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-jobs', type=int, default=20)
    a = ap.parse_args()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Starting TF-IDF Budget Curve Benchmark across {DATASETS} on rates {RATES}...")
    
    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(run_tfidf_fold)(ds, f, RATES) for ds in DATASETS for f in range(10)
    )
    print(f"All folds completed in {time.time()-t0:.2f}s!")
    
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(OUT_DIR, 'per_fold.csv'), index=False)
    
    summ = df.groupby(['dataset', 'rate', 'method']).macro.agg(['mean', 'std']).reset_index()
    summ.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
    
    pv = summ.pivot_table(index=['dataset', 'rate'], columns='method', values='mean')
    lines = [
        '# Curva de Orçamento em Representação TF-IDF',
        '',
        'Comparativo entre Random, Cluster-Random, Typ-Cover e Diffusion-Cover utilizando embeddings esparsos TF-IDF.',
        '',
        '## Macro-F1 Médio (Proxy: Regressão Logística em TF-IDF)',
        '',
        md(pv),
        '',
        '## Testes de Significância Estatística vs Random (p-valor, t-pareado)',
        ''
    ]
    
    from scipy import stats
    rand_df = df[df.method == 'random'].copy()
    test_methods = [m for m in df.method.unique() if m != 'random']
    stat_rows = []
    for m in test_methods:
        m_df = df[df.method == m].copy()
        merged = m_df.merge(rand_df, on=['dataset', 'fold', 'rate'], suffixes=(f'_{m}', '_rand'))
        for (ds, rate), g in merged.groupby(['dataset', 'rate']):
            diff = g[f'macro_{m}'] - g['macro_rand']
            if len(diff) >= 5 and diff.std() > 1e-8:
                pval = stats.ttest_rel(g[f'macro_{m}'], g['macro_rand']).pvalue
            else:
                pval = np.nan
            stat_rows.append({
                'dataset': ds,
                'rate': rate,
                'method': m,
                'mean_diff': diff.mean(),
                'p_value': pval,
                'sig_0.05': 'SIM (*)' if pval < 0.05 and diff.mean() > 0 else ('PIOR (*)' if pval < 0.05 and diff.mean() < 0 else 'não')
            })
            
    stat_df = pd.DataFrame(stat_rows)
    lines.append(md(stat_df.sort_values(['dataset', 'rate', 'method']), index=False))
    
    report_path = os.path.join(OUT_DIR, 'REPORT_TFIDF.md')
    open(report_path, 'w').write('\n'.join(lines) + '\n')
    print(f"Report written to {report_path}")
    print('\n'.join(lines[:45]))


if __name__ == '__main__':
    main()
