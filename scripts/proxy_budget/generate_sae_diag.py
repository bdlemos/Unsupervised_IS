#!/usr/bin/env python3
"""
Computes Autoencoder Reconstruction Errors and K-Means cell IDs for 20ng and acm.
Saves diagnostic parquets to results/diag-aggressive-rates/diag/<dataset>/sae-rate-40/fold_<f>.parquet
Strictly CPU-only (CUDA_VISIBLE_DEVICES="").
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans

os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(4)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UNSUP_DIR = os.path.join(REPO_DIR, 'unsupervised-is')
if UNSUP_DIR not in sys.path:
    sys.path.insert(0, UNSUP_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from src.main.python.iSel.autoencoder_is import AutoencoderIS
from proxy_lib import load_fold

DIAG_ROOT = os.path.join(REPO_DIR, 'results', 'diag-aggressive-rates', 'diag')


def process_fold(dataset, fold):
    torch.set_num_threads(4)
    out_dir = os.path.join(DIAG_ROOT, dataset, 'sae-rate-40')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'fold_{fold}.parquet')
    if os.path.exists(out_file):
        print(f"[{dataset} f{fold}] Already exists, skipping.")
        return
        
    t0 = time.time()
    X_tr, y_tr, _, _ = load_fold(dataset, fold)
    n = len(y_tr)
    
    # 1. Fit Autoencoder (50 epochs, random_state=13)
    ae = AutoencoderIS(n_epochs=50, random_state=13)
    ae.fit(X_tr, y_tr)
    scores = ae.reconstruction_errors_
    
    # 2. Tessellate space into K=sqrt(N) micro-clusters with seed 13
    n_clusters = int(np.sqrt(n))
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=13, n_init=3)
    cell_ids = km.fit_predict(X_tr)
    
    # Save parquet
    df = pd.DataFrame({
        'idx': np.arange(n),
        're_score': scores,
        'cell_id': cell_ids,
        'cell_size': np.bincount(cell_ids, minlength=n_clusters)[cell_ids]
    })
    df.to_parquet(out_file, index=False)
    print(f"[{dataset} f{fold}] Finished in {time.time()-t0:.2f}s -> {out_file}")


def main():
    datasets = ['20ng', 'acm']
    Parallel(n_jobs=10, verbose=5)(
        delayed(process_fold)(ds, f) for ds in datasets for f in range(10)
    )
    print("All diagnostic parquets generated!")


if __name__ == '__main__':
    main()
