#!/usr/bin/env python3
"""
Diagnostic runner for SAE-IS (SublinearAEIS):
Runs original, cleaning, and random at 40% and 60% reduction rates on books, dblp, and webkb,
and beta=10 variants on webkb.
Saves diagnostic parquets to results/diag-aggressive-rates/diag/<ds>/<method>/fold_<k>.parquet
and splits to results/diag-aggressive-rates/instance_selection/selection/<ds>/
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

# Add unsupervised-is to sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if os.path.basename(BASE_DIR) == 'scripts':
    REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
else:
    REPO_DIR = BASE_DIR
UNSUP_DIR = os.path.join(REPO_DIR, 'unsupervised-is')
if UNSUP_DIR not in sys.path:
    sys.path.insert(0, UNSUP_DIR)

from src.main.python.utils.general import get_data, get_splits, checkpoint_splits, translate_train_idxinfold
from src.main.python.iSel.autoencoder_is import AutoencoderIS
from src.main.python.iSel.sublinear_ae_is import _find_lambda
from src.main.python.iSel.random_is import RandomIS


def run_diagnostics():
    datasets = ['books', 'dblp', 'webkb']
    rates = [40, 60]
    out_base = os.path.join(REPO_DIR, 'results', 'diag-aggressive-rates')
    diag_base = os.path.join(out_base, 'diag')
    split_out_base = os.path.join(out_base, 'instance_selection', 'selection')
    os.makedirs(diag_base, exist_ok=True)
    os.makedirs(split_out_base, exist_ok=True)

    print(f"===========================================================")
    print(f"Starting SAE-IS Diagnostic Selection")
    print(f"Target dir: {out_base}")
    print(f"Datasets: {datasets}")
    print(f"===========================================================")

    for ds in datasets:
        print(f"\n===========================================================")
        print(f"Processing Dataset: {ds}")
        print(f"===========================================================")

        # Methods to run for this dataset
        methods = [
            ('sae-rate-40', 'original', 0.40, 1.0),
            ('sae-cleaning-rate-40', 'cleaning', 0.40, 1.0),
            ('sae-rate-60', 'original', 0.60, 1.0),
            ('sae-cleaning-rate-60', 'cleaning', 0.60, 1.0),
            ('random-rate-40', 'random', 0.40, 1.0),
            ('random-rate-60', 'random', 0.60, 1.0),
        ]
        if ds == 'webkb':
            methods.extend([
                ('sae-beta10-rate-40', 'original', 0.40, 10.0),
                ('sae-cleaning-beta10-rate-40', 'cleaning', 0.40, 10.0),
            ])

        splits_file = os.path.join(REPO_DIR, 'datasets', ds, 'splits', 'split_10.pkl')
        orig_splits_df = get_splits(splits_file)

        # Storage for split DataFrames
        splits_to_save = {
            m_name: {'test_idxs': [], 'train_idxs': []}
            for m_name, _, _, _ in methods
        }

        for f in range(10):
            t_fold_start = time.time()
            print(f"\n--- [{ds}] Fold {f}/10 ---")
            train_global_idxs = orig_splits_df.loc[f].train_idxs
            test_global_idxs = orig_splits_df.loc[f].test_idxs

            for m_name, _, _, _ in methods:
                splits_to_save[m_name]['test_idxs'].append(test_global_idxs)

            data_dir = os.path.join(REPO_DIR, 'datasets', ds, 'jina-v5')
            X_train, y_train, _, _, _ = get_data(data_dir + '/', f)
            X_dense = X_train.toarray() if hasattr(X_train, 'toarray') else np.asarray(X_train, dtype=np.float64)
            n_samples = len(y_train)

            # 1. Train Autoencoder once per fold with seed 13
            print(f"[{ds} | Fold {f}] Training Autoencoder (50 epochs, random_state=13)...")
            ae = AutoencoderIS(n_epochs=50, random_state=13)
            ae.fit(X_dense, y_train)
            scores = ae.reconstruction_errors_
            print(f"[{ds} | Fold {f}] AE scores: min={scores.min():.4f}, median={np.median(scores):.4f}, max={scores.max():.4f}")

            # 2. Tessellation into K=sqrt(N) micro-clusters with seed 13
            n_clusters = int(np.sqrt(n_samples))
            print(f"[{ds} | Fold {f}] Tessellating into {n_clusters} clusters (random_state=13)...")
            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=13, n_init=3)
            labels = km.fit_predict(X_dense)
            sizes = np.bincount(labels, minlength=n_clusters)

            # Pre-compute quotas for 40% and 60%
            quotas_cache = {}
            for red_rate in [0.40, 0.60]:
                target_keep = int(n_samples * (1.0 - red_rate))
                lam = _find_lambda(sizes, 0.5, target_keep)
                quotas = np.array([min(s, int(np.ceil(lam * (s ** 0.5)))) for s in sizes])
                quotas_cache[red_rate] = (lam, quotas)

            # 3. Run each method
            for m_name, mode, red_rate, beta in methods:
                t0 = time.time()
                m_diag_dir = os.path.join(diag_base, ds, m_name)
                os.makedirs(m_diag_dir, exist_ok=True)
                parquet_path = os.path.join(m_diag_dir, f"fold_{f}.parquet")

                if mode == 'random':
                    sel_rate = 1.0 - red_rate
                    rand_sel = RandomIS(selection_rate=sel_rate, random_state=28101994)
                    rand_sel.fit(X_dense, y_train)
                    kept_indices = rand_sel.sample_indices_

                    mask = np.zeros(n_samples, dtype=bool)
                    mask[kept_indices] = True

                    # Use quotas from the corresponding SAE rate for reference
                    _, quotas = quotas_cache[red_rate]
                    cell_quotas = quotas[labels]
                else:
                    lam, quotas = quotas_cache[red_rate]
                    mask = np.ones(n_samples, dtype=bool)

                    # Reset numpy random seed to 13 before sampling to ensure identical state
                    np.random.seed(13)

                    for c in range(n_clusters):
                        c_idx = np.where(labels == c)[0]
                        s_c = len(c_idx)
                        if s_c == 0:
                            continue
                        keep_count = quotas[c]
                        remove_count = s_c - keep_count

                        if remove_count > 0:
                            c_scores = scores[c_idx]
                            if mode == 'cleaning':
                                weights = (c_scores + 1e-8) ** beta
                                p_remove = weights / np.sum(weights)
                            else:
                                inv_scores = 1.0 / (c_scores + 1e-8)
                                weights = inv_scores ** beta
                                p_remove = weights / np.sum(weights)

                            to_remove_local = np.random.choice(
                                len(c_idx),
                                size=remove_count,
                                replace=False,
                                p=p_remove
                            )
                            mask[c_idx[to_remove_local]] = False

                    cell_quotas = quotas[labels]
                    kept_indices = np.where(mask)[0]

                # Save diagnostic parquet
                df_diag = pd.DataFrame({
                    'idx': np.array(train_global_idxs, dtype=np.int64),
                    're_score': scores.astype(np.float64),
                    'cell_id': labels.astype(np.int64),
                    'cell_size': sizes[labels].astype(np.int64),
                    'cell_quota_keep': cell_quotas.astype(np.int64),
                    'kept': mask.astype(bool),
                })
                df_diag.to_parquet(parquet_path, index=False)

                # Record selected indices
                splits_to_save[m_name]['train_idxs'].append(kept_indices)
                print(f"  Method {m_name:28s}: kept {len(kept_indices)}/{n_samples} ({len(kept_indices)/n_samples:.1%}) in {time.time()-t0:.2f}s -> {parquet_path}")

            print(f"Fold {f} completed in {time.time()-t_fold_start:.2f}s")

        # 4. Save splits for each method
        ds_split_dir = os.path.join(split_out_base, ds)
        os.makedirs(ds_split_dir, exist_ok=True)

        for m_name, _, _, _ in methods:
            df_idxinfold = pd.DataFrame(splits_to_save[m_name])
            file_idxinfold = os.path.join(ds_split_dir, f"split_10_{m_name}_idxinfold.pkl")
            file_translated = os.path.join(ds_split_dir, f"split_10_{m_name}.pkl")

            checkpoint_splits(df_idxinfold, file_idxinfold)
            df_translated = translate_train_idxinfold(df_idxinfold, orig_splits_df)
            checkpoint_splits(df_translated, file_translated)

            # Also create saida files for compatibility
            with open(os.path.join(ds_split_dir, f"saida_{m_name}"), "w") as f_out:
                f_out.write(f"{m_name}\n")
            print(f"Saved splits for {ds} / {m_name} in {ds_split_dir}")

    print("\n===========================================================")
    print("All diagnostic runs completed successfully!")
    print("===========================================================")


if __name__ == '__main__':
    run_diagnostics()
