#!/usr/bin/env python3
"""
Analyze class distribution in WebKB maintained sets at 40% reduction.
Compares:
- Full Train
- Random@40
- SAE@40 Original (beta=1)
- SAE@40 Cleaning (beta=1)
- SAE@40 Original (beta=10)
- SAE@40 Cleaning (beta=10)
Specifically examining Class 3 (137 samples) and Class 6 (182 samples).
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(REPO_DIR, 'datasets', 'webkb')
RESULTS_DIR = os.path.join(REPO_DIR, 'results', 'diag-aggressive-rates', 'instance_selection', 'selection', 'webkb')
DIAG_DIR = os.path.join(REPO_DIR, 'results', 'diag-aggressive-rates', 'diag', 'webkb')

def main():
    # Load labels
    with open(os.path.join(DATASET_DIR, 'score.txt')) as f:
        labels = np.array([int(line.strip()) for line in f])
    
    n_classes = len(np.unique(labels))
    class_names = sorted(np.unique(labels))
    
    # Load original splits
    with open(os.path.join(DATASET_DIR, 'splits', 'split_10.pkl'), 'rb') as f:
        df_orig_split = pickle.load(f)

    # Methods to analyze
    methods = {
        'Full Train': None,
        'Random@40': os.path.join(RESULTS_DIR, 'split_10_random-rate-40.pkl'),
        'SAE@40 Orig (b=1)': os.path.join(RESULTS_DIR, 'split_10_sae-rate-40.pkl'),
        'SAE@40 Clean (b=1)': os.path.join(RESULTS_DIR, 'split_10_sae-cleaning-rate-40.pkl'),
        'SAE@40 Orig (b=10)': os.path.join(RESULTS_DIR, 'split_10_sae-beta10-rate-40.pkl'),
        'SAE@40 Clean (b=10)': os.path.join(RESULTS_DIR, 'split_10_sae-cleaning-beta10-rate-40.pkl'),
    }

    loaded_splits = {}
    for m, p in methods.items():
        if p is not None:
            with open(p, 'rb') as f:
                loaded_splits[m] = pickle.load(f)

    # Class counts per method per fold
    # counts[method][fold][c]
    counts_per_fold = {m: [] for m in methods}
    retention_per_fold = {m: [] for m in methods if m != 'Full Train'}

    for f in range(10):
        full_idxs = df_orig_split.loc[f, 'train_idxs']
        full_labels = labels[full_idxs]
        full_c = Counter(full_labels)
        counts_per_fold['Full Train'].append([full_c[c] for c in class_names])

        for m in methods:
            if m == 'Full Train':
                continue
            kept_idxs = loaded_splits[m].loc[f, 'train_idxs']
            kept_labels = labels[kept_idxs]
            kept_c = Counter(kept_labels)
            counts_per_fold[m].append([kept_c[c] for c in class_names])

            # retention rate per class
            ret_rates = [kept_c[c] / full_c[c] if full_c[c] > 0 else 0.0 for c in class_names]
            retention_per_fold[m].append(ret_rates)

    print("=" * 85)
    print("ANALISE DE DISTRIBUICAO DE CLASSES NO WEBKB (10 FOLDS, REMOCAO 40%)")
    print("=" * 85)

    # Mean absolute counts
    print("\n1. CONTAGEM MEDIA DE EXEMPLOS MANTIDOS POR CLASSE (Média dos 10 folds):")
    data_counts = {'Classe': [f"Classe {c}" for c in class_names]}
    for m in methods:
        mean_cnt = np.mean(counts_per_fold[m], axis=0)
        data_counts[m] = [f"{v:.1f}" for v in mean_cnt]
    df_counts = pd.DataFrame(data_counts)
    print(df_counts.to_string(index=False))

    # Mean retention rate (kept / full_train)
    print("\n2. TAXA DE RETENCAO POR CLASSE (kept / full_train):")
    print("   [Taxa esperada para random@40 = 60.00%]")
    data_ret = {'Classe': [f"Classe {c}" for c in class_names]}
    for m in retention_per_fold:
        mean_ret = np.mean(retention_per_fold[m], axis=0)
        data_ret[m] = [f"{v*100:.2f}%" for v in mean_ret]
    df_ret = pd.DataFrame(data_ret)
    print(df_ret.to_string(index=False))

    # Mean proportion inside maintained set
    print("\n3. PROPORCAO DA CLASSE NO CONJUNTO MANTIDO (% do treino reduzido):")
    data_prop = {'Classe': [f"Classe {c}" for c in class_names]}
    for m in methods:
        arr = np.array(counts_per_fold[m])
        # sum per fold
        tot = arr.sum(axis=1, keepdims=True)
        props = np.mean(arr / tot, axis=0)
        data_prop[m] = [f"{v*100:.2f}%" for v in props]
    df_prop = pd.DataFrame(data_prop)
    print(df_prop.to_string(index=False))

    # 4. Reconstruction Error per Class
    print("\n4. ESTATISTICAS DE ERRO DE RECONSTRUCAO (RE) POR CLASSE:")
    # Collect RE scores across all folds
    re_per_class = {c: [] for c in class_names}
    for f in range(10):
        diag_f = os.path.join(DIAG_DIR, 'sae-rate-40', f"fold_{f}.parquet")
        df_d = pd.read_parquet(diag_f)
        full_idxs = df_d['idx'].values
        class_ids = labels[full_idxs]
        for c in class_names:
            re_per_class[c].extend(df_d.loc[class_ids == c, 're_score'].values)

    df_re = pd.DataFrame({
        'Classe': [f"Classe {c}" for c in class_names],
        'Total Global': [Counter(labels)[c] for c in class_names],
        'RE Médio': [np.mean(re_per_class[c]) for c in class_names],
        'RE Mediano': [np.median(re_per_class[c]) for c in class_names],
        'RE DesvPad': [np.std(re_per_class[c]) for c in class_names],
        'RE Mín': [np.min(re_per_class[c]) for c in class_names],
        'RE Máx': [np.max(re_per_class[c]) for c in class_names],
    })
    print(df_re.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 5. Cluster size distribution for Class 3 and Class 6
    print("\n5. TAMANHO MEDIO DOS CLUSTERS EM QUE CADA CLASSE CAI:")
    cell_sizes_per_class = {c: [] for c in class_names}
    unpruned_cell_prop = {c: [] for c in class_names}
    for f in range(10):
        diag_f = os.path.join(DIAG_DIR, 'sae-rate-40', f"fold_{f}.parquet")
        df_d = pd.read_parquet(diag_f)
        full_idxs = df_d['idx'].values
        class_ids = labels[full_idxs]
        is_unpruned = (df_d['cell_quota_keep'] >= df_d['cell_size']).values
        for c in class_names:
            cell_sizes_per_class[c].extend(df_d.loc[class_ids == c, 'cell_size'].values)
            unpruned_cell_prop[c].append(np.mean(is_unpruned[class_ids == c]))

    df_cells = pd.DataFrame({
        'Classe': [f"Classe {c}" for c in class_names],
        'Tamanho Médio Célula': [np.mean(cell_sizes_per_class[c]) for c in class_names],
        '% Instâncias em Célula Não Podada (a)': [np.mean(unpruned_cell_prop[c])*100 for c in class_names],
    })
    print(df_cells.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

if __name__ == '__main__':
    main()
