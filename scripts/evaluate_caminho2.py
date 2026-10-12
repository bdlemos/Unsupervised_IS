#!/usr/bin/env python3
"""
Evaluate Caminho 2: Intra-Cluster Z-Score + Softmax Sampling with Temperature tau.
Evaluates across Books, DBLP, and WebKB for 10 folds at rates 40% and 60%:
- Overlaps (overall and inside group b)
- Decile curves P(keep | decile)
- Class retention rates (specifically checking WebKB Class 3 & Class 6, and minority classes in Books & DBLP)
- Comparison between coupled seeds vs decoupled seeds
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIAG_DIR = os.path.join(REPO_DIR, 'results', 'diag-aggressive-rates', 'diag')
DATASETS_DIR = os.path.join(REPO_DIR, 'datasets')
REPORTS_DIR = os.path.join(REPO_DIR, 'reports')


def sample_zscore_softmax(scores, cell_ids, quotas, mode, tau, seed):
    n_samples = len(scores)
    mask = np.ones(n_samples, dtype=bool)
    rng = np.random.RandomState(seed)
    n_clusters = int(np.max(cell_ids)) + 1

    for c in range(n_clusters):
        c_idx = np.where(cell_ids == c)[0]
        s_c = len(c_idx)
        if s_c <= 1:
            continue
        k_c = quotas[c_idx[0]]
        r_c = s_c - k_c
        if r_c > 0:
            c_s = scores[c_idx]
            mu = np.mean(c_s)
            sigma = np.std(c_s)
            if sigma > 1e-8:
                z = (c_s - mu) / sigma
            else:
                z = np.zeros_like(c_s)

            if mode == 'original':
                logits = - z / tau
            else:
                logits = + z / tau

            logits -= np.max(logits)
            weights = np.exp(logits)
            p_remove = weights / np.sum(weights)

            to_remove_local = rng.choice(s_c, size=r_c, replace=False, p=p_remove)
            mask[c_idx[to_remove_local]] = False

    return mask


def run_evaluation():
    datasets = ['books', 'dblp', 'webkb']
    rates = [40, 60]
    taus = [0.5, 1.0, 2.0]

    all_results = {}

    for rate in rates:
        all_results[rate] = {}
        target_keep_pct = 100 - rate

        print(f"\n================================================================================")
        print(f"AVALIANDO CAMINHO 2: TAXA DE REMOÇÃO {rate}% (TARGET KEEP: {target_keep_pct}%)")
        print(f"================================================================================")

        for ds in datasets:
            print(f"\n--- Dataset: {ds} (Taxa {rate}%) ---")

            # Load labels
            with open(os.path.join(DATASETS_DIR, ds, 'score.txt')) as f:
                labels = np.array([int(line.strip()) for line in f])
            classes = sorted(np.unique(labels))

            ds_results = {}

            for tau in taus:
                overlaps_decoupled = []
                overlaps_b_decoupled = []
                overlaps_coupled = []
                overlaps_b_coupled = []
                rand_expected_b_list = []

                p_orig_deciles = []
                p_clean_deciles = []

                ret_orig_per_class = []
                ret_clean_per_class = []
                ret_rand_per_class = []

                for f in range(10):
                    parquet_f = os.path.join(DIAG_DIR, ds, f'sae-rate-{rate}', f"fold_{f}.parquet")
                    rand_parquet_f = os.path.join(DIAG_DIR, ds, f'random-rate-{rate}', f"fold_{f}.parquet")
                    df = pd.read_parquet(parquet_f)
                    df_rand = pd.read_parquet(rand_parquet_f)

                    scores = df['re_score'].values
                    cell_ids = df['cell_id'].values
                    quotas = df['cell_quota_keep'].values
                    sizes = df['cell_size'].values
                    train_idxs = df['idx'].values
                    fold_labels = labels[train_idxs]

                    is_b = (quotas < sizes)
                    r_b = np.sum(quotas[is_b]) / np.sum(sizes[is_b])
                    rand_expected_b_list.append(r_b)

                    # Decoupled seeds
                    mask_orig_dec = sample_zscore_softmax(scores, cell_ids, quotas, 'original', tau, seed=1000 + f)
                    mask_clean_dec = sample_zscore_softmax(scores, cell_ids, quotas, 'cleaning', tau, seed=2000 + f)

                    # Coupled seeds (same seed)
                    mask_orig_coup = sample_zscore_softmax(scores, cell_ids, quotas, 'original', tau, seed=13 + f)
                    mask_clean_coup = sample_zscore_softmax(scores, cell_ids, quotas, 'cleaning', tau, seed=13 + f)

                    # Overlaps decoupled
                    ov_dec = np.sum(mask_orig_dec & mask_clean_dec) / np.sum(mask_orig_dec)
                    ov_b_dec = np.sum(mask_orig_dec[is_b] & mask_clean_dec[is_b]) / np.sum(mask_orig_dec[is_b])
                    overlaps_decoupled.append(ov_dec)
                    overlaps_b_decoupled.append(ov_b_dec)

                    # Overlaps coupled
                    ov_coup = np.sum(mask_orig_coup & mask_clean_coup) / np.sum(mask_orig_coup)
                    ov_b_coup = np.sum(mask_orig_coup[is_b] & mask_clean_coup[is_b]) / np.sum(mask_orig_coup[is_b])
                    overlaps_coupled.append(ov_coup)
                    overlaps_b_coupled.append(ov_b_coup)

                    # Deciles on decoupled
                    deciles = pd.qcut(scores, 10, labels=False, duplicates='drop')
                    p_orig_deciles.append([np.mean(mask_orig_dec[deciles == d]) for d in range(10)])
                    p_clean_deciles.append([np.mean(mask_clean_dec[deciles == d]) for d in range(10)])

                    # Class retention
                    tot_c = Counter(fold_labels)
                    k_orig = Counter(fold_labels[mask_orig_dec])
                    k_clean = Counter(fold_labels[mask_clean_dec])
                    k_rand = Counter(fold_labels[df_rand['kept'].values])

                    ret_orig_per_class.append([k_orig[c] / tot_c[c] for c in classes])
                    ret_clean_per_class.append([k_clean[c] / tot_c[c] for c in classes])
                    ret_rand_per_class.append([k_rand[c] / tot_c[c] for c in classes])

                mean_ov_dec = np.mean(overlaps_decoupled)
                mean_ov_b_dec = np.mean(overlaps_b_decoupled)
                mean_ov_coup = np.mean(overlaps_coupled)
                mean_ov_b_coup = np.mean(overlaps_b_coupled)
                mean_rand_b = np.mean(rand_expected_b_list)

                print(f"  [tau={tau}] Sobreposicao Global: {mean_ov_dec:.4f} | Em (b): {mean_ov_b_dec:.4f} (vs Rand: {mean_rand_b:.4f}) | Mesma seed: {mean_ov_coup:.4f}")

                ds_results[tau] = {
                    'ov_dec': mean_ov_dec,
                    'ov_b_dec': mean_ov_b_dec,
                    'ov_coup': mean_ov_coup,
                    'ov_b_coup': mean_ov_b_coup,
                    'rand_expected_b': mean_rand_b,
                    'p_orig_dec': np.mean(p_orig_deciles, axis=0),
                    'p_clean_dec': np.mean(p_clean_deciles, axis=0),
                    'ret_orig': np.mean(ret_orig_per_class, axis=0),
                    'ret_clean': np.mean(ret_clean_per_class, axis=0),
                    'ret_rand': np.mean(ret_rand_per_class, axis=0),
                    'classes': classes,
                    'class_counts': Counter(labels),
                }

            all_results[rate][ds] = ds_results

    # Build comprehensive markdown report
    report_path = os.path.join(REPORTS_DIR, 'caminho2_zscore_softmax.md')
    with open(report_path, 'w') as out:
        out.write("# Avaliação Experimental do Caminho 2: Z-Score Local + Softmax Temperado\n\n")
        out.write("**Data:** 10 de Outubro de 2026  \n")
        out.write("**Formulação Avaliada:**\n")
        out.write("$$z_i = \\frac{s_i - \\mu_c}{\\sigma_c + 1e-8}, \\quad p_i = \\frac{\\exp(\\pm z_i / \\tau)}{\\sum_{j \\in C_c} \\exp(\\pm z_j / \\tau)}$$\n\n")
        out.write("Onde:\n")
        out.write("- **Original:** logits = $-z / \\tau$ (remove menor erro relativo à célula);\n")
        out.write("- **Cleaning:** logits = $+z / \\tau$ (remove maior erro relativo à célula);\n")
        out.write("- Avaliado para $\\tau \\in \\{0.5, 1.0, 2.0\\}$ em 10 folds de Books, DBLP e WebKB a **40% e 60% de remoção**.\n\n")

        for rate in rates:
            target_keep_pct = 100 - rate
            out.write(f"---\n\n## Taxa de Remoção: {rate}% (Target Keep: {target_keep_pct}%)\n\n")

            out.write("### 1. Resumo das Sobreposições (Original × Cleaning)\n\n")
            out.write("| Dataset | Temperatura $\\tau$ | Sobreposição Global (Desacoplada) | Sobreposição em (b) Podadas | Aleatório Esperado em (b) | Sobreposição (Mesma Seed) |\n")
            out.write("|---|:---:|:---:|:---:|:---:|:---:|\n")
            for ds in datasets:
                for tau in taus:
                    r = all_results[rate][ds][tau]
                    out.write(f"| **{ds}** | $\\tau={tau}$ | **{r['ov_dec']:.4f}** | **{r['ov_b_dec']:.4f}** | {r['rand_expected_b']:.4f} | {r['ov_coup']:.4f} |\n")

            out.write("\n### 2. Curvas de Decis $P(\\text{manter} \\mid \\text{decil de RE})$ (com $\\tau=0.5$)\n\n")
            out.write("| Decil | Books (Orig) | Books (Clean) | DBLP (Orig) | DBLP (Clean) | WebKB (Orig) | WebKB (Clean) |\n")
            out.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for d in range(10):
                b_o = all_results[rate]['books'][0.5]['p_orig_dec'][d]
                b_c = all_results[rate]['books'][0.5]['p_clean_dec'][d]
                d_o = all_results[rate]['dblp'][0.5]['p_orig_dec'][d]
                d_c = all_results[rate]['dblp'][0.5]['p_clean_dec'][d]
                w_o = all_results[rate]['webkb'][0.5]['p_orig_dec'][d]
                w_c = all_results[rate]['webkb'][0.5]['p_clean_dec'][d]
                out.write(f"| **D{d}** | {b_o:.4f} | {b_c:.4f} | {d_o:.4f} | {d_c:.4f} | {w_o:.4f} | {w_c:.4f} |\n")

            out.write("\n### 3. Retenção por Classe no WebKB\n\n")
            out.write(f"*(Taxa esperada para Random@{rate} = {target_keep_pct}.00%)*\n\n")
            out.write(f"| Classe | Total Global | Random@{rate} | Caminho 2 (Orig $\\tau=1.0$) | Caminho 2 (Clean $\\tau=1.0$) | Caminho 2 (Orig $\\tau=0.5$) | Caminho 2 (Clean $\\tau=0.5$) |\n")
            out.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            w_classes = all_results[rate]['webkb'][1.0]['classes']
            w_counts = all_results[rate]['webkb'][1.0]['class_counts']
            for idx, c in enumerate(w_classes):
                tot = w_counts[c]
                r_rand = all_results[rate]['webkb'][1.0]['ret_rand'][idx] * 100
                r_orig_1 = all_results[rate]['webkb'][1.0]['ret_orig'][idx] * 100
                r_clean_1 = all_results[rate]['webkb'][1.0]['ret_clean'][idx] * 100
                r_orig_05 = all_results[rate]['webkb'][0.5]['ret_orig'][idx] * 100
                r_clean_05 = all_results[rate]['webkb'][0.5]['ret_clean'][idx] * 100
                out.write(f"| **Classe {c}** | {tot} | {r_rand:.2f}% | {r_orig_1:.2f}% | {r_clean_1:.2f}% | {r_orig_05:.2f}% | {r_clean_05:.2f}% |\n")

            out.write("\n### 4. Retenção por Classe em Books e DBLP (com $\\tau=1.0$)\n\n")
            for ds in ['books', 'dblp']:
                out.write(f"#### {ds.upper()} (@{rate}%)\n\n")
                out.write(f"| Classe | Total | Random@{rate} | Caminho 2 (Orig $\\tau=1.0$) | Caminho 2 (Clean $\\tau=1.0$) | Diferença Orig × Clean |\n")
                out.write("|:---:|:---:|:---:|:---:|:---:|:---:|\n")
                c_list = all_results[rate][ds][1.0]['classes']
                c_cnts = all_results[rate][ds][1.0]['class_counts']
                for idx, c in enumerate(c_list):
                    tot = c_cnts[c]
                    r_rand = all_results[rate][ds][1.0]['ret_rand'][idx] * 100
                    r_orig = all_results[rate][ds][1.0]['ret_orig'][idx] * 100
                    r_clean = all_results[rate][ds][1.0]['ret_clean'][idx] * 100
                    diff = abs(r_orig - r_clean)
                    out.write(f"| **C{c}** | {tot} | {r_rand:.2f}% | {r_orig:.2f}% | {r_clean:.2f}% | **{diff:.2f}%** |\n")
                out.write("\n")

    print(f"\nRelatório completo salvo com sucesso em: {report_path}")

if __name__ == '__main__':
    run_evaluation()
