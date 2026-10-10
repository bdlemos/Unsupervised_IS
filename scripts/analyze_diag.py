#!/usr/bin/env python3
"""
Analysis script for SAE-IS diagnostic results.
Computes:
1. P(keep | RE decile) for original, cleaning, random (average over 10 folds)
2. Decomposition of overlap original x cleaning: group (a) unpruned vs group (b) pruned
3. Quota distribution (histogram of cell_quota_keep / cell_size)
4. Beta=10 analysis for webkb
"""

import os
import glob
import numpy as np
import pandas as pd

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIAG_DIR = os.path.join(REPO_DIR, 'results', 'diag-aggressive-rates', 'diag')


def analyze_dataset_rate(ds, rate):
    orig_name = f"sae-rate-{rate}"
    clean_name = f"sae-cleaning-rate-{rate}"
    rand_name = f"random-rate-{rate}"

    orig_dir = os.path.join(DIAG_DIR, ds, orig_name)
    clean_dir = os.path.join(DIAG_DIR, ds, clean_name)
    rand_dir = os.path.join(DIAG_DIR, ds, rand_name)

    if not os.path.isdir(orig_dir) or not os.path.isdir(clean_dir):
        print(f"Skipping {ds} rate {rate} - directories not found.")
        return None

    # Storage across 10 folds
    p_keep_orig_folds = []
    p_keep_clean_folds = []
    p_keep_rand_folds = []

    frac_a_folds = []
    frac_b_folds = []
    overlap_overall_folds = []
    overlap_a_folds = []
    overlap_b_folds = []
    rand_expected_b_folds = []

    quota_ratios_all_cells = []
    quota_ratios_weighted = []

    for f in range(10):
        f_orig = os.path.join(orig_dir, f"fold_{f}.parquet")
        f_clean = os.path.join(clean_dir, f"fold_{f}.parquet")
        f_rand = os.path.join(rand_dir, f"fold_{f}.parquet")

        df_orig = pd.read_parquet(f_orig)
        df_clean = pd.read_parquet(f_clean)
        df_rand = pd.read_parquet(f_rand)

        scores = df_orig['re_score'].values
        kept_orig = df_orig['kept'].values
        kept_clean = df_clean['kept'].values
        kept_rand = df_rand['kept'].values

        # 1. P(manter | decil)
        # Using pd.qcut on scores for fold
        deciles = pd.qcut(scores, q=10, labels=False, duplicates='drop')
        n_dec = 10
        p_orig_d = [np.mean(kept_orig[deciles == d]) for d in range(n_dec)]
        p_clean_d = [np.mean(kept_clean[deciles == d]) for d in range(n_dec)]
        p_rand_d = [np.mean(kept_rand[deciles == d]) for d in range(n_dec)]

        p_keep_orig_folds.append(p_orig_d)
        p_keep_clean_folds.append(p_clean_d)
        p_keep_rand_folds.append(p_rand_d)

        # 2. Decomposição da sobreposição
        # group (a): cell_quota_keep >= cell_size
        # group (b): cell_quota_keep < cell_size
        is_a = (df_orig['cell_quota_keep'] >= df_orig['cell_size']).values
        is_b = ~is_a

        n_total = len(df_orig)
        frac_a = np.sum(is_a) / n_total
        frac_b = np.sum(is_b) / n_total
        frac_a_folds.append(frac_a)
        frac_b_folds.append(frac_b)

        # Overall overlap: |orig & clean| / |orig|
        ov_all = np.sum(kept_orig & kept_clean) / np.sum(kept_orig)
        overlap_overall_folds.append(ov_all)

        # Overlap in (a):
        ov_a = np.sum(kept_orig[is_a] & kept_clean[is_a]) / np.sum(kept_orig[is_a]) if np.sum(kept_orig[is_a]) > 0 else 1.0
        overlap_a_folds.append(ov_a)

        # Overlap in (b):
        ov_b = np.sum(kept_orig[is_b] & kept_clean[is_b]) / np.sum(kept_orig[is_b]) if np.sum(kept_orig[is_b]) > 0 else 0.0
        overlap_b_folds.append(ov_b)

        # Random expected overlap in (b): retention rate in (b)
        # In (b), if kept at rate r_b, overlap with itself is r_b
        r_b = np.sum(kept_clean[is_b]) / np.sum(is_b) if np.sum(is_b) > 0 else 0.0
        rand_expected_b_folds.append(r_b)

        # 3. Quota ratios
        # Unique cells in this fold
        df_cells = df_orig.drop_duplicates(subset=['cell_id'])
        ratios = (df_cells['cell_quota_keep'] / df_cells['cell_size']).values
        quota_ratios_all_cells.extend(ratios)

        # Instance-weighted ratios
        quota_ratios_weighted.extend((df_orig['cell_quota_keep'] / df_orig['cell_size']).values)

    return {
        'dataset': ds,
        'rate': rate,
        'p_orig': np.mean(p_keep_orig_folds, axis=0),
        'p_clean': np.mean(p_keep_clean_folds, axis=0),
        'p_rand': np.mean(p_keep_rand_folds, axis=0),
        'frac_a': np.mean(frac_a_folds),
        'frac_b': np.mean(frac_b_folds),
        'overlap_overall': np.mean(overlap_overall_folds),
        'overlap_a': np.mean(overlap_a_folds),
        'overlap_b': np.mean(overlap_b_folds),
        'rand_expected_b': np.mean(rand_expected_b_folds),
        'quota_ratios_all_cells': np.array(quota_ratios_all_cells),
        'quota_ratios_weighted': np.array(quota_ratios_weighted),
    }


def analyze_beta10_webkb():
    orig_name = "sae-beta10-rate-40"
    clean_name = "sae-cleaning-beta10-rate-40"
    orig_dir = os.path.join(DIAG_DIR, 'webkb', orig_name)
    clean_dir = os.path.join(DIAG_DIR, 'webkb', clean_name)

    if not os.path.isdir(orig_dir) or not os.path.isdir(clean_dir):
        return None

    p_keep_orig_folds = []
    p_keep_clean_folds = []
    frac_a_folds = []
    overlap_overall_folds = []
    overlap_b_folds = []
    rand_expected_b_folds = []

    for f in range(10):
        df_orig = pd.read_parquet(os.path.join(orig_dir, f"fold_{f}.parquet"))
        df_clean = pd.read_parquet(os.path.join(clean_dir, f"fold_{f}.parquet"))

        scores = df_orig['re_score'].values
        kept_orig = df_orig['kept'].values
        kept_clean = df_clean['kept'].values

        deciles = pd.qcut(scores, q=10, labels=False, duplicates='drop')
        p_orig_d = [np.mean(kept_orig[deciles == d]) for d in range(10)]
        p_clean_d = [np.mean(kept_clean[deciles == d]) for d in range(10)]
        p_keep_orig_folds.append(p_orig_d)
        p_keep_clean_folds.append(p_clean_d)

        is_a = (df_orig['cell_quota_keep'] >= df_orig['cell_size']).values
        is_b = ~is_a

        frac_a_folds.append(np.mean(is_a))
        ov_all = np.sum(kept_orig & kept_clean) / np.sum(kept_orig)
        overlap_overall_folds.append(ov_all)

        ov_b = np.sum(kept_orig[is_b] & kept_clean[is_b]) / np.sum(kept_orig[is_b])
        overlap_b_folds.append(ov_b)
        rand_expected_b_folds.append(np.sum(kept_clean[is_b]) / np.sum(is_b))

    return {
        'p_orig': np.mean(p_keep_orig_folds, axis=0),
        'p_clean': np.mean(p_keep_clean_folds, axis=0),
        'frac_a': np.mean(frac_a_folds),
        'overlap_overall': np.mean(overlap_overall_folds),
        'overlap_b': np.mean(overlap_b_folds),
        'rand_expected_b': np.mean(rand_expected_b_folds),
    }


def main():
    print("=" * 80)
    print("DIAGNOSTIC ANALYSIS RESULTS")
    print("=" * 80)

    results = []
    for ds in ['books', 'dblp', 'webkb']:
        for rate in [40, 60]:
            res = analyze_dataset_rate(ds, rate)
            if res:
                results.append(res)

    for res in results:
        ds = res['dataset']
        rate = res['rate']
        print(f"\n=======================================================")
        print(f"Dataset: {ds} | Taxa de remoção: {rate}% (Target Keep: {100-rate}%)")
        print(f"=======================================================")

        print("\n--- 1. P(manter | decil de RE) ---")
        df_p = pd.DataFrame({
            'Decil': [f"D{d} ({['Min RE', '', '', '', '', '', '', '', '', 'Max RE'][d]})" for d in range(10)],
            'Original': res['p_orig'],
            'Cleaning': res['p_clean'],
            'Random': res['p_rand'],
        })
        print(df_p.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\n--- 2. Decomposição da Sobreposição (Original x Cleaning) ---")
        print(f"Fração de instâncias em (a) [células não podadas]: {res['frac_a']:.4f} ({res['frac_a']*100:.2f}%)")
        print(f"Fração de instâncias em (b) [células podadas]:     {res['frac_b']:.4f} ({res['frac_b']*100:.2f}%)")
        print(f"Sobreposição em (a) [não podadas]:                 {res['overlap_a']:.4f} (sempre 1.0000)")
        print(f"Sobreposição em (b) [células podadas]:             {res['overlap_b']:.4f}")
        print(f"Aleatório esperado em (b):                         {res['rand_expected_b']:.4f}")
        print(f"Sobreposição Geral (Global):                       {res['overlap_overall']:.4f}")

        # Histogram of quota ratios
        ratios = res['quota_ratios_all_cells']
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9999, 1.0001]
        counts, _ = np.histogram(ratios, bins=bins)
        bin_labels = ["[0.0, 0.2)", "[0.2, 0.4)", "[0.4, 0.6)", "[0.6, 0.8)", "[0.8, 1.0)", "1.0 (inteiras)"]
        print("\n--- 3. Distribuição das Cotas por Célula (cell_quota_keep / cell_size) ---")
        for bl, c in zip(bin_labels, counts):
            bar = "#" * int(c / len(ratios) * 40)
            print(f"  {bl:15s}: {c:5d} células ({c/len(ratios)*100:5.1f}%) | {bar}")

    # Beta=10
    b10_res = analyze_beta10_webkb()
    if b10_res:
        print("\n" + "=" * 80)
        print("PASSO 3: SANIDADE COM BETA=10 NO WEBKB (Taxa 40%)")
        print("=" * 80)
        print(f"Sobreposição Geral (beta=10):                      {b10_res['overlap_overall']:.4f}")
        print(f"Sobreposição em (b) podadas (beta=10):             {b10_res['overlap_b']:.4f}")
        print(f"Aleatório esperado em (b):                         {b10_res['rand_expected_b']:.4f}")
        print("\nP(manter | decil) para beta=10:")
        df_b10 = pd.DataFrame({
            'Decil': [f"D{d}" for d in range(10)],
            'Original (beta=10)': b10_res['p_orig'],
            'Cleaning (beta=10)': b10_res['p_clean'],
        })
        print(df_b10.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == '__main__':
    main()
