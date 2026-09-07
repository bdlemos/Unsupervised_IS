#!/usr/bin/env python3
"""
Exp 5 — Entropy vs Empirical Optimal Rate
==========================================

Consumes results from Exp 4 (SAE-IS at fixed rates) and compares the
ESAE adaptive rate with the empirically best rate for each dataset.

Generates:
1. Per-dataset curve: Macro-F1 vs fixed rate, with ESAE operating point marked
2. Aggregate plot: H (entropy) vs best empirical rate, with regression line
3. Summary table (CSV)

Usage:
    python analysis/exp5_entropy_vs_empirical.py
    python analysis/exp5_entropy_vs_empirical.py --exp4-dir results/exp4-fixed-rates
"""

import sys
import os
import argparse
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)


DEFAULT_DATASETS = ['sst2', 'ohsumed', 'reuters90', 'books', 'trec', '20ng']
FIXED_RATES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def _get_macro_f1(row):
    """Extract Macro-F1 from a CSV row (dict). Uses macro_avg_ic if available,
    otherwise averages mac0..mac9."""
    if 'macro_avg_ic' in row and row['macro_avg_ic']:
        try:
            return float(row['macro_avg_ic'])
        except (ValueError, TypeError):
            pass
    mac_cols = [f'mac{i}' for i in range(10)]
    vals = []
    for c in mac_cols:
        if c in row and row[c]:
            try:
                vals.append(float(row[c]))
            except (ValueError, TypeError):
                pass
    return sum(vals) / len(vals) if vals else None


def load_exp4_results(exp4_dir: str, datasets: list) -> pd.DataFrame:
    """Load Macro-F1 results from Exp 4 fixed-rate runs.

    Expects the structure:
        exp4_dir/rate_<X>/classificacao/results/results_modernbert.csv

    CSV columns: dataset, method, reduction, class, mac0..mac9, macro_avg_ic

    Returns a DataFrame with columns: dataset, rate, macro_f1
    """
    import csv
    rows = []

    for rate in FIXED_RATES:
        rate_int = int(rate * 100)
        rate_dir = os.path.join(exp4_dir, f'rate_{rate_int}')
        results_csv = os.path.join(rate_dir, 'classificacao', 'results',
                                   'results_modernbert.csv')

        if not os.path.exists(results_csv):
            print(f"  WARNING: Missing results for rate {rate_int}%: {results_csv}")
            continue

        with open(results_csv) as f:
            reader = csv.DictReader(f)
            for csv_row in reader:
                ds = csv_row.get('dataset', '')
                if ds in datasets:
                    f1 = _get_macro_f1(csv_row)
                    if f1 is not None:
                        rows.append({'dataset': ds, 'rate': rate, 'macro_f1': f1})

        print(f"  Loaded rate {rate_int}%: {results_csv}")

    return pd.DataFrame(rows)


def _compute_normalized_entropy(sizes):
    """Compute the normalized entropy of the cluster size distribution.
    Mirrors the logic in entropy_sublinear_ae_is.py.
    """
    sizes = np.array(sizes)
    total = np.sum(sizes)
    if total == 0:
        return 1.0
    nonzero = sizes[sizes > 0]
    k = len(nonzero)
    if k <= 1:
        return 1.0
    probs = nonzero / total
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(k)
    return entropy / max_entropy


def load_esae_results(esae_results_dir: str, datasets: list) -> dict:
    """Load ESAE reduction rate and Macro-F1 from actual output files.

    Sources:
    - Per-dataset selection JSON: instance_selection/selection/<ds>/saida_entropy-sublinear-ae-is.json
      → provides reduction_mean (rate)
    - Classification CSV: classificacao/results/results_modernbert.csv
      → provides Macro-F1 per dataset
    - Entropy is NOT saved in outputs, so we note it as None here.
      (It can optionally be recomputed from raw data if needed.)

    Returns dict: dataset -> {entropy, rate, macro_f1}
    """
    import csv
    results = {}

    # 1. Load reduction rates from per-dataset selection JSONs
    for ds in datasets:
        json_path = os.path.join(esae_results_dir, 'instance_selection',
                                 'selection', ds,
                                 'saida_entropy-sublinear-ae-is.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            info = data.get('info', {})
            results[ds] = {
                'entropy': None,  # Not saved in outputs
                'rate': info.get('reduction_mean', None),
            }
            print(f"  {ds}: reduction_mean={info.get('reduction_mean', 'N/A')}")
        else:
            print(f"  WARNING: No ESAE selection JSON for {ds}: {json_path}")

    # 2. Also try the summary JSON (nested dict: {dataset: {method: {stats}}})
    summary_path = os.path.join(esae_results_dir, 'instance_selection',
                                'selection_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        for ds in datasets:
            if ds not in results and ds in summary:
                method_data = summary[ds].get('entropy-sublinear-ae-is', {})
                reduction = method_data.get('reduction', {})
                results[ds] = {
                    'entropy': None,
                    'rate': reduction.get('mean', None) if isinstance(reduction, dict) else reduction,
                }

    # 3. Load Macro-F1 from classification CSV
    clf_csv = os.path.join(esae_results_dir, 'classificacao', 'results',
                           'results_modernbert.csv')
    if os.path.exists(clf_csv):
        with open(clf_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = row.get('dataset', '')
                if ds in datasets and row.get('method') == 'entropy-sublinear-ae-is':
                    f1 = _get_macro_f1(row)
                    if ds in results:
                        results[ds]['macro_f1'] = f1
                    else:
                        results[ds] = {
                            'entropy': None,
                            'rate': float(row.get('reduction', 0)),
                            'macro_f1': f1,
                        }
    else:
        print(f"  WARNING: No classification CSV at {clf_csv}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Exp 5: Entropy vs Empirical Optimal Rate')
    parser.add_argument('--exp4-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, '..', 'results',
                                             'exp4-fixed-rates'),
                        help='Directory with Exp 4 fixed-rate results')
    parser.add_argument('--esae-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, '..', 'results', 'jina-v5'),
                        help='Directory with ESAE baseline results')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, '..', 'results',
                                             'exp5-entropy-vs-empirical'),
                        help='Output directory')
    parser.add_argument('--datasets', type=str,
                        default=','.join(DEFAULT_DATASETS))
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Exp 5: Entropy vs Empirical Optimal Rate")
    print(f"  Exp 4 dir: {args.exp4_dir}")
    print(f"  ESAE dir:  {args.esae_dir}")
    print(f"  Output:    {args.output_dir}")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\n[Step 1] Loading Exp 4 fixed-rate results...")
    exp4_df = load_exp4_results(args.exp4_dir, datasets)

    if exp4_df.empty:
        print("  ERROR: No Exp 4 results found. Run Exp 4 first.")
        print("  Expected structure: exp4-dir/rate_<X>/classificacao/results/results_modernbert.csv")
        print("  This script will generate placeholder plots for now.")

    print(f"\n[Step 2] Loading ESAE baseline results...")
    esae_results = load_esae_results(args.esae_dir, datasets)

    # ── Plot 1: Per-dataset F1 vs Rate curves ──────────────────────────────
    print(f"\n[Step 3] Generating per-dataset F1 vs Rate plots...")
    sns.set_theme(style='whitegrid', font_scale=1.1)

    for ds in datasets:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        # Fixed rate curve
        ds_data = exp4_df[exp4_df['dataset'] == ds] if not exp4_df.empty else pd.DataFrame()
        if not ds_data.empty:
            ds_sorted = ds_data.sort_values('rate')
            ax.plot(ds_sorted['rate'] * 100, ds_sorted['macro_f1'],
                    'b-o', linewidth=2, markersize=8, label='SAE-IS (fixed rate)')

        # ESAE operating point
        if ds in esae_results and esae_results[ds].get('rate') is not None:
            esae_rate = esae_results[ds]['rate']
            esae_f1 = esae_results[ds].get('macro_f1')
            esae_label = f"ESAE (adaptive: {esae_rate:.1%})"

            ax.axvline(x=esae_rate * 100, color='red', linestyle='--',
                       linewidth=2, alpha=0.7, label=esae_label)

            # If we have the ESAE F1, plot it as a point
            if esae_f1 is not None:
                ax.scatter([esae_rate * 100], [esae_f1], color='red',
                           s=120, zorder=5, marker='*',
                           label=f'ESAE F1={esae_f1:.4f}')

        ax.set_xlabel('Reduction Rate (%)')
        ax.set_ylabel('Macro-F1')
        ax.set_title(f'{ds.upper()} — Macro-F1 vs Reduction Rate')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f'{ds}_f1_vs_rate.png'))
        plt.close(fig)

    # ── Plot 2: Entropy vs Best Empirical Rate ─────────────────────────────
    print(f"\n[Step 4] Generating Entropy vs Best Empirical Rate plot...")

    summary_rows = []
    for ds in datasets:
        ds_data = exp4_df[exp4_df['dataset'] == ds] if not exp4_df.empty else pd.DataFrame()
        best_rate = ds_data.loc[ds_data['macro_f1'].idxmax(), 'rate'] if not ds_data.empty else None
        best_f1 = ds_data['macro_f1'].max() if not ds_data.empty else None

        esae_info = esae_results.get(ds, {})
        entropy = esae_info.get('entropy')
        esae_rate = esae_info.get('rate')
        esae_f1 = esae_info.get('macro_f1')

        summary_rows.append({
            'dataset': ds,
            'entropy_H': entropy,
            'esae_rate': esae_rate,
            'esae_f1': esae_f1,
            'best_empirical_rate': best_rate,
            'best_empirical_f1': best_f1,
            'delta_rate': (esae_rate - best_rate) if (esae_rate and best_rate) else None,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, 'entropy_vs_empirical_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Summary saved to {summary_csv}")

    # Scatter plot: H vs best empirical rate
    valid = summary_df.dropna(subset=['entropy_H', 'best_empirical_rate'])
    if len(valid) >= 2:
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        ax.scatter(valid['entropy_H'], valid['best_empirical_rate'] * 100,
                   s=100, c='blue', zorder=5, label='Best Empirical Rate')
        ax.scatter(valid['entropy_H'], valid['esae_rate'] * 100,
                   s=100, c='red', marker='x', zorder=5, label='ESAE Adaptive Rate')

        for _, row in valid.iterrows():
            ax.annotate(row['dataset'], (row['entropy_H'], row['best_empirical_rate'] * 100),
                        textcoords='offset points', xytext=(5, 5), fontsize=9)

        # Correlation
        rho, p = spearmanr(valid['entropy_H'], valid['best_empirical_rate'])
        ax.text(0.02, 0.95, f'Spearman ρ = {rho:.3f} (p={p:.3f})',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Normalized Entropy (H)')
        ax.set_ylabel('Reduction Rate (%)')
        ax.set_title('Entropy vs Empirically Optimal Reduction Rate')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, 'entropy_vs_best_rate.png'))
        plt.close(fig)
    else:
        print("  Not enough data for entropy vs rate plot (need Exp 4 results first)")

    print(f"\n{'='*70}")
    print(f"  Exp 5 complete. Results in {args.output_dir}")
    print(f"{'='*70}")
    print(f"\n  Summary:")
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
