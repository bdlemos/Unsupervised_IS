#!/usr/bin/env python3
"""
Exp 1 — Reconstruction Error Analysis
=======================================

Analyzes whether the autoencoder reconstruction error (RE) truly captures
redundancy (low RE) and noise/boundary cases (high RE).

For each dataset:
1. Trains the same AE used by ESAE-IS (fold 0 only)
2. Trains a Logistic Regression as a lightweight proxy classifier
3. For each instance, computes:
   - Reconstruction error (from AE)
   - Distance to class centroid (proxy for centrality/redundancy)
   - Decision margin (from LR)
   - Whether LR classified it correctly (proxy for difficulty)
   - k-NN label purity (fraction of k nearest neighbors with same label)
4. Generates plots and saves a CSV with all per-instance metrics

Usage:
    python analysis/exp1_reconstruction_analysis.py --dataset sst2
    python analysis/exp1_reconstruction_analysis.py --dataset sst2,ohsumed,reuters90
    python analysis/exp1_reconstruction_analysis.py  # runs all 3 defaults
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.main.python.utils.general import get_data
from src.main.python.iSel.autoencoder_is import AutoencoderIS


# ── Configuration ──────────────────────────────────────────────────────────

DEFAULT_DATASETS = ['sst2', 'ohsumed', 'reuters90']
DEFAULT_INPUTREP = 'jina-v5'
AE_EPOCHS = 50
AE_RANDOM_STATE = 13
KNN_K = 10  # k for label purity computation
LR_MAX_ITER = 1000


def compute_class_centroid_distances(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance from each instance to its own class centroid."""
    classes = np.unique(y)
    centroids = {}
    for c in classes:
        centroids[c] = X[y == c].mean(axis=0)

    distances = np.zeros(len(y))
    for i in range(len(y)):
        distances[i] = np.linalg.norm(X[i] - centroids[y[i]])

    return distances


def compute_knn_label_purity(X: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    """Fraction of k nearest neighbors with the same label."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    # Skip the first neighbor (self)
    neighbor_labels = y[indices[:, 1:]]
    purity = np.mean(neighbor_labels == y[:, None], axis=1)
    return purity


def analyze_dataset(dataset_name: str, datasets_dir: str, output_dir: str, inputrep: str):
    """Run the full reconstruction error analysis for one dataset."""
    print(f"\n{'='*70}")
    print(f"  Exp 1: Reconstruction Error Analysis — {dataset_name.upper()}")
    print(f"{'='*70}")

    dataset_path = os.path.join(datasets_dir, dataset_name, inputrep) + '/'
    plots_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(plots_dir, exist_ok=True)

    # ── Step 0: Load data (fold 0 only) ────────────────────────────────────
    print(f"[Step 0] Loading data from {dataset_path} (fold 0)...")
    X_train, y_train, X_test, y_test, n_classes = get_data(dataset_path, f=0)

    # Convert sparse to dense if needed
    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
    X_train = np.asarray(X_train, dtype=np.float64)

    print(f"  Instances: {len(y_train)}, Features: {X_train.shape[1]}, Classes: {n_classes}")

    # ── Step 1: Train Autoencoder and get reconstruction errors ────────────
    print(f"[Step 1] Training Autoencoder ({AE_EPOCHS} epochs)...")
    ae = AutoencoderIS(
        n_epochs=AE_EPOCHS,
        low_percentile=0.0,
        high_percentile=100.0,
        beta=0.0,
        theta=0.0,
        random_state=AE_RANDOM_STATE,
    )
    ae.fit(X_train, y_train)
    re_scores = ae.reconstruction_errors_
    print(f"  RE stats: min={re_scores.min():.6f}, median={np.median(re_scores):.6f}, "
          f"max={re_scores.max():.6f}")

    # ── Step 2: Train Logistic Regression proxy ────────────────────────────
    print(f"[Step 2] Training Logistic Regression (max_iter={LR_MAX_ITER})...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=LR_MAX_ITER, random_state=AE_RANDOM_STATE, n_jobs=-1)
    lr.fit(X_scaled, y_train)

    y_pred = lr.predict(X_scaled)
    is_correct = (y_pred == y_train).astype(int)
    train_acc = is_correct.mean()
    print(f"  Train accuracy: {train_acc:.4f}")

    # LR prediction confidence [0, 1]
    probs = lr.predict_proba(X_scaled)
    confidence = np.max(probs, axis=1)

    # ── Step 3: Compute per-instance metrics ───────────────────────────────
    print(f"[Step 3] Computing per-instance metrics...")

    print(f"  Computing class centroid distances...")
    centroid_dist = compute_class_centroid_distances(X_train, y_train)

    print(f"  Computing {KNN_K}-NN label purity...")
    knn_purity = compute_knn_label_purity(X_train, y_train, k=KNN_K)

    # ── Step 4: Build dataframe and save CSV ───────────────────────────────
    df = pd.DataFrame({
        'reconstruction_error': re_scores,
        'centroid_distance': centroid_dist,
        'lr_confidence': confidence,
        'lr_correct': is_correct,
        'knn_purity': knn_purity,
        'true_label': y_train,
        'predicted_label': y_pred,
    })

    csv_path = os.path.join(plots_dir, 'per_instance_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved metrics CSV to {csv_path}")

    # ── Step 5: Compute correlations ───────────────────────────────────────
    print(f"\n[Step 5] Spearman correlations with Reconstruction Error:")
    for col in ['centroid_distance', 'lr_confidence', 'knn_purity']:
        rho, p = spearmanr(re_scores, df[col])
        print(f"  RE vs {col}: rho={rho:.4f}, p={p:.2e}")

    # ── Step 6: Generate plots ─────────────────────────────────────────────
    print(f"[Step 6] Generating plots...")

    sns.set_theme(style='whitegrid', font_scale=1.1)
    fig_kw = dict(figsize=(10, 6), dpi=150)

    # Plot 1: RE distribution colored by LR correct/incorrect
    fig, ax = plt.subplots(**fig_kw)
    sns.kdeplot(re_scores[is_correct == 1], label='LR Correct', color='green',
                fill=True, alpha=0.3, ax=ax)
    sns.kdeplot(re_scores[is_correct == 0], label='LR Incorrect', color='red',
                fill=True, alpha=0.3, ax=ax)
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Density')
    ax.set_title(f'{dataset_name.upper()} — RE Distribution by LR Classification')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'plot1_re_dist_by_correctness.png'))
    plt.close(fig)

    # Plot 2: Scatter RE vs centroid distance
    fig, ax = plt.subplots(**fig_kw)
    scatter = ax.scatter(centroid_dist, re_scores, c=y_train, cmap='tab10',
                         alpha=0.3, s=8, edgecolors='none')
    ax.set_xlabel('Distance to Class Centroid')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title(f'{dataset_name.upper()} — RE vs Class Centroid Distance')
    rho, _ = spearmanr(centroid_dist, re_scores)
    ax.text(0.02, 0.95, f'Spearman ρ = {rho:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'plot2_re_vs_centroid_dist.png'))
    plt.close(fig)

    # Plot 3: Scatter RE vs LR prediction confidence [0, 1]
    fig, ax = plt.subplots(**fig_kw)
    colors = ['red' if c == 0 else 'green' for c in is_correct]
    ax.scatter(confidence, re_scores, c=colors, alpha=0.3, s=8, edgecolors='none')
    ax.set_xlabel('LR Prediction Confidence [0, 1]')
    ax.set_ylabel('Reconstruction Error')
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(f'{dataset_name.upper()} — RE vs LR Prediction Confidence')
    rho, _ = spearmanr(confidence, re_scores)
    ax.text(0.02, 0.95, f'Spearman ρ = {rho:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Incorrect'),
    ]
    ax.legend(handles=legend_elements)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'plot3_re_vs_lr_confidence.png'))
    plt.close(fig)

    # Plot 4: Boxplot of RE by kNN purity quartiles
    fig, ax = plt.subplots(**fig_kw)
    # First qcut without labels to discover actual number of bins (duplicates may reduce it)
    purity_bins = pd.qcut(knn_purity, q=4, duplicates='drop')
    n_bins = len(purity_bins.categories)
    if n_bins == 4:
        bin_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    elif n_bins == 3:
        bin_labels = ['Low', 'Mid', 'High']
    elif n_bins == 2:
        bin_labels = ['Low', 'High']
    else:
        bin_labels = [f'Bin {i+1}' for i in range(n_bins)]
    purity_quartiles = pd.qcut(knn_purity, q=4, labels=bin_labels, duplicates='drop')
    df_box = pd.DataFrame({'RE': re_scores, 'Purity Quartile': purity_quartiles})
    sns.boxplot(data=df_box, x='Purity Quartile', y='RE', palette='viridis', ax=ax)
    ax.set_xlabel(f'{KNN_K}-NN Label Purity Quartile')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title(f'{dataset_name.upper()} — RE by Neighborhood Purity')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'plot4_re_by_purity_quartile.png'))
    plt.close(fig)

    # Plot 5: Violin of RE by correct/incorrect
    fig, ax = plt.subplots(**fig_kw)
    df_violin = pd.DataFrame({
        'RE': re_scores,
        'LR Prediction': np.where(is_correct, 'Correct', 'Incorrect')
    })
    sns.violinplot(data=df_violin, x='LR Prediction', y='RE',
                   palette={'Correct': 'green', 'Incorrect': 'red'},
                   inner='quartile', ax=ax)
    ax.set_ylabel('Reconstruction Error')
    ax.set_title(f'{dataset_name.upper()} — RE Distribution by LR Prediction')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'plot5_re_violin_correctness.png'))
    plt.close(fig)

    print(f"\n  All plots saved to {plots_dir}/")
    print(f"  ✓ {dataset_name.upper()} analysis complete\n")


def main():
    parser = argparse.ArgumentParser(
        description='Exp 1: Reconstruction Error Analysis')
    parser.add_argument('--datasets', type=str, default=','.join(DEFAULT_DATASETS),
                        help='Comma-separated dataset names')
    parser.add_argument('--datasets-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, '..', 'datasets'),
                        help='Path to datasets directory')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, '..', 'results',
                                             'exp1-reconstruction-error'),
                        help='Output directory for results')
    parser.add_argument('--inputrep', type=str, default=DEFAULT_INPUTREP,
                        help='Input representation (tfidf, jina-v5)')
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',')]

    print(f"Exp 1: Reconstruction Error Analysis")
    print(f"  Datasets: {datasets}")
    print(f"  Input rep: {args.inputrep}")
    print(f"  Output: {args.output_dir}")

    for ds in datasets:
        analyze_dataset(ds, args.datasets_dir, args.output_dir, args.inputrep)

    print(f"\n{'='*70}")
    print(f"  All datasets processed. Results in {args.output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
