#!/usr/bin/env python3
"""
Evaluate the iSel algorithm on multiple imbalanced datasets.

python unsupervised-is/analysis/evaluate_is.py

Results are saved to
analysis/plots/<dataset>/<plot_type>.png
and summarized in
analysis/summary_results.md

"""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main.python.utils.general import get_data
from src.main.python.iSel.entropy_sublinear_ae_is import EntropySublinearAEIS

def evaluate_dataset(dataset_name, clustering_method, report_file):
    datasets_dir = os.environ.get("DATASETS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets")))
    dataset_path = os.path.join(datasets_dir, dataset_name, "jina-v5")
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots', dataset_name, clustering_method)
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading data...")
    X_train, y_train, X_test, y_test, n_classes = get_data(dataset_path, f=0)

    # Original class distribution
    orig_counts = Counter(y_train)
    classes = sorted(orig_counts.keys())

    print(f"Original instances: {len(y_train)}")
    for c in classes:
        print(f"Class {c}: {orig_counts[c]} instances")

    majority_class = max(orig_counts, key=orig_counts.get)
    minority_class = min(orig_counts, key=orig_counts.get)

    print(f"\nRunning EntropySublinearAEIS with clustering: {clustering_method}...")
    selector = EntropySublinearAEIS(
        r_max=0.50,            # max reduction
        alpha=15.0,            # entropy sensitivity (high due to K-Means uniform distribution)
        n_clusters=200,        # micro-clusters
        gamma=0.5,             # square-root sublinear
        ae_epochs=50,          # autoencoder epochs
        random_state=13,
        clustering_method=clustering_method
    )

    X_reduced, y_reduced = selector.fit_transform(X_train, y_train) if hasattr(selector, 'fit_transform') else selector.fit(X_train, y_train).X_, selector.y_

    print("\nCalculating metrics...")
    reduced_counts = Counter(y_reduced)

    total_reduction = 1.0 - len(y_reduced) / len(y_train)
    print(f"Total reduction: {total_reduction:.2%}")

    retention_rates = {}
    print("\nRetention rate per class:")
    for c in classes:
        ret_rate = reduced_counts.get(c, 0) / orig_counts[c]
        retention_rates[c] = ret_rate
        print(f"Class {c}: {ret_rate:.2%} ({reduced_counts.get(c, 0)} / {orig_counts[c]})")

    min_retention = retention_rates[minority_class]
    maj_retention = retention_rates[majority_class]
    protection_ratio = min_retention / maj_retention if maj_retention > 0 else 0

    print(f"\nProtection Ratio (Minority {minority_class} / Majority {majority_class}): {protection_ratio:.4f}")

    if hasattr(selector, 'local_distances_'):
        print("\nMean local KNN distance per class:")
        for c in classes:
            class_dists = selector.local_distances_[y_train == c]
            print(f"Class {c}: {np.mean(class_dists):.4f}")

    print("\nPlotting results...")

    # Plot 1: Class Distribution Before/After
    plt.figure(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.35

    orig_vals = [orig_counts[c] for c in classes]
    red_vals = [reduced_counts.get(c, 0) for c in classes]

    plt.bar(x - width/2, orig_vals, width, label='Original')
    plt.bar(x + width/2, red_vals, width, label='Reduced')

    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title('Class Distribution Before and After Selection')
    plt.xticks(x, classes)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'))
    plt.close()

    # Plot 2: Retention Rates
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, [retention_rates[c] for c in classes], color='skyblue')

    # Highlight minority and majority
    bars[classes.index(minority_class)].set_color('green')  # minority
    bars[classes.index(majority_class)].set_color('red')    # majority

    plt.axhline(y=1.0 - total_reduction, color='r', linestyle='-', alpha=0.3, label='Global Average Retention')

    plt.xlabel('Class')
    plt.ylabel('Retention Rate')
    plt.title(f'Retention Rate by Class (Protection Ratio: {protection_ratio:.2f})')
    plt.xticks(classes)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'retention_rates.png'))
    plt.close()

    # Plot 3: Sparsity Score vs Retention if data available
    if hasattr(selector, 'local_distances_') and hasattr(selector, 'mask'):
        plt.figure(figsize=(10, 6))

        dists = selector.local_distances_
        mask = selector.mask

        # Plot distributions for kept vs removed
        sns.kdeplot(dists[mask], label='Kept', color='green', fill=True, alpha=0.3)
        sns.kdeplot(dists[~mask], label='Removed', color='red', fill=True, alpha=0.3)

        plt.axvline(x=selector.guard_threshold_, color='k', linestyle='--', label='Global Guard Threshold')

        plt.xlabel('Local KNN Distance (Sparsity Proxy)')
        plt.ylabel('Density')
        plt.title('Sparsity Distribution: Kept vs Removed')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'sparsity_distribution.png'))
        plt.close()

    summary_text = f"""
## Dataset: {dataset_name.upper()} | Clustering: {clustering_method.upper()}
- **Total Reduction:** {total_reduction:.2%}
- **Minority Retention (Class {minority_class}):** {min_retention:.2%} ({reduced_counts.get(minority_class, 0)} / {orig_counts[minority_class]})
- **Majority Retention (Class {majority_class}):** {maj_retention:.2%} ({reduced_counts.get(majority_class, 0)} / {orig_counts[majority_class]})
- **Protection Ratio:** **{protection_ratio:.4f}** (Goal > 1.0)

![Class Distribution](./plots/{dataset_name}/{clustering_method}/class_distribution.png)
![Retention Rates](./plots/{dataset_name}/{clustering_method}/retention_rates.png)

"""
    with open(report_file, 'a') as f:
        f.write(summary_text)

    print(f"\n================ SUMMARY for {dataset_name} ({clustering_method}) ================")
    print(f"Total Reduction: {total_reduction:.2%}")
    print(f"Minority Retention (Class {minority_class}): {min_retention:.2%}")
    print(f"Majority Retention (Class {majority_class}): {maj_retention:.2%}")
    print(f"Protection Ratio: {protection_ratio:.4f} (Goal > 1.0)")
    print("=====================================================\n")

def main():
    datasets = ['trec', 'ohsumed'] # DBScan can be memory intensive, run on smaller ones
    clustering_methods = ['dbscan']
    report_file = os.path.join(os.path.dirname(__file__), 'summary_results.md')

    with open(report_file, 'w') as f:
        f.write("# EntropySublinearAEIS Evaluation Report: Clustering Strategies\n\n")
        f.write("Evaluation of SublinearAEIS using different clustering methods.\n\n")

    for ds in datasets:
        for clustering_method in clustering_methods:
            print(f"\n\n>>> EVALUATING DATASET: {ds.upper()} | CLUSTERING: {clustering_method.upper()} <<<")
            evaluate_dataset(ds, clustering_method, report_file)

if __name__ == "__main__":
    main()
