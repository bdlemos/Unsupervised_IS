"""
Statistical Analysis: SublinearAEIS vs No-IS and BIOIS
Uses the Corrected Resampled Paired t-Test (Nadeau & Bengio, 2003)
with Holm-Bonferroni correction for multiple comparisons.

The corrected t-test accounts for the non-independence of folds
in k-fold CV by adjusting the variance estimate:
    σ² = (1/k + n_test/n_train) * s²
where s² is the sample variance of the k fold differences.
"""

import csv
import numpy as np
from scipy import stats
from collections import defaultdict

# =========================================================================
# Corrected Resampled Paired t-Test (Nadeau & Bengio, 2003)
# =========================================================================
def corrected_paired_ttest(scores_a, scores_b, k=10, n_train_ratio=0.9):
    """
    Corrected resampled t-test for k-fold cross-validation.
    
    The standard paired t-test underestimates variance because CV folds
    overlap in training data. Nadeau & Bengio (2003) proposed a correction:
    
        t = mean(d) / sqrt( (1/k + n_test/n_train) * var(d) )
    
    Parameters
    ----------
    scores_a, scores_b : array-like of shape (k,)
        Per-fold scores for methods A and B
    k : int
        Number of folds
    n_train_ratio : float
        Fraction of data used for training (0.9 for 10-fold CV)
    
    Returns
    -------
    t_stat, p_value, mean_diff, ci_low, ci_high
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    diffs = scores_a - scores_b
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)
    
    n_test_ratio = 1.0 - n_train_ratio
    
    # Corrected variance
    corrected_var = (1.0 / k + n_test_ratio / n_train_ratio) * var_diff
    
    if corrected_var < 1e-15:
        return 0.0, 1.0, mean_diff, mean_diff, mean_diff
    
    t_stat = mean_diff / np.sqrt(corrected_var)
    
    # Degrees of freedom = k - 1
    df = k - 1
    p_value = 2 * stats.t.sf(abs(t_stat), df)
    
    # 95% Confidence Interval
    t_crit = stats.t.ppf(0.975, df)
    margin = t_crit * np.sqrt(corrected_var)
    ci_low = mean_diff - margin
    ci_high = mean_diff + margin
    
    return t_stat, p_value, mean_diff, ci_low, ci_high

def holm_bonferroni(p_values):
    """
    Holm-Bonferroni step-down correction for multiple comparisons.
    Returns adjusted p-values.
    """
    n = len(p_values)
    indices = np.argsort(p_values)
    adjusted = np.zeros(n)
    
    for rank, idx in enumerate(indices):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    
    # Enforce monotonicity (step-down)
    for i in range(1, n):
        idx = indices[i]
        prev_idx = indices[i - 1]
        if adjusted[idx] < adjusted[prev_idx]:
            adjusted[idx] = adjusted[prev_idx]
    
    return adjusted

# =========================================================================
# Data Loading
# =========================================================================
def parse_csv(filepath):
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['dataset'].strip(), row['method'].strip())
            fold_scores = []
            for i in range(10):
                col = f'mac{i}'
                if col in row and row[col].strip():
                    fold_scores.append(float(row[col]))
            results[key] = {
                'reduction': float(row['reduction']),
                'folds': np.array(fold_scores),
                'mean': np.mean(fold_scores),
                'std': np.std(fold_scores),
            }
    return results

def parse_times_csv(filepath):
    results = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['dataset'].strip(), row['method'].strip())
            fold_times = []
            for i in range(10):
                col = f'time{i}'
                if col in row and row[col].strip():
                    fold_times.append(float(row[col]))
            results[key] = {
                'folds': np.array(fold_times),
                'mean': np.mean(fold_times),
                'std': np.std(fold_times),
            }
    return results

results = parse_csv('/data/bernardolemos/results/jina-v5/classificacao/results/results_from_outputs.csv')
times = parse_times_csv('/data/bernardolemos/results/jina-v5/classificacao/results/times_from_outputs.csv')

method_names = {'entropy-sublinear-ae-is': 'ESAE-IS', 'biois': 'BIOIS', 'no-is': 'No-IS', 'autoencoder-is': 'AE-IS', 'gmm-is': 'GMM-IS', 'adaptive-perplexity': 'AP-IS', 'adaptive-cluster-is': 'AC-IS', 'sublinear-ae-is': 'SAE-IS'}

datasets = sorted(set(k[0] for k in results.keys()))


ALPHA = 0.05

METHOD_TO_COMPARE = 'entropy-sublinear-ae-is'

# Only datasets where sublinear-ae-is was run
datasets_sub = [ds for ds in datasets if (ds, METHOD_TO_COMPARE) in results]
# =========================================================================
# COMPARISON 1: Method_to_compare vs No-IS
# =========================================================================
print("=" * 110)
print(f"COMPARISON 1: {method_names[METHOD_TO_COMPARE]} vs {method_names['no-is']}")
print("Corrected Resampled Paired t-Test (Nadeau & Bengio, 2003) + Holm-Bonferroni Correction")
print("=" * 110)

raw_pvals_nois = []
test_results_nois = []


for ds in datasets_sub:
    sub_key = (ds, METHOD_TO_COMPARE)
    nois_key = (ds, 'no-is')
    
    if sub_key in results and nois_key in results:
        sub_folds = results[sub_key]['folds']
        nois_folds = results[nois_key]['folds']

        
        if len(sub_folds) != len(nois_folds):
            print(f"Skipping {ds}: {len(sub_folds)} folds for {METHOD_TO_COMPARE} vs {len(nois_folds)} folds for no-is")
            continue
        
        t_stat, p_val, mean_diff, ci_lo, ci_hi = corrected_paired_ttest(sub_folds, nois_folds)
        
        # Time comparison (already includes IS time)
        time_sub = times.get(sub_key, {}).get('mean', 0)
        time_nois = times.get(nois_key, {}).get('mean', 0)
        time_save_pct = ((time_nois - time_sub) / time_nois * 100) if time_nois > 0 else 0
        
        raw_pvals_nois.append(p_val)
        test_results_nois.append({
            'ds': ds,
            'sub_mean': results[sub_key]['mean'],
            'nois_mean': results[nois_key]['mean'],
            'mean_diff': mean_diff,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            't_stat': t_stat,
            'p_val': p_val,
            'time_sub': time_sub,
            'time_nois': time_nois,
            'time_save_pct': time_save_pct,
            'reduction': results[sub_key]['reduction'],
        })

# Holm-Bonferroni correction
adj_pvals_nois = holm_bonferroni(np.array(raw_pvals_nois))

print(f"\n{'Dataset':<16} | {method_names[METHOD_TO_COMPARE]:<10} | {method_names['no-is']:<10} | {'Δ F1':<8} | {'95% CI':<20} | {'t-stat':<8} | {'p-raw':<8} | {'p-adj':<8} | {'Sig?':<6} | {'Time Save':<10}")
print("-" * 130)

wins = losses = ties = 0
sig_wins = sig_losses = 0

for i, r in enumerate(test_results_nois):
    p_adj = adj_pvals_nois[i]
    
    if p_adj < ALPHA:
        if r['mean_diff'] > 0:
            sig = "✅ WIN"
            sig_wins += 1
            wins += 1
        else:
            sig = "❌ LOSS"
            sig_losses += 1
            losses += 1
    else:
        sig = "— n.s."
        ties += 1
    
    print(f"{r['ds']:<16} | {r['sub_mean']:.4f}     | {r['nois_mean']:.4f}     | {r['mean_diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['t_stat']:+6.2f}   | {r['p_val']:.4f}   | {p_adj:.4f}   | {sig:<6} | {r['time_save_pct']:+.1f}%")

print("-" * 130)
print(f"Statistically Significant: {sig_wins} wins, {sig_losses} losses, {ties} not significant (α={ALPHA})")
print(f"Average Δ F1: {np.mean([r['mean_diff'] for r in test_results_nois]):+.4f}")
print(f"Average Time Saved: {np.mean([r['time_save_pct'] for r in test_results_nois]):.1f}%")


# =========================================================================
# COMPARISON 2: SELECTED_METHOD vs BIOIS
# =========================================================================
print("\n\n" + "=" * 110)
print(f"COMPARISON 2: {method_names[METHOD_TO_COMPARE]} vs {method_names['biois']} (Supervised SOTA)")
print("Corrected Resampled Paired t-Test (Nadeau & Bengio, 2003) + Holm-Bonferroni Correction")
print("=" * 110)

raw_pvals_biois = []
test_results_biois = []

for ds in datasets_sub:
    sub_key = (ds, METHOD_TO_COMPARE)
    bio_key = (ds, 'biois')
    
    if sub_key in results and bio_key in results:
        sub_folds = results[sub_key]['folds']
        bio_folds = results[bio_key]['folds']

        if len(sub_folds) != len(bio_folds):
            print(f"Skipping {ds}: {len(sub_folds)} folds for {METHOD_TO_COMPARE} vs {len(bio_folds)} folds for biois")
            continue
        
        t_stat, p_val, mean_diff, ci_lo, ci_hi = corrected_paired_ttest(sub_folds, bio_folds)
        
        # Time
        time_sub = times.get(sub_key, {}).get('mean', 0)
        time_bio = times.get(bio_key, {}).get('mean', 0)
        
        raw_pvals_biois.append(p_val)
        test_results_biois.append({
            'ds': ds,
            'sub_mean': results[sub_key]['mean'],
            'bio_mean': results[bio_key]['mean'],
            'bio_red': results[bio_key]['reduction'],
            'sub_red': results[sub_key]['reduction'],
            'mean_diff': mean_diff,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            't_stat': t_stat,
            'p_val': p_val,
            'time_sub': time_sub,
            'time_bio': time_bio,
        })

adj_pvals_biois = holm_bonferroni(np.array(raw_pvals_biois))

print(f"\n{'Dataset':<16} | {method_names[METHOD_TO_COMPARE]:<10} | {method_names['biois']:<10} | {'Δ F1':<8} | {'95% CI':<20} | {'t-stat':<8} | {'p-raw':<8} | {'p-adj':<8} | {'Sig?':<6} | {'Red Sub':<8} | {'Red BIO':<8}")
print("-" * 140)

wins2 = losses2 = ties2 = 0
sig_wins2 = sig_losses2 = 0

for i, r in enumerate(test_results_biois):
    p_adj = adj_pvals_biois[i]
    
    if p_adj < ALPHA:
        if r['mean_diff'] > 0:
            sig = "✅ WIN"
            sig_wins2 += 1
            wins2 += 1
        else:
            sig = "❌ LOSS"
            sig_losses2 += 1
            losses2 += 1
    else:
        sig = "— n.s."
        ties2 += 1
    
    print(f"{r['ds']:<16} | {r['sub_mean']:.4f}     | {r['bio_mean']:.4f}     | {r['mean_diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['t_stat']:+6.2f}   | {r['p_val']:.4f}   | {p_adj:.4f}   | {sig:<6} | {r['sub_red']:.1%}    | {r['bio_red']:.1%}")

print("-" * 140)
print(f"Statistically Significant: {sig_wins2} wins, {sig_losses2} losses, {ties2} not significant (α={ALPHA})")
print(f"Average Δ F1: {np.mean([r['mean_diff'] for r in test_results_biois]):+.4f}")


# =========================================================================
# COMPARISON 3: Selected Method vs All Other IS Methods (Rankings)
# =========================================================================
print("\n\n" + "=" * 110)
print("COMPARISON 3: Average Rank across datasets (lower = better)")
print("=" * 110)

methods_of_interest = ['no-is', 'biois', 'autoencoder-is', 'gmm-is', 
                       'adaptive-cluster-is', 'adaptive-perplexity', 'sublinear-ae-is', 'entropy-sublinear-ae-is']

ranks = defaultdict(list)

for ds in datasets_sub:
    ds_scores = []
    for m in methods_of_interest:
        key = (ds, m)
        if key in results:
            ds_scores.append((m, results[key]['mean']))
    
    ds_scores.sort(key=lambda x: -x[1])  # Sort descending by F1
    
    for rank, (m, _) in enumerate(ds_scores, 1):
        ranks[m].append(rank)

print(f"\n{'Method':<25} | {'Avg Rank':<10} | {'Median Rank':<12} | {'Best (#1)':<10} | {'Top-3':<8}")
print("-" * 80)

method_ranks = []
for m in methods_of_interest:
    if m in ranks:
        avg_r = np.mean(ranks[m])
        med_r = np.median(ranks[m])
        best = sum(1 for r in ranks[m] if r == 1)
        top3 = sum(1 for r in ranks[m] if r <= 3)
        method_ranks.append((m, avg_r, med_r, best, top3))

method_ranks.sort(key=lambda x: x[1])

for m, avg_r, med_r, best, top3 in method_ranks:
    marker = " ← PROPOSED" if m == METHOD_TO_COMPARE else ""
    print(f"{m:<25} | {avg_r:<10.2f} | {med_r:<12.1f} | {best:<10} | {top3:<8}{marker}")


# =========================================================================
# SUMMARY for Paper
# =========================================================================
print("\n\n" + "=" * 110)
print("SUMMARY TABLE FOR PAPER (copy-paste ready)")
print("=" * 110)

print(f"\n{'Dataset':<16} | {method_names['no-is']:<10} | {method_names['biois']:<10} | {method_names[METHOD_TO_COMPARE]:<10} | {'Red%':<6} | {'vs No-IS':<14} | {'vs BIOIS':<14} | {'Time Save':<10}")
print("-" * 120)

for i, ds in enumerate(datasets_sub):
    sub_key = (ds, METHOD_TO_COMPARE)
    nois_key = (ds, 'no-is')
    bio_key = (ds, 'biois')
    
    sub_f1 = results.get(sub_key, {}).get('mean', 0)
    nois_f1 = results.get(nois_key, {}).get('mean', 0)
    bio_f1 = results.get(bio_key, {}).get('mean', 0)
    red = results.get(sub_key, {}).get('reduction', 0)
    
    # Find the adjusted p-values
    nois_idx = [j for j, r in enumerate(test_results_nois) if r['ds'] == ds]
    bio_idx = [j for j, r in enumerate(test_results_biois) if r['ds'] == ds]
    
    vs_nois = ""
    if nois_idx:
        p_adj = adj_pvals_nois[nois_idx[0]]
        diff = sub_f1 - nois_f1
        if p_adj < ALPHA:
            vs_nois = f"{diff:+.4f} {'✅' if diff > 0 else '❌'} p={p_adj:.3f}"
        else:
            vs_nois = f"{diff:+.4f} ≈  p={p_adj:.3f}"
    
    vs_bio = ""
    if bio_idx:
        p_adj = adj_pvals_biois[bio_idx[0]]
        diff = sub_f1 - bio_f1
        if p_adj < ALPHA:
            vs_bio = f"{diff:+.4f} {'✅' if diff > 0 else '❌'} p={p_adj:.3f}"
        else:
            vs_bio = f"{diff:+.4f} ≈  p={p_adj:.3f}"
    
    time_save = ""
    if nois_idx:
        time_save = f"{test_results_nois[nois_idx[0]]['time_save_pct']:+.1f}%"
    
    print(f"{ds:<16} | {nois_f1:.4f}     | {bio_f1:.4f}     | {sub_f1:.4f}     | {red:.1%}  | {vs_nois:<14} | {vs_bio:<14} | {time_save}")
