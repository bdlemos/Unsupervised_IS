"""
Statistical Analysis: SublinearAEIS vs No-IS and BIOIS
Uses a Non-Inferiority adaptation of the Corrected Resampled Paired
t-Test (Nadeau & Bengio, 2003) with Holm-Bonferroni correction.

Instead of testing for a mere difference (two-tailed), we test:
    H₀: μ_method - μ_baseline ≤ -δ   (method is inferior)
    H₁: μ_method - μ_baseline > -δ    (method is non-inferior)
with a strict margin δ = 0.02 (2% F1).

The corrected variance accounts for fold non-independence:
    σ² = (1/k + n_test/n_train) * s²
"""

import csv
import numpy as np
from scipy import stats
from collections import defaultdict

# =========================================================================
# Non-Inferiority margin (2% F1 acceptable drop)
# =========================================================================
DELTA = 0.025

# =========================================================================
# Non-Inferiority Corrected Resampled Paired t-Test (Nadeau & Bengio, 2003)
# =========================================================================
def corrected_paired_ttest_ni(scores_a, scores_b, delta=DELTA, k=10, n_train_ratio=0.9):
    """
    One-tailed non-inferiority test using the corrected resampled t-test.
    
    Tests H₀: mean(A - B) ≤ -δ  (A is inferior to B by at least δ)
    vs    H₁: mean(A - B) > -δ  (A is non-inferior to B)
    
    The t-statistic is shifted by δ:
        t = (mean(d) - (-δ)) / sqrt( (1/k + n_test/n_train) * var(d) )
          = (mean(d) + δ) / sqrt( (1/k + n_test/n_train) * var(d) )
    
    Parameters
    ----------
    scores_a, scores_b : array-like of shape (k,)
        Per-fold scores for methods A (proposed) and B (baseline)
    delta : float
        Non-inferiority margin (default: 0.02 = 2% F1)
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
    
    # Corrected variance (Nadeau & Bengio, 2003)
    corrected_var = (1.0 / k + n_test_ratio / n_train_ratio) * var_diff
    
    if corrected_var < 1e-15:
        # If variance is zero, non-inferiority holds trivially if mean_diff > -delta
        p_ni = 0.0 if mean_diff > -delta else 1.0
        return 0.0, p_ni, mean_diff, mean_diff, mean_diff
    
    # One-tailed non-inferiority t-statistic: shift by δ
    t_stat = (mean_diff + delta) / np.sqrt(corrected_var)
    
    # Degrees of freedom = k - 1
    df = k - 1
    # One-tailed p-value (upper tail: reject H₀ when t is large)
    p_value = stats.t.sf(t_stat, df)
    
    # 95% Confidence Interval (still two-sided for reporting)
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

import os
from pathlib import Path

RESULTS_BASE = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parent.parent / "results" / "jina-v5"))
results = parse_csv(RESULTS_BASE / "classificacao" / "results" / "results_from_outputs.csv")
times = parse_times_csv(RESULTS_BASE / "classificacao" / "results" / "times_from_outputs.csv")

method_names = {'entropy-sublinear-ae-is': 'ESAE-IS', 'biois': 'BIOIS', 'no-is': 'No-IS', 'autoencoder-is': 'AE-IS', 'gmm-is': 'GMM-IS', 'adaptive-perplexity': 'AP-IS', 'adaptive-cluster-is': 'AC-IS', 'sublinear-ae-is': 'SAE-IS'}

datasets = sorted(set(k[0] for k in results.keys()))


ALPHA = 0.05

METHOD_TO_COMPARE = 'entropy-sublinear-ae-is'

# Only datasets where sublinear-ae-is was run
datasets_sub = [ds for ds in datasets if (ds, METHOD_TO_COMPARE) in results]
# =========================================================================
# COMPARISON 1: Method_to_compare vs No-IS (Non-Inferiority Test)
# =========================================================================
print("=" * 110)
print(f"COMPARISON 1: {method_names[METHOD_TO_COMPARE]} vs {method_names['no-is']} — Non-Inferiority Test (δ={DELTA})")
print("One-tailed Corrected Resampled Paired t-Test (Nadeau & Bengio, 2003) + Holm-Bonferroni")
print(f"H₀: ESAE-IS is inferior to No-IS by ≥ {DELTA} F1  |  H₁: ESAE-IS is non-inferior")
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
        
        t_stat, p_val, mean_diff, ci_lo, ci_hi = corrected_paired_ttest_ni(sub_folds, nois_folds)
        
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

print(f"\n{'Dataset':<16} | {method_names[METHOD_TO_COMPARE]:<10} | {method_names['no-is']:<10} | {'Δ F1':<8} | {'95% CI':<20} | {'t-stat':<8} | {'p-raw':<8} | {'p-adj':<8} | {'Non-Inf?':<10} | {'Time Save':<10}")
print("-" * 140)

non_inf_count = 0
fail_count = 0

for i, r in enumerate(test_results_nois):
    p_adj = adj_pvals_nois[i]
    
    if p_adj < ALPHA:
        sig = "✅ Non-Inf."
        non_inf_count += 1
    else:
        sig = "❌ Inferior"
        fail_count += 1
    
    print(f"{r['ds']:<16} | {r['sub_mean']:.4f}     | {r['nois_mean']:.4f}     | {r['mean_diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['t_stat']:+6.2f}   | {r['p_val']:.4f}   | {p_adj:.4f}   | {sig:<10} | {r['time_save_pct']:+.1f}%")

print("-" * 140)
print(f"Non-Inferiority established: {non_inf_count}/{non_inf_count + fail_count} datasets (α={ALPHA}, δ={DELTA})")
print(f"Average Δ F1: {np.mean([r['mean_diff'] for r in test_results_nois]):+.4f}")
print(f"Average Time Saved: {np.mean([r['time_save_pct'] for r in test_results_nois]):.1f}%")


# =========================================================================
# COMPARISON 2: SELECTED_METHOD vs BIOIS
# =========================================================================
print("\n\n" + "=" * 110)
print(f"COMPARISON 2: {method_names[METHOD_TO_COMPARE]} vs {method_names['biois']} (Supervised SOTA) — Non-Inferiority Test (δ={DELTA})")
print("One-tailed Corrected Resampled Paired t-Test (Nadeau & Bengio, 2003) + Holm-Bonferroni")
print(f"H₀: ESAE-IS is inferior to BIOIS by ≥ {DELTA} F1  |  H₁: ESAE-IS is non-inferior")
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
        
        t_stat, p_val, mean_diff, ci_lo, ci_hi = corrected_paired_ttest_ni(sub_folds, bio_folds)
        
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

print(f"\n{'Dataset':<16} | {method_names[METHOD_TO_COMPARE]:<10} | {method_names['biois']:<10} | {'Δ F1':<8} | {'95% CI':<20} | {'t-stat':<8} | {'p-raw':<8} | {'p-adj':<8} | {'Non-Inf?':<10} | {'Red Sub':<8} | {'Red BIO':<8}")
print("-" * 150)

non_inf_count2 = 0
fail_count2 = 0

for i, r in enumerate(test_results_biois):
    p_adj = adj_pvals_biois[i]
    
    if p_adj < ALPHA:
        sig = "✅ Non-Inf."
        non_inf_count2 += 1
    else:
        sig = "❌ Inferior"
        fail_count2 += 1
    
    print(f"{r['ds']:<16} | {r['sub_mean']:.4f}     | {r['bio_mean']:.4f}     | {r['mean_diff']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['t_stat']:+6.2f}   | {r['p_val']:.4f}   | {p_adj:.4f}   | {sig:<10} | {r['sub_red']:.1%}    | {r['bio_red']:.1%}")

print("-" * 150)
print(f"Non-Inferiority established: {non_inf_count2}/{non_inf_count2 + fail_count2} datasets (α={ALPHA}, δ={DELTA})")
print(f"Average Δ F1: {np.mean([r['mean_diff'] for r in test_results_biois]):+.4f}")


# =========================================================================
# COMPARISON 3: Selected Method vs All Other IS Methods (Rankings)
#   — Friedman Test + Nemenyi Post-Hoc Critical Difference
# =========================================================================
print("\n\n" + "=" * 110)
print("COMPARISON 3: Friedman Rank Test + Nemenyi Post-Hoc (lower rank = better)")
print("=" * 110)

methods_of_interest = ['no-is', 'biois', 'autoencoder-is', 'gmm-is',
                       'adaptive-cluster-is', 'adaptive-perplexity', 'sublinear-ae-is', 'entropy-sublinear-ae-is']

# --- Filter to methods present across ALL datasets_sub -----------------
methods_present = [m for m in methods_of_interest
                   if all((ds, m) in results for ds in datasets_sub)]

N = len(datasets_sub)   # number of datasets (blocks)
k = len(methods_present) # number of methods (treatments)

print(f"\nMethods with data on all {N} datasets: {k}")
for m in methods_present:
    print(f"  • {method_names.get(m, m)}")

# --- Build score matrix & rank matrix (N × k) -------------------------
score_matrix = np.zeros((N, k))
for i, ds in enumerate(datasets_sub):
    for j, m in enumerate(methods_present):
        score_matrix[i, j] = results[(ds, m)]['mean']

# Rank per dataset: highest F1 → rank 1 (use scipy for tied-rank handling)
rank_matrix = np.zeros((N, k))
for i in range(N):
    rank_matrix[i, :] = stats.rankdata(-score_matrix[i, :])

avg_ranks = np.mean(rank_matrix, axis=0)
med_ranks = np.median(rank_matrix, axis=0)

# --- Descriptive Rank Table --------------------------------------------
print(f"\n{'Method':<25} | {'Avg Rank':<10} | {'Median Rank':<12} | {'Best (#1)':<10} | {'Top-3':<8}")
print("-" * 80)

sorted_idx = np.argsort(avg_ranks)
for idx in sorted_idx:
    m = methods_present[idx]
    best = int(np.sum(rank_matrix[:, idx] == 1))
    top3 = int(np.sum(rank_matrix[:, idx] <= 3))
    marker = " ← PROPOSED" if m == METHOD_TO_COMPARE else ""
    print(f"{method_names.get(m, m):<25} | {avg_ranks[idx]:<10.2f} | {med_ranks[idx]:<12.1f} | {best:<10} | {top3:<8}{marker}")

# --- Friedman Test (χ² omnibus) ----------------------------------------
friedman_stat, friedman_p = stats.friedmanchisquare(
    *[score_matrix[:, j] for j in range(k)]
)

# Iman-Davenport F correction (more powerful, recommended by Demšar 2006)
chi2_F = (12 * N / (k * (k + 1))) * np.sum((avg_ranks - (k + 1) / 2.0) ** 2)
F_F = ((N - 1) * chi2_F) / (N * (k - 1) - chi2_F)
df1 = k - 1
df2 = (k - 1) * (N - 1)
p_iman = stats.f.sf(F_F, df1, df2)

print(f"\n{'─' * 80}")
print(f"Friedman χ²  = {friedman_stat:.4f}  (df = {k-1}, p = {friedman_p:.4e})")
print(f"Iman-Davenport F = {F_F:.4f}  (df₁ = {df1}, df₂ = {df2}, p = {p_iman:.4e})")
if friedman_p < ALPHA:
    print(f"→ Significant rank differences exist (p < {ALPHA}); proceeding to Nemenyi post-hoc.")
else:
    print(f"→ No significant rank differences detected (p ≥ {ALPHA}); post-hoc shown for completeness.")

# --- Nemenyi Post-Hoc Critical Difference (Demšar, 2006) ---------------
# Critical values q_α for the Studentized Range / √2 (Nemenyi test)
# Source: Demšar (2006), Table 5a — q_α values for two-tailed comparison
NEMENYI_Q_005 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}

if k in NEMENYI_Q_005:
    q_alpha = NEMENYI_Q_005[k]
else:
    # Fallback: approximate using Tukey q / sqrt(2) from scipy
    from scipy.stats import studentized_range
    q_alpha = studentized_range.ppf(0.95, k, df=np.inf) / np.sqrt(2)

CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))

print(f"\nNemenyi Critical Difference (α = {ALPHA}): CD = {CD:.4f}  (q_{ALPHA} = {q_alpha:.3f}, k = {k}, N = {N})")
print(f"Two methods differ significantly if |R̄_i − R̄_j| ≥ {CD:.4f}")

# --- Pairwise Nemenyi comparisons from the proposed method -------------
esae_idx = methods_present.index(METHOD_TO_COMPARE)
esae_rank = avg_ranks[esae_idx]

print(f"\n{'Comparison':<35} | {'ΔRank':<8} | {'CD':<8} | {'Significant?':<15}")
print("-" * 80)

for j in sorted_idx:
    m = methods_present[j]
    if m == METHOD_TO_COMPARE:
        continue
    diff = abs(avg_ranks[j] - esae_rank)
    sig = "YES (p < 0.05)" if diff >= CD else "no"
    direction = "worse" if avg_ranks[j] > esae_rank else "better"
    print(f"{method_names.get(METHOD_TO_COMPARE, METHOD_TO_COMPARE)} vs {method_names.get(m, m):<20} | {diff:<8.4f} | {CD:<8.4f} | {sig}")

# --- Full pairwise Nemenyi matrix (for completeness) -------------------
print(f"\nFull pairwise rank differences (CD = {CD:.4f}, * = significant at α = {ALPHA}):")
header = f"{'':20}" + "".join(f"{method_names.get(methods_present[j], methods_present[j]):>10}" for j in sorted_idx)
print(header)
for i_pos, i in enumerate(sorted_idx):
    row_label = method_names.get(methods_present[i], methods_present[i])
    row_str = f"{row_label:20}"
    for j in sorted_idx:
        if i == j:
            row_str += f"{'—':>10}"
        else:
            diff = abs(avg_ranks[i] - avg_ranks[j])
            marker = "*" if diff >= CD else ""
            row_str += f"{diff:>8.2f}{marker:>2}"
    print(row_str)


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
            vs_nois = f"{diff:+.4f} NI p={p_adj:.3f}"
        else:
            vs_nois = f"{diff:+.4f} ✗  p={p_adj:.3f}"
    
    vs_bio = ""
    if bio_idx:
        p_adj = adj_pvals_biois[bio_idx[0]]
        diff = sub_f1 - bio_f1
        if p_adj < ALPHA:
            vs_bio = f"{diff:+.4f} NI p={p_adj:.3f}"
        else:
            vs_bio = f"{diff:+.4f} ✗  p={p_adj:.3f}"
    
    time_save = ""
    if nois_idx:
        time_save = f"{test_results_nois[nois_idx[0]]['time_save_pct']:+.1f}%"
    
    print(f"{ds:<16} | {nois_f1:.4f}     | {bio_f1:.4f}     | {sub_f1:.4f}     | {red:.1%}  | {vs_nois:<14} | {vs_bio:<14} | {time_save}")
