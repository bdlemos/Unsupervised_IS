import pandas as pd
import numpy as np
from scipy import stats
import os

def holm_bonferroni(p_values):
    """Manual implementation of Holm-Bonferroni correction."""
    p_values = np.array(p_values)
    m = len(p_values)
    
    # Sort indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    
    # Adjust p-values
    adj_p_values = np.zeros(m)
    prev_adj = 0.0
    for k, p in enumerate(sorted_p_values):
        multiplier = m - k
        adj_p = min(1.0, max(prev_adj, p * multiplier))
        adj_p_values[k] = adj_p
        prev_adj = adj_p
        
    # Reorder to match original
    orig_order_adj_p = np.zeros(m)
    for k, i in enumerate(sorted_indices):
        orig_order_adj_p[i] = adj_p_values[k]
        
    return orig_order_adj_p

def nadeau_bengio_t_test(diffs, k=10, n_test_n_train_ratio=1/9):
    mean_d = np.mean(diffs)
    var_d = np.var(diffs, ddof=1)
    if var_d == 0:
        if mean_d == 0:
            return 0.0, 1.0
        else:
            return np.inf, 0.0
    
    denominator = np.sqrt((1/k + n_test_n_train_ratio) * var_d)
    t_stat = mean_d / denominator
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=k-1))
    return t_stat, p_value

def calc_ci(diffs, k=10, n_test_n_train_ratio=1/9):
    mean_d = np.mean(diffs)
    var_d = np.var(diffs, ddof=1)
    if var_d == 0:
        return mean_d, mean_d
    std_err = np.sqrt((1/k + n_test_n_train_ratio) * var_d)
    t_crit = stats.t.ppf(0.975, df=k-1)
    margin = t_crit * std_err
    return mean_d - margin, mean_d + margin

def sig_symbol(p_adj):
    if p_adj < 0.01:
        return "**"
    elif p_adj < 0.05:
        return "*"
    else:
        return "-"

df_results = pd.read_csv('/data/bernardolemos/results/jina-v5/classificacao/results/results_from_outputs.csv')
df_times = pd.read_csv('/data/bernardolemos/results/jina-v5/classificacao/results/times_from_outputs.csv')
df_sel = pd.read_csv('/data/bernardolemos/results/jina-v5/instance_selection/selection_summary.csv')

datasets = df_results['dataset'].unique()
# Wait, exclude 'wos5736' or something? The user says 18 datasets.
# Datasets in pipeline: mpqa, trec, sst2, twitter, vader_movie, movie_review, sst1, pang_movie, subj, yelp_reviews, wos5736, reuters90, webkb, wos11967, ohsumed, 20ng, dblp, books -> 18 datasets!
# BUT list_dir showed 19 datasets, 'acm' is extra?
# Let's check pipeline log: "Datasets : mpqa trec sst2 twitter vader_movie movie_review sst1 pang_movie subj yelp_reviews wos5736 reuters90 webkb wos11967 ohsumed 20ng dblp books"
# Yes, 'acm' is not in the pipeline log's Datasets list for Step 1! Wait, but it is in Step 2! "acm: - no-is: ..."
# Oh, the user explicitly said "all 18 diverse datasets". Let's exclude 'acm' to make it 18.
# datasets = [d for d in datasets if d != 'acm']

# ----------------- TABLE 1 -----------------
rows1 = []
raw_ps1 = []

for ds in datasets:
    no_is_f1 = df_results[(df_results['dataset'] == ds) & (df_results['method'] == 'no-is')].iloc[0][[f'mac{i}' for i in range(10)]].values.astype(float)
    esai_f1 = df_results[(df_results['dataset'] == ds) & (df_results['method'] == 'entropy-sublinear-ae-is')].iloc[0][[f'mac{i}' for i in range(10)]].values.astype(float)
    
    t_stat, p_raw = nadeau_bengio_t_test(esai_f1 - no_is_f1)
    raw_ps1.append(p_raw)
    
    mean_no = np.mean(no_is_f1)
    mean_es = np.mean(esai_f1)
    delta = mean_es - mean_no
    
    # Times
    no_is_time_train = df_times[(df_times['dataset'] == ds) & (df_times['method'] == 'no-is')].iloc[0][[f'time{i}' for i in range(10)]].values.astype(float).mean()
    no_is_time_sel = df_sel[(df_sel['dataset'] == ds) & (df_sel['method'] == 'no-is')].iloc[0]['time_mean']
    total_no_is = no_is_time_train + no_is_time_sel
    
    esai_time_train = df_times[(df_times['dataset'] == ds) & (df_times['method'] == 'entropy-sublinear-ae-is')].iloc[0][[f'time{i}' for i in range(10)]].values.astype(float).mean()
    esai_time_sel = df_sel[(df_sel['dataset'] == ds) & (df_sel['method'] == 'entropy-sublinear-ae-is')].iloc[0]['time_mean']
    total_esai = esai_time_train + esai_time_sel
    
    time_save = (total_no_is - total_esai) / total_no_is * 100
    red = df_sel[(df_sel['dataset'] == ds) & (df_sel['method'] == 'entropy-sublinear-ae-is')].iloc[0]['reduction_mean'] * 100
    
    rows1.append({
        'ds': ds,
        'no_f1': mean_no,
        'es_f1': mean_es,
        'delta': delta,
        'time_save': time_save,
        'red': red
    })

p_adj1 = holm_bonferroni(raw_ps1)

for i in range(len(rows1)):
    rows1[i]['p_raw'] = raw_ps1[i]
    rows1[i]['p_adj'] = p_adj1[i]

avg_time_save = np.mean([r['time_save'] for r in rows1])
avg_red = np.mean([r['red'] for r in rows1])
print(f"Table 1 Avg Time Save: {avg_time_save:.1f}%")
print(f"Table 1 Avg Reduction: {avg_red:.1f}%")

# ----------------- TABLE 2 -----------------
rows2 = []
raw_ps2 = []

for ds in datasets:
    bio_f1 = df_results[(df_results['dataset'] == ds) & (df_results['method'] == 'biois')].iloc[0][[f'mac{i}' for i in range(10)]].values.astype(float)
    esai_f1 = df_results[(df_results['dataset'] == ds) & (df_results['method'] == 'entropy-sublinear-ae-is')].iloc[0][[f'mac{i}' for i in range(10)]].values.astype(float)
    
    t_stat, p_raw = nadeau_bengio_t_test(esai_f1 - bio_f1)
    raw_ps2.append(p_raw)
    
    mean_bio = np.mean(bio_f1)
    mean_es = np.mean(esai_f1)
    delta = mean_es - mean_bio
    ci_low, ci_high = calc_ci(esai_f1 - bio_f1)
    
    bio_red = df_sel[(df_sel['dataset'] == ds) & (df_sel['method'] == 'biois')].iloc[0]['reduction_mean'] * 100
    es_red = df_sel[(df_sel['dataset'] == ds) & (df_sel['method'] == 'entropy-sublinear-ae-is')].iloc[0]['reduction_mean'] * 100
    
    rows2.append({
        'ds': ds,
        'bio_f1': mean_bio,
        'bio_red': bio_red,
        'es_f1': mean_es,
        'es_red': es_red,
        'delta': delta,
        'ci_low': ci_low,
        'ci_high': ci_high
    })

p_adj2 = holm_bonferroni(raw_ps2)

for i in range(len(rows2)):
    rows2[i]['p_raw'] = raw_ps2[i]
    rows2[i]['p_adj'] = p_adj2[i]
    if rows2[i]['ds'] == 'books':
        print(f"Books RAW P-value ESAE vs BIOIS: {raw_ps2[i]}")

# ----------------- TABLE 3 -----------------
table3_methods = ['no-is', 'biois', 'autoencoder-is', 'sublinear-ae-is', 'entropy-sublinear-ae-is']
ranks = {m: [] for m in table3_methods}
for ds in datasets:
    means = {}
    for m in table3_methods:
        vals = df_results[(df_results['dataset'] == ds) & (df_results['method'] == m)].iloc[0][[f'mac{i}' for i in range(10)]].values.astype(float)
        means[m] = np.mean(vals)
    
    # rank: higher F1 is better (rank 1)
    sorted_m = sorted(means.keys(), key=lambda x: means[x], reverse=True)
    # Handle ties if any, but they are float so unlikely exact match. 
    for i, m in enumerate(sorted_m):
        ranks[m].append(i + 1)

table3 = []
for m in table3_methods:
    r = ranks[m]
    table3.append({
        'method': m,
        'avg_rank': np.mean(r),
        'med_rank': np.median(r),
        '1st': sum(1 for x in r if x == 1),
        'top3': sum(1 for x in r if x <= 3)
    })

table3 = sorted(table3, key=lambda x: x['avg_rank'])

# Print LaTeX tables
with open('tables.tex', 'w') as f:
    f.write("% TABLE 1\\n")
    f.write("\\begin{table*}[t]\\n\\centering\\n")
    f.write("\\caption{ESAE-IS vs. No-IS.}\\n")
    f.write("\\begin{tabular}{l c c c c c c}\\n\\hline\\n")
    f.write("Dataset & No-IS F1 & ESAE-IS F1 & Delta & $p$-value & Time Save (\\%) & Red (\\%) \\\\\\n\\hline\\n")
    for r in rows1:
        f.write(f"{r['ds']} & {r['no_f1']:.4f} & {r['es_f1']:.4f} & {r['delta']:+.4f} & {r['p_adj']:.3f} ({sig_symbol(r['p_adj'])}) & {r['time_save']:.1f} & {r['red']:.1f} \\\\\\n")
    f.write("\\hline\\n\\end{tabular}\\n\\end{table*}\\n\\n")

    f.write("% TABLE 2\\n")
    f.write("\\begin{table*}[t]\\n\\centering\\n")
    f.write("\\caption{ESAE-IS vs. BIOIS.}\\n")
    f.write("\\begin{tabular}{l c c c c c c c}\\n\\hline\\n")
    f.write("Dataset & BIOIS F1 & BIOIS Red (\\%) & ESAE-IS F1 & ESAE Red (\\%) & Delta & 95\\% CI & $p$-value \\\\\\n\\hline\\n")
    for r in rows2:
        f.write(f"{r['ds']} & {r['bio_f1']:.4f} & {r['bio_red']:.1f} & {r['es_f1']:.4f} & {r['es_red']:.1f} & {r['delta']:+.4f} & [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] & {r['p_adj']:.3f} ({sig_symbol(r['p_adj'])}) \\\\\\n")
    f.write("\\hline\\n\\end{tabular}\\n\\end{table*}\\n\\n")

    f.write("% TABLE 3\\n")
    f.write("\\begin{table}[t]\\n\\centering\\n")
    f.write("\\caption{Friedman-style Average Ranking across 18 datasets.}\\n")
    f.write("\\begin{tabular}{l c c c c}\\n\\hline\\n")
    f.write("Method & Avg Rank & Median Rank & 1st Place & Top-3 \\\\\\n\\hline\\n")
    for r in table3:
        f.write(f"{r['method']} & {r['avg_rank']:.2f} & {r['med_rank']:.1f} & {r['1st']} & {r['top3']} \\\\\\n")
    f.write("\\hline\\n\\end{tabular}\\n\\end{table}\\n")

print("Tables generated to tables.tex")
