#!/usr/bin/env python3
"""
Consolidates all budget curve experiments into a unified Master Comparison Report.
Includes:
  - Random
  - Cluster-Random (Ablation)
  - Typ-Cover (Density Coverage)
  - SAE-IS (Original - Reconstruction Error)
  - SAE-IS (Clean - Inverse Score)
  - Diffusion-Cover (Manifold PageRank + Sublinear Coverage)
  - Diffusion-Facility (Submodular Facility Location)

Generates:
  results/proxy-budget-curve/MASTER_COMPARISON.md
"""

import os, sys
import pandas as pd
import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'results', 'proxy-budget-curve')
OUT_FILE = os.path.join(RES, 'MASTER_COMPARISON.md')


def md(df, floatfmt: str = '.4f', index: bool = True) -> str:
    d = df.reset_index() if index else df
    fmt = lambda v: format(v, floatfmt) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ('-' if pd.isna(v) else str(v))
    head = '| ' + ' | '.join(map(str, d.columns)) + ' |'
    sep = '| ' + ' | '.join('---' for _ in d.columns) + ' |'
    body = ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in d.itertuples(index=False)]
    return '\n'.join([head, sep] + body)


def main():
    dfs = []
    
    # 1. Base Budget Curve
    f_base = os.path.join(RES, 'budget_curve', 'per_fold.csv')
    if os.path.exists(f_base):
        dfs.append(pd.read_csv(f_base))
        
    # 2. SAE Budget Curve
    f_sae = os.path.join(RES, 'sae_budget', 'per_fold.csv')
    if os.path.exists(f_sae):
        dfs.append(pd.read_csv(f_sae))
        
    # 3. Diffusion Budget Curve
    f_diff = os.path.join(RES, 'diffusion_budget', 'per_fold.csv')
    if os.path.exists(f_diff):
        dfs.append(pd.read_csv(f_diff))
        
    if not dfs:
        print("No results found.")
        return
        
    all_df = pd.concat(dfs, ignore_index=True)
    
    # Filter to rates 90, 95, 98, 99
    sub = all_df[all_df.rate.isin([90, 95, 98, 99])].copy()
    
    # Pivot table of means
    means = sub.groupby(['dataset', 'rate', 'method']).macro.mean().unstack('method')
    stds = sub.groupby(['dataset', 'rate', 'method']).macro.std().unstack('method')
    
    lines = [
        "# Relatório Consolidado: Curva de Orçamento Extremo e Novos Paradigmas",
        "",
        "Este relatório compara diretamente todos os métodos avaliados sob o mesmo protocolo experimental rigoroso (10 folds, Proxy validado com embeddings `jina-v5`).",
        "",
        "## 1. Tabela Comparativa de Macro-F1 Médio",
        "",
        md(means),
        "",
        "## 2. Testes de Significância Estatística vs Random (p-valor, t-pareado)",
        ""
    ]
    
    # Statistical tests against Random
    rand_df = sub[sub.method == 'random'].copy()
    test_methods = [m for m in sub.method.unique() if m != 'random']
    
    stat_rows = []
    for m in test_methods:
        m_df = sub[sub.method == m].copy()
        merged = m_df.merge(rand_df, on=['dataset', 'fold', 'rate'], suffixes=(f'_{m}', '_rand'))
        for (ds, rate), g in merged.groupby(['dataset', 'rate']):
            diff = g[f'macro_{m}'] - g['macro_rand']
            if len(diff) >= 5 and diff.std() > 1e-8:
                pval = stats.ttest_rel(g[f'macro_{m}'], g['macro_rand']).pvalue
            else:
                pval = np.nan
            stat_rows.append({
                'dataset': ds,
                'rate': rate,
                'method': m,
                'mean_diff': diff.mean(),
                'p_value': pval,
                'sig_0.05': 'SIM (*)' if pval < 0.05 and diff.mean() > 0 else ('PIOR (*)' if pval < 0.05 and diff.mean() < 0 else 'não')
            })
            
    stat_df = pd.DataFrame(stat_rows)
    if not stat_df.empty:
        # Pivot table for p-values and significant wins
        lines.append(md(stat_df.sort_values(['dataset', 'rate', 'method']), index=False))
        
    lines.append("")
    open(OUT_FILE, 'w').write('\n'.join(lines))
    print(f"Report written to {OUT_FILE}")
    print('\n'.join(lines[:45]))


if __name__ == '__main__':
    main()
