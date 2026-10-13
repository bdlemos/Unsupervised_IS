#!/usr/bin/env python3
"""
Step B — Budget curve with the cheap proxy.

Compares, for each reduction rate in RATES:
  random, cluster-random (gamma 0.5), typ-cover (gamma 0.5), typ-cover (gamma 1.0)

Pre-registered stop criterion (decided BEFORE running):
  typ-cover must beat random with paired p<0.05 at >=90% reduction
  in at least 3 of the datasets. Otherwise we drop this line.

Output (only): results/proxy-budget-curve/budget_curve/
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from proxy_lib import OUT_ROOT, load_fold, proxy_eval, select, typicality, md

OUT = os.path.join(OUT_ROOT, 'budget_curve')
METHODS = [('random', 0.5), ('cluster-random', 0.5), ('typ-cover', 0.5), ('typ-cover', 1.0)]


def mname(m, g):
    return m if m == 'random' else f'{m}-g{g:g}'


def run_fold(ds, fold, rates):
    X_tr, y_tr, X_te, y_te = load_fold(ds, fold)
    typ = typicality(X_tr)
    n = len(y_tr)
    rows = []
    for rate in rates:
        n_keep = max(int(round(n * (1 - rate / 100))), 2)
        for m, g in METHODS:
            idx = select(m, X_tr, n_keep, seed=1000 + fold, typ=typ, gamma=g)
            if len(np.unique(y_tr[idx])) < 2:
                continue
            r = proxy_eval(X_tr[idx], y_tr[idx], X_te, y_te)
            rows.append(dict(dataset=ds, fold=fold, rate=rate, method=mname(m, g), n_keep=len(idx),
                             n_classes_kept=len(np.unique(y_tr[idx])),
                             macro=r['macro'], micro=r['micro']))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='webkb,books,dblp,acm,20ng,sst1')
    ap.add_argument('--rates', default='50,80,90,95,98,99')
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--n-jobs', type=int, default=30)
    ap.add_argument('--report-only', action='store_true',
                    help='rebuild REPORT.md from per_fold.csv without recomputing')
    a = ap.parse_args()
    datasets = a.datasets.split(',')
    rates = [int(r) for r in a.rates.split(',')]
    os.makedirs(OUT, exist_ok=True)

    if a.report_only:
        df = pd.read_csv(os.path.join(OUT, 'per_fold.csv'))
    else:
        res = Parallel(n_jobs=a.n_jobs, verbose=5)(
            delayed(run_fold)(ds, f, rates) for ds in datasets for f in range(a.folds))
        df = pd.DataFrame([r for rows in res for r in rows])
        df.to_csv(os.path.join(OUT, 'per_fold.csv'), index=False)

    summ = df.groupby(['dataset', 'rate', 'method']).macro.agg(['mean', 'std']).reset_index()
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)

    tests = []
    for (ds, rate), g in df.groupby(['dataset', 'rate']):
        base = g[g.method == 'random'].set_index('fold').macro
        for m in g.method.unique():
            if m == 'random':
                continue
            x = g[g.method == m].set_index('fold').macro
            common = base.index.intersection(x.index)
            if len(common) < 3:
                continue
            t = stats.ttest_rel(x[common], base[common])
            tests.append(dict(dataset=ds, rate=rate, method=m, diff=x[common].mean() - base[common].mean(),
                              p=t.pvalue, n=len(common)))
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(OUT, 'tests_vs_random.csv'), index=False)

    lines = ['# Curva de orçamento (proxy: Regressão Logística sobre jina-v5)', '']
    pv = summ.pivot_table(index=['dataset', 'rate'], columns='method', values='mean')
    lines.append('## Macro-F1 médio\n')
    lines.append(md(pv))
    tp = tests.pivot_table(index=['dataset', 'rate'], columns='method', values='diff')
    pp = tests.pivot_table(index=['dataset', 'rate'], columns='method', values='p')
    cell = tp.copy().astype(object)
    for c in tp.columns:
        cell[c] = [f'{d:+.4f} (p={p:.3f})' + ('*' if p < 0.05 else '') for d, p in zip(tp[c], pp[c])]
    lines.append('\n## Diferença vs Random (teste t pareado)\n')
    lines.append(md(cell))

    # Pre-registered stop criterion
    lines.append('\n## Critério de parada pré-registrado\n')
    for m in [x for x in tests.method.unique() if x.startswith('typ-cover')]:
        hi = tests[(tests.method == m) & (tests.rate >= 90)]
        wins = hi[(hi['diff'] > 0) & (hi.p < 0.05)].groupby('dataset').size()
        n_ds = int((wins > 0).sum())
        verdict = 'PASSA' if n_ds >= 3 else 'NÃO PASSA'
        lines.append(f'- {m}: vence o Random (p<0.05) em >=90% de redução em {n_ds} datasets '
                     f'({", ".join(wins.index) or "nenhum"}) -> **{verdict}**')
    open(os.path.join(OUT, 'REPORT.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
