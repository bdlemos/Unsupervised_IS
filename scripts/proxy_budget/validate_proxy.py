#!/usr/bin/env python3
"""
Step A — Validate the proxy against ModernBERT.

Re-uses the selections ALREADY produced in results/exp-aggressive-rates/ (read-only),
trains the cheap proxy on each selected subset, and compares with the ModernBERT
Macro-F1 stored there.

Output (only): results/proxy-budget-curve/validation/
"""
import os, sys, json, glob, pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from proxy_lib import REPO_DIR, OUT_ROOT, load_fold, proxy_eval, md

AGG = os.path.join(REPO_DIR, 'results', 'exp-aggressive-rates')
OUT = os.path.join(OUT_ROOT, 'validation')
DATASETS = ['webkb', 'books', 'dblp', 'acm']
RATES = [40, 60]
METHODS = ['random-rate', 'sae-rate', 'sae-softmax-rate']


def run_fold(ds, fold, combos):
    X_tr, y_tr, X_te, y_te = load_fold(ds, fold)
    rows = []
    for rate, m, idx in combos:
        r = proxy_eval(X_tr[idx], y_tr[idx], X_te, y_te)
        rows.append(dict(dataset=ds, rate=rate, method=m, fold=fold,
                         proxy_macro=r['macro'], proxy_micro=r['micro']))
    r = proxy_eval(X_tr, y_tr, X_te, y_te)
    rows.append(dict(dataset=ds, rate=0, method='full', fold=fold,
                     proxy_macro=r['macro'], proxy_micro=r['micro']))
    return rows


def bert_macro(ds, rate, m, fold):
    f = os.path.join(AGG, f'rate_{rate}', 'classificacao', 'output', 'modernbert',
                     f'{ds}_{m}-{rate}', f'measures.fold_{fold}.json')
    return json.load(open(f))['macro'] if os.path.exists(f) else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    if '--report-only' in sys.argv:
        df = pd.read_csv(os.path.join(OUT, 'per_fold.csv'))
    else:
        jobs = []
        for ds in DATASETS:
            for fold in range(10):
                combos = []
                for rate in RATES:
                    for m in METHODS:
                        p = os.path.join(AGG, f'rate_{rate}', 'instance_selection', 'selection', ds,
                                         f'split_10_{m}-{rate}_idxinfold.pkl')
                        if os.path.exists(p):
                            combos.append((rate, m, np.asarray(pickle.load(open(p, 'rb')).loc[fold, 'train_idxs'])))
                jobs.append((ds, fold, combos))

        res = Parallel(n_jobs=20, verbose=5)(delayed(run_fold)(*j) for j in jobs)
        df = pd.DataFrame([r for rows in res for r in rows])
    # ModernBERT scores are (re)read every time, so folds finished later are picked up.
    df['bert_macro'] = [bert_macro(r.dataset, r.rate, r.method, r.fold) if r.method != 'full' else np.nan
                        for r in df.itertuples()]
    df.to_csv(os.path.join(OUT, 'per_fold.csv'), index=False)

    sel = df[df.method != 'full'].dropna(subset=['bert_macro'])
    agg = sel.groupby(['dataset', 'rate', 'method']).agg(
        proxy_macro=('proxy_macro', 'mean'), bert_macro=('bert_macro', 'mean'),
        n_folds=('fold', 'nunique')).reset_index()
    agg.to_csv(os.path.join(OUT, 'summary.csv'), index=False)

    lines = ['# Validação do proxy (Regressão Logística) vs ModernBERT', '']
    lines.append('## Médias por configuração (só folds com resultado do ModernBERT)\n')
    lines.append(md(agg, index=False))
    full = df[df.method == 'full'].groupby('dataset').proxy_macro.mean()
    lines.append('\n## Proxy com 100% dos dados (referência)\n')
    lines.append(md(full.to_frame()))

    rho_lvl = stats.spearmanr(agg.proxy_macro, agg.bert_macro)
    lines.append(f'\n## Correlação de nível (Spearman, {len(agg)} configs): rho={rho_lvl.correlation:.3f} p={rho_lvl.pvalue:.3g}')
    lines.append('(Fácil: é dominada pela diferença entre datasets.)\n')

    # The relevant check: do proxy and BERT agree on the *difference* method − random?
    d = []
    for (ds, rate), g in agg.groupby(['dataset', 'rate']):
        base = g[g.method == 'random-rate']
        if base.empty:
            continue
        base = base.iloc[0]
        for r in g[g.method != 'random-rate'].itertuples():
            d.append(dict(dataset=ds, rate=rate, method=r.method,
                          d_proxy=r.proxy_macro - base.proxy_macro,
                          d_bert=r.bert_macro - base.bert_macro))
    d = pd.DataFrame(d)
    d.to_csv(os.path.join(OUT, 'deltas_vs_random.csv'), index=False)
    rho_d = stats.spearmanr(d.d_proxy, d.d_bert)
    sign = (np.sign(d.d_proxy) == np.sign(d.d_bert)).mean()
    lines.append('## Diferenças método − Random (o teste que importa)\n')
    lines.append(md(d, floatfmt='+.4f', index=False))
    lines.append(f'\nSpearman(Δproxy, Δbert) = {rho_d.correlation:.3f} (p={rho_d.pvalue:.3g}); '
                 f'concordância de sinal = {sign:.0%} ({len(d)} pares)')
    open(os.path.join(OUT, 'REPORT.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
