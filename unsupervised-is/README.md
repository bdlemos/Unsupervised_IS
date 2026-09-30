# unsupervised-is — Instance Selection Methods

Biblioteca de métodos de seleção de instâncias não-supervisionada para classificação de texto. Contém os métodos propostos (ESAE-IS, SAE-IS) e os baselines usados na avaliação experimental.

## Estrutura

```
unsupervised-is/
├── scripts/
│   ├── run_generateSplit.py        # Entry-point: registry de métodos + geração de splits
│   └── read_selection_ci.py        # Gera sumário com intervalos de confiança
├── bash/
│   └── run_unsupervised_selection.sh  # Orquestrador de seleção (loop dataset × method)
├── src/main/python/
│   ├── iSel/                       # Implementação dos métodos de IS
│   │   ├── base.py                 # InstanceSelectionMixin (classe base)
│   │   ├── entropy_sublinear_ae_is.py  # ESAE-IS (método principal)
│   │   ├── sublinear_ae_is.py      # SAE-IS (taxa fixa, sem entropia)
│   │   ├── ablation_methods.py     # 5 variantes de ablação (Exp 2)
│   │   ├── autoencoder_is.py       # Autoencoder IS (baseline + componente do ESAE)
│   │   ├── e2sc.py                 # E2SC (baseline supervisionado)
│   │   ├── cnn.py                  # Condensed Nearest Neighbor
│   │   ├── lssm.py                 # Local Set-based Smoother
│   │   ├── lsbo.py                 # Local Set Border Selector
│   │   └── ...                     # Outros baselines (biois, gmm, perplexity, etc.)
│   └── utils/                      # Helpers (arguments, data loading, etc.)
├── analysis/                       # Scripts de análise (Exp 1, Exp 5)
│   ├── exp1_reconstruction_analysis.py
│   └── exp5_entropy_vs_empirical.py
└── LICENSE.md
```

## Método Principal: ESAE-IS

**Entropy-adaptive Sublinear Autoencoder Instance Selection** (`EntropySublinearAEIS`)

Pipeline em 4 etapas:
1. **Autoencoder Scoring** — Treina um AE para obter o reconstruction error de cada instância
2. **Tessellação Espacial** — MiniBatchKMeans com K = √N micro-clusters
3. **Taxa Adaptativa via Entropia** — Calcula a entropia normalizada da distribuição de tamanhos dos clusters para determinar automaticamente a taxa de redução
4. **Remoção Sublinear AE-guided** — Dentro de cada cluster, remove instâncias com cap sublinear (√s_c), priorizando as de menor reconstruction error (mais redundantes)

### Interface

Todos os métodos implementam `InstanceSelectionMixin`:

```python
class InstanceSelectionMixin:
    def fit(self, X, y):
        """Chama select_data e retorna self."""
    def select_data(self, X, y):
        """Executa a seleção. Popula self.sample_indices_."""
```

Após `fit(X, y)`, os índices selecionados estão em `selector.sample_indices_`.

## Registry de Métodos

Os métodos são registrados em `scripts/run_generateSplit.py` na função `get_selector(method)`. Cada nome de método (string) mapeia para uma instância do selector correspondente.

### Execução

A seleção é tipicamente invocada pelo pipeline principal (`run_pipeline.sh` Step 1), mas pode ser executada diretamente:

```bash
cd unsupervised-is
python scripts/run_generateSplit.py \
  -d sst2 -m entropy-sublinear-ae-is \
  --datain ../datasets --out ../results/jina-v5/instance_selection \
  --inputrep jina-v5
```

### Outputs

Para cada combinação dataset × method × fold:
- `selection/<dataset>/split_10_<method>.pkl` — PKL com os índices selecionados (traduzidos para índices globais)
- `selection/<dataset>/split_10_<method>_idxinfold.pkl` — PKL com índices relativos ao fold
- `selection/<dataset>/saida_<method>.json` — Métricas de tempo e redução

O script `read_selection_ci.py` consolida todos os JSONs em `selection_summary.csv`.
