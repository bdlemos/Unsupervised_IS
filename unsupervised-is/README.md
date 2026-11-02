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
├── src/
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
│   │   └── ...                     # Outros baselines (biois, random, ablações, cluster-random)
│   └── utils/                      # Helpers (arguments, data loading, etc.)
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

Após `fit(X, y)`, os índices selecionados estão em `selector.sample_indices_`, e
`selector.selection_record()` devolve os parâmetros do construtor (`params`) e os valores
que decidiram a seleção (`selection`: K efetivo, entropia, taxa alvo, λ, células podadas, etc.).

Os métodos da família ESAE (ESAE-IS, SAE-IS e as ablações) são montados a partir de
`iSel/esae_core.py` (tesselação, regra de entropia, cotas sublineares e remoção dentro das
células). `n_clusters=-1` significa K = √N. Toda a aleatoriedade usa `RandomState(random_state)`.

## Registry de Métodos

Os métodos são registrados em `scripts/run_generateSplit.py` na função `get_selector(method)`. Cada nome de método (string) mapeia para uma instância do selector correspondente.

### Execução

A seleção é tipicamente invocada pelo pipeline principal (`bash/run_pipeline.sh` Step 1), mas pode ser executada diretamente:

```bash
cd unsupervised-is
python scripts/run_generateSplit.py \
  -d sst2 -m entropy-sublinear-ae-is \
  --datain ../datasets --out ../results/jina-v5/instance_selection \
  --inputrep jina-v5
```

`--seed N` gera uma repetição independente: o `random_state` fixo do método é trocado por um
derivado de `(N, fold)`, e as saídas ganham o sufixo `-seedN` (ex.: `split_10_random-rate-40-seed3.pkl`).
Sem `--seed`, o comportamento é o de sempre (o `random_state` do registry).

### Outputs

Para cada combinação dataset × method × fold:
- `selection/<dataset>/split_10_<method>.pkl` — PKL com os índices selecionados (traduzidos para índices globais)
- `selection/<dataset>/split_10_<method>_idxinfold.pkl` — PKL com índices relativos ao fold
- `selection/<dataset>/saida_<method>.json` — Métricas de tempo e redução
- `selection/<dataset>/split_10_<method>_params.json` — Por fold: parâmetros do seletor, `random_state`, valores de decisão (`selection`), redução realizada, tempo e o commit do código (`GIT_COMMIT` sobrepõe, útil no Docker sem `.git`)

O script `read_selection_ci.py` consolida todos os JSONs em `selection_summary.csv`.
