# atcBench — Automated Text Classification Benchmark

Benchmark de classificação de texto que avalia o impacto da seleção de instâncias na qualidade do modelo. Treina e avalia um classificador por fold, salvando métricas detalhadas (Macro-F1, accuracy, tempos) para cada combinação dataset × método de IS.

## Estrutura

```
atcBench/
├── main.py                         # Entry-point (Hydra config)
├── run.sh                          # Orquestrador: loop dataset × method via xargs
├── conf/
│   ├── config_default.yaml         # Config Hydra principal
│   ├── data/template.yaml          # Template parametrizável (dataset + IS method)
│   ├── model/modernbert.yaml       # Config do ModernBERT
│   └── paths/default.yaml          # Paths de I/O
├── scripts/
│   ├── generate_results_csv.py     # Consolida métricas (Macro-F1) em CSV
│   └── generate_times_csv.py       # Consolida tempos (IS + treino) em CSV
└── src/
    ├── model/
    │   ├── general.py              # Factory de classificadores
    │   ├── slm.py                  # ModernBERT (classificador principal)
    │   ├── slmdatahandle.py        # Tokenização e DataLoader
    │   ├── traditional.py          # SVM/LR (classificadores tradicionais)
    │   └── llm.py                  # LLM-based classification
    └── utils/
        └── utils.py                # Data loading, config, seed, save
```

## Como Funciona

### 1. Configuração via Hydra

O `main.py` usa [Hydra](https://hydra.cc/) para compor a configuração. Na prática, o `run.sh` invoca:

```bash
python main.py "data=sst2_entropy-sublinear-ae-is"
```

O `main.py` reescreve esse argumento em:
```
data=template data.dataset_name=sst2 data.is_method=entropy-sublinear-ae-is
```

Isso resolve automaticamente os paths dos embeddings, labels e splits gerados pelo `unsupervised-is`.

### 2. Loop de Treinamento

Para cada fold (padrão: 10):
1. Carrega o split gerado pelo IS method (ou o split completo para `no-is`)
2. Instancia o classificador (ModernBERT por padrão)
3. Treina com early stopping (patience=3)
4. Salva probabilidades preditas + métricas por fold

### 3. Orquestração via `run.sh`

O `run.sh` gera todas as combinações dataset × method e as executa com `xargs -P` para paralelismo controlado:

```bash
bash run.sh --num-processes 1 --methods "entropy-sublinear-ae-is" --datasets "sst2,ohsumed"
```

Cada job gera seu próprio log em `$RESULTS_DIR/classificacao/logs_modernbert/<dataset>/<method>.log`.

## Modelo Padrão: ModernBERT

| Parâmetro | Valor |
|---|---|
| Modelo | `answerdotai/ModernBERT-base` |
| Max Length | 256 tokens |
| Batch Size | 32 |
| Learning Rate | 5e-5 |
| Epochs | 10 (max) |
| Patience | 3 |
| Warmup | 6% dos steps |

## Outputs

Para cada combinação dataset × method × fold:
- `$RESULTS_DIR/classificacao/output/<model_tag>/<dataset>_<method>/measures.fold_<N>.json` — Métricas detalhadas (F1, accuracy, precision, recall, tempos)
- `$RESULTS_DIR/classificacao/output/<model_tag>/<dataset>_<method>/y_pred_proba.fold_<N>.json` — Probabilidades preditas

### CSVs Consolidados (gerados pelo pipeline Steps 4-5)
- `results_modernbert.csv` — Macro-F1 por dataset × method (+ reduction rate)
- `times_modernbert.csv` — Tempos totais (IS time + training time) por dataset × method

## Variável de Ambiente

| Variável | Descrição | Default |
|---|---|---|
| `RESULTS_DIR` | Diretório base para outputs | Herdado do `run_pipeline.sh` |
| `DATASETS_DIR` | Diretório dos datasets | `/app/datasets` (Docker) ou `../datasets` |
| `TB_CONFIG_NAME` | Config Hydra a usar | `config_default.yaml` |
