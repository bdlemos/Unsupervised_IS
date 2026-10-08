# Unsupervised Instance Selection Pipeline

Pipeline completo de seleção de instâncias não-supervisionada (`unsupervised-is`) seguido de benchmark de classificação de texto (`atcBench`).

---

## Estrutura

```
.
├── dockerfile                          # Imagem Docker do pipeline
├── .dockerignore                       # Exclui pastas grandes do build context
├── requirements.txt                    # Dependências unificadas de ambos os subprojetos
├── run_pipeline.sh                     # Orquestrador principal (5 steps)
├── run_sensitivity.sh                  # Exp 3: Sensitivity analysis do ESAE-IS
├── run_fixed_rates.sh                  # Exp 4: SAE-IS at fixed reduction rates
├── unsupervised-is/                    # Projeto de seleção de instâncias (ver README próprio)
│   ├── scripts/run_generateSplit.py    #   Entry-point da seleção (registry de métodos)
│   └── src/main/python/iSel/          #   Implementação dos métodos de IS
├── atcBench/                           # Projeto de benchmark de classificação (ver README próprio)
│   ├── main.py                        #   Entry-point (Hydra config)
│   └── run.sh                         #   Executa classificação com xargs pool
├── info_scripts/                       # Scripts utilitários
│   └── download_datasets.py           #   Download dos datasets do Zenodo
├── datasets/                           # Dados (montados via -v no Docker)
└── results/                            # Saídas (montados via -v no Docker)
```

> Para detalhes de cada subprojeto, consulte:
> - [`unsupervised-is/README.md`](unsupervised-is/README.md) — Métodos de seleção, interface, registry
> - [`atcBench/README.md`](atcBench/README.md) — Classificador, Hydra config, outputs

**Diretórios Externos (Data / Results):**
Os dados e resultados ficam separados do código-fonte. O padrão é:
- `./datasets` (ou `$DATASETS_DIR`): Dados brutos e embeddings (`tfidf`, `jina-v5`).
- `./results` (ou `$RESULTS_DIR`): Todos os outputs estruturados por `inputrep` ou por experimento.

---

## Execução via Docker (Recomendado)

### 1. Build da imagem

```bash
docker build -t unsupervised-is .
```

### 2. Executar o pipeline (Run Completo)

Basta mapear as pastas `datasets` e `results` e o código se resolve via variáveis de ambiente.

```bash
docker run -d --rm \
  --gpus '"device=1"' \
  --cpus="16" \
  --memory="32g" \
  -v $(pwd):/app/host \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/results:/app/results \
  --name pipeline-run \
  unsupervised-is \
  bash -c 'bash run_pipeline.sh \
    --methods "entropy-sublinear-ae-is" \
    --datasets "sst2,ohsumed,reuters90" \
    --inputrep "jina-v5" > /app/host/pipeline.log 2>&1'

# Acompanhe em tempo real:
tail -f pipeline.log
# Ou via docker logs:
docker logs -f pipeline-run
```

### 3. Executar com diretório de resultados isolado (Experimentos)

Para não interferir nos resultados base, use `RESULTS_DIR` para direcionar outputs para um diretório dedicado:

```bash
docker run -d --rm \
  --gpus '"device=1"' --cpus="16" --memory="32g" \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/results:/app/results \
  --name exp2-ablation \
  unsupervised-is \
  bash -c 'RESULTS_DIR=/app/results/exp2-ablation bash run_pipeline.sh \
    --methods "ablation-random-esae-rate,ablation-cluster-uniform" \
    --datasets "sst2,ohsumed,reuters90,books,trec,20ng" \
    --inputrep "jina-v5" 2>&1 | tee /app/results/exp2.log'
```

### 4. Executar scripts de experimentos dedicados

```bash
# Exp 3: Sensitivity Analysis
docker run -d --rm \
  --gpus '"device=1"' --cpus="16" --memory="32g" \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/results:/app/results \
  --name exp3-sensitivity \
  unsupervised-is \
  bash -c 'source /app/venv/bin/activate && \
    bash run_sensitivity.sh > /app/results/exp3.log 2>&1'

# Exp 4: Fixed Reduction Rates
docker run -d --rm \
  --gpus '"device=1"' --cpus="16" --memory="32g" \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/results:/app/results \
  --name exp4-fixed-rates \
  unsupervised-is \
  bash -c 'source /app/venv/bin/activate && \
    bash run_fixed_rates.sh > /app/results/exp4.log 2>&1'
```

---

## Parâmetros do `run_pipeline.sh`

| Parâmetro | Descrição | Default | Exemplo |
|---|---|---|---|
| `--methods` | Métodos de seleção (vírgula) | V1 defaults | `"entropy-sublinear-ae-is,cnn-is"` |
| `--datasets` | Datasets (vírgula) | Auto-descoberto por tamanho | `"sst2,ohsumed,20ng"` |
| `--inputrep` | Representação dos embeddings | `tfidf` | `"jina-v5"` |
| `--steps` | Steps específicos a rodar | Todos (1–5) | `"1,2,3"` ou `"4,5"` |
| `--model` | Modelo de classificação (Step 3–5) | `modernbert` | `"modernbert"` |

**Variável de ambiente:**
| Variável | Descrição | Default |
|---|---|---|
| `RESULTS_DIR` | Diretório de saída dos resultados | `results/<inputrep>` |
| `DATASETS_DIR` | Diretório dos datasets | `./datasets` |

---

## Pipeline: 5 Steps

| Step | Nome | Descrição |
|---|---|---|
| 1 | `selection` | Roda seleção de instâncias (`run_unsupervised_selection.sh`) |
| 2 | `summary` | Gera sumário de tempo e redução (`read_selection_ci.py`) |
| 3 | `benchmark` | Treina e avalia classificador ModernBERT (`atcBench/run.sh`) |
| 4 | `metrics` | Consolida métricas de classificação (Macro-F1) em CSV |
| 5 | `times` | Consolida tempos (IS + treino) em CSV |

Os steps podem ser executados seletivamente com `--steps "4,5"` (útil para regenerar CSVs sem retreinar).

---

## Métodos Disponíveis

### Método Principal (ESAE-IS)
| Método | Classe | Descrição |
|---|---|---|
| `entropy-sublinear-ae-is` | `EntropySublinearAEIS` | **ESAE-IS completo** — entropia adaptativa + tessellação sqrt(N) + AE scoring + remoção sublinear |
| `sublinear-ae-is` | `SublinearAEIS` | SAE-IS — taxa fixa (25%) + tessellação sqrt(N) + AE scoring + remoção sublinear |

### Baselines Não-Supervisionados (V1)
| Método | Descrição |
|---|---|
| `no-is` | Sem seleção (full dataset) |
| `biois` | Baseline IS |
| `random-is` | Remoção aleatória (50%) |
| `perplexity-is` | Baseado em perplexidade LDA |
| `autoencoder-is` | AE com thresholds fixos |
| `gmm-is` | Gaussian Mixture Model |
| `adaptive-perplexity` | Adaptive IS (imbalance-aware) |
| `adaptive-v2-perplexity` | Adaptive V2 (quartile-density) |
| `adaptive-cluster-is` | Adaptive com clustering |

### Baselines Supervisionados
| Método | Descrição |
|---|---|
| `e2sc-is` | E2SC (KNN approximated + heuristic beta) |
| `cnn-is` | Condensed Nearest Neighbor |
| `lssm-is` | Local Set-based Smoother |
| `lsbo-is` | Local Set Border Selector |

### Ablation Variants (Exp 2)
| Método | A (AE) | B (Cluster) | C (Sublinear) | Descrição |
|---|---|---|---|---|
| `ablation-random-esae-rate` | ✗ | ✗ | ✗ | Random sampling @ taxa adaptativa ESAE |
| `ablation-cluster-uniform` | ✗ | ✓ | ✗ | Clustering + remoção uniforme por cluster |
| `ablation-cluster-sublinear-random` | ✗ | ✓ | ✓ | Clustering + cap sublinear + remoção random |
| `ablation-ae-esae-rate` | ✓ | ✗ | — | AE global ranking @ taxa adaptativa |
| `ablation-cluster-uniform-ae` | ✓ | ✓ | ✗ | Clustering + linear + AE-guided |

### Parametric Methods (Exp 3 & 4)
| Método | Descrição |
|---|---|
| `esae-K50`, `esae-K100`, `esae-K200`, `esae-KsqrtN` | ESAE com variação de K |
| `esae-alpha5`, ..., `esae-alpha25` | ESAE com variação de α |
| `esae-gamma03`, `esae-gamma05`, `esae-gamma07` | ESAE com variação de γ |
| `esae-rmax03`, ..., `esae-rmax06` | ESAE com variação de r_max |
| `sae-rate-10`, ..., `sae-rate-40` | SAE-IS @ taxas fixas (10%–40%) |

---

## Estrutura de Resultados

### Resultados Base (V1)

| Path | Conteúdo |
|---|---|
| `results/jina-v5/instance_selection/selection/<dataset>/` | Splits gerados (PKL) por método |
| `results/jina-v5/instance_selection/selection_summary.csv` | Resumo de tempo e taxa de redução |
| `results/jina-v5/classificacao/output/` | Métricas JSON por fold (atcBench) |
| `results/jina-v5/classificacao/logs_modernbert/<dataset>/` | Logs de treino por dataset/método |
| `results/jina-v5/classificacao/results/results_modernbert.csv` | CSV final com Macro-F1 |
| `results/jina-v5/classificacao/results/times_modernbert.csv` | CSV final com tempos |

### Resultados dos Experimentos V2

```
results/
├── jina-v5/                          # V1 — Resultados base (intocados)
├── exp1-reconstruction-error/        # Exp 1: Análise do Reconstruction Error
├── exp2-ablation/                    # Exp 2: 2^k Factorial Ablation Study
├── exp3-sensitivity/                 # Exp 3: Sensitivity Analysis (one-at-a-time)
│   └── K_50/, K_100/, alpha_5/, ... # Cada configuração em subdiretório
├── exp4-fixed-rates/                 # Exp 4: SAE-IS @ fixed reduction rates
│   └── rate_10/, rate_15/, ...      # Cada taxa em subdiretório
├── exp5-entropy-vs-empirical/        # Exp 5: Entropia estimada vs empírica
├── exp6-random-matched-rate/         # Exp 6: Random @ matched rate
├── exp7-entropy-instrumentation/     # Exp 7: Instrumentação detalhada da entropia
├── exp8-entropy-estimator/           # Exp 8: Comparação de estimadores de entropia
└── exp9-tessellation/                # Exp 9: Comparação de métodos de tessellação
```

---

## Execução Local

### Pré-requisitos

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Executar o pipeline

```bash
# Run completo em background
nohup bash run_pipeline.sh --methods "entropy-sublinear-ae-is" --inputrep "jina-v5" > pipeline.log 2>&1 &
tail -f pipeline.log
```

---

## Datasets

Os datasets utilizados nos experimentos estão hospedados no [Zenodo](https://zenodo.org/) e podem ser baixados automaticamente com o script [`info_scripts/download_datasets.py`](info_scripts/download_datasets.py):

```bash
# Baixar todos os datasets
python info_scripts/download_datasets.py

# Baixar apenas um dataset específico
python info_scripts/download_datasets.py webkb

# Baixar um dataset e gerar seus embeddings Jina v5 (10 folds)
bash info_scripts/prepare_dataset_jina.sh webkb
```

O script baixa os ZIPs do Zenodo, extrai e organiza na estrutura esperada:
```
datasets/<name>/
├── texts.txt           # Textos brutos
├── score.txt           # Labels
├── splits/
│   └── split_10.pkl    # Splits pré-definidos (10-fold)
└── tfidf/              # Representação TF-IDF (train/test por fold)
```

**21 datasets disponíveis:** `20ng`, `acm`, `agnews`, `books`, `dblp`, `medline`, `movie_review`, `mpqa`, `ohsumed`, `pang_movie`, `reuters90`, `sst1`, `sst2`, `subj`, `trec`, `twitter`, `vader_movie`, `webkb`, `wos5736`, `wos11967`, `yelp_2013`, `yelp_reviews`.

O wrapper também gera os embeddings `jina-v5` (256 dimensões) em `datasets/<name>/jina-v5/`. Para usar outro diretório, defina `DATASETS_DIR` antes de executar.

---

## Notas Importantes

- **n_clusters = sqrt(N):** Todos os métodos que usam tessellação (ESAE-IS, SAE-IS, ablation variants) calculam o número de micro-clusters dinamicamente como `sqrt(N)`, onde N é o número de instâncias de treino do fold.
- **Auto-descoberta de datasets:** Quando `$DATASETS_DIR` existe, o pipeline descobre automaticamente todos os datasets e os ordena do menor para o maior (por tamanho de `texts.txt`), para que os mais pesados fiquem por último.
- **Tolerância a falhas:** Se um método falha em **todos** os datasets no Step 1, ele é automaticamente removido dos steps seguintes. Falhas parciais são reportadas mas não bloqueiam o pipeline.
- **Rebuild obrigatório:** Após alterar qualquer código em `unsupervised-is/` ou `atcBench/`, é necessário fazer `docker build -t unsupervised-is .` antes de executar novamente.
