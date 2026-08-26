# Unsupervised Instance Selection Pipeline

Pipeline completo de seleção de instâncias não-supervisionada (`unsupervised-is`) seguido de benchmark de classificação de texto (`atcBench`).

---

## Estrutura

```
.
├── dockerfile                          # Imagem Docker do pipeline
├── .dockerignore                       # Exclui pastas grandes do build context para ficar muito rápido
├── requirements.txt                    # Dependências unificadas de ambos os subprojetos
├── run_pipeline.sh                     # Orquestrador principal dinâmico
├── unsupervised-is/                    # Projeto de seleção de instâncias
│   └── (scripts e lógicas de seleção)
├── atcBench/                           # Projeto de benchmark de classificação
│   └── (scripts de classificação e yaml configs)
```

**Diretórios Externos (Data / Results):**
Os dados e resultados ficam separados do código-fonte ou resolvidos relativamente ao diretório do projeto. O padrão é:
- `./datasets` (ou `$DATASETS_DIR`): Onde residem os dados brutos e embeddings gerados (incluindo pastas como `tfidf` e `jina-v5`).
- `./results` (ou `$RESULTS_DIR`): Onde o pipeline salvará todos os CSVs e outputs estruturados por `inputrep`.

---

## Execução via Docker (Recomendado)

### 1. Build da imagem

```bash
docker build -t unsupervised-is .
```

### 2. Executar o pipeline (Run Completo)

Basta mapear as pastas `datasets` e `results` e o código se resolve por completo via variáveis de ambiente ou diretórios padrão.

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
  bash -c 'bash run_pipeline.sh --methods "adaptive-cluster-is" --datasets "agnews,yelp_2013,medline" --inputrep "jina-v5" 2>&1 | tee /app/host/pipeline_geral.log'

# Acompanhe os logs em tempo real na máquina local:
tail -f pipeline_geral.log
```

### Parâmetros disponíveis do `run_pipeline.sh`

| Parâmetro | Descrição | Exemplo |
|---|---|---|
| `--methods` | Métodos de seleção separados por vírgula | `"adaptive-perplexity,adaptive-cluster-is"` |
| `--datasets` | Datasets separados por vírgula | `"mpqa,sst2,twitter"` |
| `--inputrep` | Tipo da representação dos dados (`tfidf`, `jina-v5`, etc) | `"jina-v5"` |
| `--steps` | Passos específicos a serem rodados | `"1,2"` ou `"selection,benchmark"` |

**Métodos disponíveis:** `no-is`, `biois`, `random-is`, `perplexity-is`, `autoencoder-is`, `pca-autoencoder-is`, `gmm-is`, `adaptive-perplexity`, `adaptive-v2-perplexity`, `adaptive-cluster-is`.

**Datasets (auto-descobertos):** Todos os que existirem fisicamente no diretório `$DATASETS_DIR` (padrão: `./datasets/`).

---

## Execução local

### Pré-requisitos

Com a unificação, existe apenas **um** ambiente virtual.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Executar o pipeline

Nenhuma variável de ambiente manual obscura é necessária, contanto que as pastas `./datasets` e `./results` (ou via `DATASETS_DIR` e `RESULTS_DIR`) já existam na máquina física (o script descobre e aponta para elas por padrão).

```bash
# Run completo em background com salvamento de log
nohup bash run_pipeline.sh --methods "adaptive-cluster-is" --inputrep "jina-v5" > pipeline_geral.log 2>&1 &
tail -f pipeline_geral.log
```

---

## Estrutura de Outputs

Com o suporte nativo e dinâmico a diferentes representações de embeddings via CLI (`--inputrep jina-v5`), todos os resultados passam a se organizar automaticamente pelo seu tipo no disco. Se você rodar o comando com `jina-v5`, a pasta `results` ficará com esta cara:

| Path | Conteúdo |
|---|---|
| `results/jina-v5/instance_selection/selection/<dataset>/` | Splits gerados por método na seleção não-supervisionada |
| `results/jina-v5/instance_selection/selection_summary.csv` | Resumo de tempo e taxa de redução aplicados a cada método |
| `results/jina-v5/classificacao/output/` | Métricas JSON detalhadas do atcBench (classificação em si) por fold |
| `results/jina-v5/classificacao/results/results_from_outputs.csv` | CSV consolidado com as métricas de performance finais (macro F1) para o paper |
| `results/jina-v5/classificacao/results/times_from_outputs.csv` | CSV consolidado com os tempos absolutos somados de todas as partes do processo (IS + Treino) |
| `atcBench/logs/run_<dataset>_<method>.log` | Log verboso individual para os jobs do atcBench |
