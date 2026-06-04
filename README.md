# Unsupervised Instance Selection Pipeline

Pipeline completo de seleção de instâncias não-supervisionada (`unsupervised-is`) seguido de benchmark de classificação de texto (`atcBench`).

---

## Estrutura

```
.
├── dockerfile                          # Imagem Docker do pipeline
├── .dockerignore                       # Exclui venvs e caches do build context
├── run_pipeline.sh                     # Orquestrador principal
├── unsupervised-is/                    # Projeto de seleção de instâncias
│   ├── bash/run_unsupervised_selection.sh
│   ├── scripts/run_generateSplit.py
│   ├── settings/requirements_docker.txt
│   └── resources/
│       ├── datasets/                   # Datasets de entrada
│       ├── outsel/                     # Splits gerados pela seleção
│       └── logs/                       # Logs de execução
├── atcBench/                           # Projeto de benchmark
│   ├── run.sh
│   ├── main.py
│   ├── conf/data/template.yaml
│   ├── settings/requirements_docker.txt
│   ├── resources/output/               # Resultados do benchmark
│   ├── resources/results/              # Tabelas CSV consolidadas de resultados
│   └── logs/                           # Logs por combinação dataset/método
```

---

## Execução via Docker (recomendado)

### 1. Build da imagem

```bash
docker build -t unsupervised-is /data/bernardolemos/
```

### 2. Criar pastas de output no host (só na primeira vez)

```bash
mkdir -p /data/bernardolemos/unsupervised-is/resources/logs
mkdir -p /data/bernardolemos/atcBench/resources/output
mkdir -p /data/bernardolemos/atcBench/resources/results
mkdir -p /data/bernardolemos/atcBench/logs
```

### 3. Run completo (todos os datasets e métodos)

```bash
docker run -d --rm \
  --gpus '"device=1"' \
  --cpus="16" \
  --memory="32g" \
  -v /data/bernardolemos/unsupervised-is/resources:/app/unsupervised-is/resources \
  -v /data/bernardolemos/atcBench/resources/output:/app/atcBench/resources/output \
  -v /data/bernardolemos/atcBench/resources/results:/app/atcBench/resources/results \
  -v /data/bernardolemos/atcBench/logs:/app/atcBench/logs \
  --name pipeline-run \
  unsupervised-is \
  bash run_pipeline.sh --datasets "mpqa,trec,sst2,twitter,vader_movie,movie_review,sst1,pang_movie,subj,yelp_reviews,wos5736,reuters90,webkb,wos11967,ohsumed,20ng,agnews,dblp,books,yelp_2013,medline"

# Acompanhar os logs em tempo real
docker logs -f pipeline-run
```

### 3.1. Persistindo os logs de execução em arquivo (equivalente ao nohup)

#### Abordagem recomendada
Executa em background (`-d`) gerenciado pelo Docker, mas usa o utilitário `tee` dentro do container para salvar a saída na pasta raiz do container:

```bash
docker run -d --rm \
  --gpus '"device=1"' \
  --cpus="16" \
  --memory="32g" \
  -v /data/bernardolemos:/app/host \
  -v /data/bernardolemos/unsupervised-is/resources:/app/unsupervised-is/resources \
  -v /data/bernardolemos/atcBench/resources/output:/app/atcBench/resources/output \
  -v /data/bernardolemos/atcBench/resources/results:/app/atcBench/resources/results \
  -v /data/bernardolemos/atcBench/logs:/app/atcBench/logs \
  --name pipeline-run \
  unsupervised-is \
  bash -c "bash run_pipeline.sh 2>&1 | tee /app/host/pipeline_geral.log"

# Você pode acompanhar tanto via docker quanto lendo o arquivo na máquina física:
docker logs -f pipeline-run
# OU:
tail -f /data/bernardolemos/pipeline_geral.log
```


### 4. Run filtrado (teste rápido)

```bash
docker run -it --rm \
  --gpus '"device=1"' \
  --cpus="16" \
  --memory="32g" \
  -v /data/bernardolemos/unsupervised-is/resources:/app/unsupervised-is/resources \
  -v /data/bernardolemos/atcBench/resources/output:/app/atcBench/resources/output \
  -v /data/bernardolemos/atcBench/resources/results:/app/atcBench/resources/results \
  -v /data/bernardolemos/atcBench/logs:/app/atcBench/logs \
  unsupervised-is \
  bash run_pipeline.sh \
    --methods "adaptive-v2-perplexity" \
    --datasets "webkb,twitter"
```

### Parâmetros disponíveis

| Parâmetro | Descrição | Exemplo |
|---|---|---|
| `--methods` | Métodos de seleção separados por vírgula | `"adaptive-perplexity,no-is"` |
| `--datasets` | Datasets separados por vírgula | `"mpqa,sst2,twitter"` |

**Métodos disponíveis:** `no-is`, `biois`, `random-is`, `perplexity-is`, `autoencoder-is`, `pca-autoencoder-is`, `gmm-is`, `adaptive-perplexity`, `adaptive-v2-perplexity`

**Datasets disponíveis (auto-descobertos de `resources/datasets/`):**
`20ng`, `agnews`, `books`, `dblp`, `medline`, `movie_review`, `mpqa`, `ohsumed`, `pang_movie`, `reuters90`, `sst1`, `sst2`, `subj`, `trec`, `vader_movie`, `webkb`, `wos11967`, `wos5736`, `yelp_2013`, `yelp_reviews`

### Parar o container

```bash
docker stop pipeline-run
```

---

## Execução local

### Pré-requisitos

- Python 3.12+
- Virtualenvs criados em cada subprojecto:

```bash
# unsupervised-is
cd unsupervised-is
python -m venv venv
source venv/bin/activate
pip install -r settings/requirements_docker.txt
deactivate

# atcBench
cd ../atcBench
python -m venv venv
source venv/bin/activate
pip install -r settings/requirements_docker.txt
deactivate
```

### Variável de ambiente necessária

O `atcBench` usa a variável `UNSUPERVISED_IS_RESOURCES` para localizar os datasets e splits. **Sem ela o pipeline assume o path do Docker (`/app/...`).**

```bash
export UNSUPERVISED_IS_RESOURCES=/data/bernardolemos/unsupervised-is/resources
```

### Executar o pipeline

```bash
cd /data/bernardolemos

# Run completo
bash run_pipeline.sh

# Run filtrado
bash run_pipeline.sh \
  --methods "adaptive-v2-perplexity" \
  --datasets "webkb,twitter"
```

### Background (equivalente ao -d do Docker)

```bash
nohup bash run_pipeline.sh > pipeline_geral.log 2>&1 &
tail -f pipeline_geral.log
```

---

## Outputs

| Path | Conteúdo |
|---|---|
| `unsupervised-is/resources/outsel/selection/<dataset>/` | Splits gerados por método |
| `unsupervised-is/resources/outsel/selection_summary.csv` | Resumo de tempo e redução |
| `atcBench/resources/output/` | Métricas de classificação por fold |
| `atcBench/resources/results/results_from_outputs.csv` | CSV consolidado com as métricas de classificação |
| `atcBench/resources/results/times_from_outputs.csv` | CSV consolidado com os tempos totais (Seleção + Treino) |
| `atcBench/logs/run_<dataset>_<method>.log` | Log detalhado por combinação |
