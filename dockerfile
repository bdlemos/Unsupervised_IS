# Base com Python 3.8 (compatível com requirements antigos: numpy 1.17, sklearn 0.21, etc.)
FROM python:3.12.3


ENV DEBIAN_FRONTEND=noninteractive

# Ferramentas de sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git bash python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# ─── Projeto 1: unsupervised-is ──────────────────────────────────────────────
# Estrutura: bash/ notebooks/ scripts/ settings/ src/
WORKDIR /app/unsupervised-is

COPY unsupervised-is/settings/requirements_docker.txt ./settings/requirements_docker.txt

RUN python -m venv venv \
    && venv/bin/pip install --upgrade pip \
    && venv/bin/pip install --no-cache-dir -r ./settings/requirements_docker.txt

# Copia o restante do código do projeto (inclui notebooks/ e scripts/)
COPY unsupervised-is/ ./

# ─── Projeto 2: atcBench ─────────────────────────────────────────────────────
WORKDIR /app/atcBench

COPY atcBench/settings/requirements_docker.txt ./settings/requirements_docker.txt

RUN python -m venv venv \
    && venv/bin/pip install --upgrade pip \
    && venv/bin/pip install --no-cache-dir -r ./settings/requirements_docker.txt

# Copia o restante do código do projeto
COPY atcBench/ ./

# ─── Pipeline principal ───────────────────────────────────────────────────────
WORKDIR /app

# Ajusta ROOT_DIR do script para /app (de /home/bernardo/projects)
COPY run_pipeline.sh ./run_pipeline.sh
RUN sed -i 's|ROOT_DIR=.*|ROOT_DIR="/app"|' run_pipeline.sh \
    && chmod +x run_pipeline.sh

CMD ["bash", "run_pipeline.sh"]
