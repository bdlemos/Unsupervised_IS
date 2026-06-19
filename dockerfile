# Base com Python 3.8 (compatível com requirements antigos: numpy 1.17, sklearn 0.21, etc.)
FROM python:3.12.3


ENV DEBIAN_FRONTEND=noninteractive

# Ferramentas de sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git bash python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ─── Dependências Unificadas ─────────────────────────────────────────────────
# Copia o arquivo unificado de requirements
COPY requirements.txt ./requirements.txt

# Cria um único ambiente virtual e instala as dependências
RUN python -m venv venv \
    && venv/bin/pip install --upgrade pip \
    && venv/bin/pip install --no-cache-dir -r ./requirements.txt

# ─── Projeto 1: unsupervised-is ──────────────────────────────────────────────
COPY unsupervised-is/ ./unsupervised-is/

# ─── Projeto 2: atcBench ─────────────────────────────────────────────────────
COPY atcBench/ ./atcBench/

# ─── Pipeline principal ───────────────────────────────────────────────────────
# Ajusta ROOT_DIR do script para /app (de /home/bernardo/projects)
COPY run_pipeline.sh ./run_pipeline.sh
RUN sed -i 's|ROOT_DIR=.*|ROOT_DIR="/app"|' run_pipeline.sh \
    && chmod +x run_pipeline.sh

CMD ["bash", "run_pipeline.sh"]
