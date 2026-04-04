# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Qwen3-8B Tool-Use Fine-Tuning Pipeline — Azure Container Image
#
# The container is a pure CPU orchestrator:
#   - Generates synthetic training data locally
#   - Dispatches training/inference to Tinker's remote GPUs via API
#   - Logs all metrics to Weights & Biases
#   - No local model weights are downloaded
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# System deps: git (for pip VCS installs), ca-certificates, curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy source
COPY src/       ./src/
COPY scripts/   ./scripts/
COPY configs/   ./configs/

# Make scripts executable
RUN chmod +x scripts/*.sh

# Ensure src/ is on PYTHONPATH so `from constants import ...` works without install
ENV PYTHONPATH=/app/src

# Runtime secrets — all injected at `az container create` time as env vars.
# Do NOT bake real values here.
ENV TINKER_API_KEY=""
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""
ENV WANDB_PROJECT="qwen3-8b-tool-use"
ENV HF_TOKEN=""
ENV HF_REPO_ID=""
ENV BASE_MODEL="Qwen/Qwen3-8B"
ENV EVAL_SAMPLES="1000"
ENV SYNTH_SAMPLES="15000"
ENV BENCHMARK_FILTER=""
ENV STAGE="all"

# Default: run the full pipeline.
# Override CMD at container create time to run a specific stage, e.g.:
#   --command-line "bash scripts/run_pipeline.sh baseline"
CMD ["bash", "scripts/run_pipeline.sh"]
