#!/bin/bash
# prepare_datasets.sh wrapper script
# Prepares and processes all datasets for training

set -e

# Always run from the repo root, regardless of where the script is called from
cd "$(dirname "$0")/.."

# Load .env so HF_TOKEN and friends are available to Python
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "Loaded .env"
else
    echo "WARNING: .env not found — copy .env.example to .env and fill in HF_TOKEN"
fi

echo "================================================"
echo "Dataset Preparation Pipeline"
echo "================================================"

# Configuration
DATA_DIR="${DATA_DIR:-./data/processed}"
RAW_DATA_DIR="${RAW_DATA_DIR:-./data/raw}"
SAMPLES="${SAMPLES:-40000}"

# Create directories
mkdir -p "$RAW_DATA_DIR"
mkdir -p "$DATA_DIR"

echo "Raw data directory: $RAW_DATA_DIR"
echo "Processed data directory: $DATA_DIR"

# Step 1: Generate synthetic data (fast, no network needed)
echo ""
echo "Step 1/2: Generating synthetic data..."
python3 scripts/generate_synthetic.py \
    --num-samples "${SAMPLES:-15000}" \
    --output-dir "$RAW_DATA_DIR/synthetic"

echo ""
echo "Step 2/2: Loading & processing all datasets..."

# Run Python script for data preparation
python3 << 'EOF'
import sys
import logging
from pathlib import Path
sys.path.insert(0, './src')

from data_loader import ToolUseDataLoader
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting dataset preparation...")

# Initialize data loader
loader = ToolUseDataLoader("configs/dataset_config.yaml")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare datasets
logger.info("Loading and preprocessing datasets...")
datasets = loader.prepare_datasets(tokenizer, max_length=2048)

logger.info("Saving datasets...")
datasets.save_to_disk("./data/processed/")

logger.info(f"Train set: {len(datasets['train'])} examples")
logger.info(f"Validation set: {len(datasets['validation'])} examples")
logger.info(f"Test set: {len(datasets['test'])} examples")

logger.info("Dataset preparation complete!")

EOF

echo "================================================"
echo "✓ Dataset preparation complete"
echo "================================================"
