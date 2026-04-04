#!/bin/bash
# prepare_datasets.sh wrapper script
# Prepares and processes all datasets for training

set -e

# Always run from the repo root, regardless of where the script is called from
cd "$(dirname "$0")/.."

# Load .env so WANDB keys and friends are available to Python
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "Loaded .env"
else
    echo "WARNING: .env not found — copy .env.example to .env and fill in your keys"
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
echo "Step 2/2: Processing and tokenizing synthetic data..."

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
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess (keep all original columns intact)
logger.info("Loading and preprocessing datasets...")
dataset = loader.load_all_datasets()
dataset = loader.preprocess(dataset)
if loader.config.get("balance_sources", True):
    dataset = loader._balance_sources(dataset)
train_data, val_data, test_data = loader.split_dataset(dataset)

# Save raw test split as JSONL for evaluation BEFORE text normalization.
# This preserves all original fields: instruction, tool_calls, tools, query,
# function, api_call, etc. — everything _normalize_row needs to extract
# expected tools and prompts for structured evaluation.
import json
test_raw_path = Path("./data/processed/test_raw.jsonl")
with open(test_raw_path, "w") as f:
    for row in test_data:
        f.write(json.dumps(row) + "\n")
logger.info(f"Saved raw test split → {test_raw_path} ({len(test_data)} examples, all fields preserved)")

# Now normalize to unified text column for tokenization (training use only)
logger.info("Normalizing to unified text column for training...")
train_data = loader._normalize_to_text(train_data)
val_data   = loader._normalize_to_text(val_data)
test_data  = loader._normalize_to_text(test_data)

# Tokenize and save Arrow splits for training
logger.info("Tokenizing datasets for training...")
from datasets import DatasetDict
train_tok = loader.tokenize_dataset(train_data, tokenizer, max_length=2048)
val_tok   = loader.tokenize_dataset(val_data,   tokenizer, max_length=2048)
test_tok  = loader.tokenize_dataset(test_data,  tokenizer, max_length=2048)
tokenized = DatasetDict({"train": train_tok, "validation": val_tok, "test": test_tok})

logger.info("Saving tokenized datasets...")
tokenized.save_to_disk("./data/processed/")

logger.info(f"Train set: {len(tokenized['train'])} examples")
logger.info(f"Validation set: {len(tokenized['validation'])} examples")
logger.info(f"Test set (tokenized): {len(tokenized['test'])} examples")
logger.info(f"Test set (raw JSONL): {len(test_data)} examples")

logger.info("Dataset preparation complete!")

EOF

echo "================================================"
echo "✓ Dataset preparation complete"
echo "================================================"
