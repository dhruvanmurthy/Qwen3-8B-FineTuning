# Step-by-Step Execution Plan

Complete guide to execute the Qwen3-8B fine-tuning project from start to finish.

Pipeline: **Baseline Eval → SFT Training → GRPO Training → Final Comparison**

## Phase 1: Local Setup (1 hour)

### Step 1.1: Create Virtual Environment
```bash
# PowerShell on Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Or conda
conda create -n qwen3-finetune python=3.11
conda activate qwen3-finetune
```

### Step 1.2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 1.3: Configure Environment Variables
```bash
# Copy template
cp .env.example .env

# Edit .env with your credentials:
#   HF_TOKEN=<your_token>
#   WANDB_API_KEY=<your_key>
```

### Step 1.4: Verify Installation
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoTokenizer; print('Transformers OK')"
python -c "from peft import LoraConfig; print('PEFT OK')"
python -c "from trl import GRPOConfig; print('TRL OK')"
```

## Phase 2: W&B Setup (10 min)

### Step 2.1: Create W&B Project
```bash
wandb login
# Project: qwen3-8b-tool-use at wandb.ai/dhruvanmurthy/qwen3-8b-tool-use
```

## Phase 3: Data Preparation (1-2 hours)

### Step 3.1: Download & Process Datasets
```bash
bash scripts/prepare_datasets.sh
```

### Step 3.2: Verify Dataset
```bash
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/processed')
print(f'Train samples: {len(ds[\"train\"])}')
print(f'Validation samples: {len(ds[\"validation\"])}')
print(f'Test samples: {len(ds[\"test\"])}')
print(f'Columns: {ds[\"train\"].column_names}')
"
```

Expected: ~3,043 train / ~380 val / ~381 test, columns: `input_ids`, `attention_mask`, `labels`.

## Phase 4: Baseline Evaluation (15-30 min)

Evaluate the unmodified base model to establish a performance floor.

```bash
bash scripts/run_pipeline.sh baseline
# or:
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --mode baseline \
  --output-dir outputs/eval_baseline
```

### Verify
```bash
cat outputs/eval_baseline/baseline_results.json | python -m json.tool
# Expect tool_selection ~65%, argument_accuracy ~50%, multi_step ~40%
```

## Phase 5: SFT Training (6-8 hours on T4)

Supervised fine-tuning with QLoRA (rank 64, LR 2e-4, 3 epochs).

### Local smoke test first (~10 min)
```bash
python src/train.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --data_dir data/processed \
  --output_dir outputs/sft_smoke \
  --num_train_epochs 1 \
  --max_steps 20 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1
```

### Full SFT run
```bash
bash scripts/run_pipeline.sh sft
# or full local command:
bash scripts/run_local_training.sh
```

### Evaluate SFT
```bash
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --sft-adapter outputs/sft/final_adapter \
  --mode sft \
  --output-dir outputs/eval_sft
```

Expected: tool_selection ~85%, argument_accuracy ~75%, multi_step ~70%.

## Phase 6: GRPO Training (1-2 hours on T4)

Reinforcement learning with binary verifiable rewards (rank 32, LR 3e-5, 50 steps).

### Prerequisites
- SFT adapter must exist at `outputs/sft/final_adapter/`
- SFT eval should show tool_selection > 60% (otherwise GRPO rewards will be all zero)

### Run GRPO
```bash
bash scripts/run_pipeline.sh grpo
# or:
python src/train_grpo.py \
  --base-model Qwen/Qwen3-8B \
  --sft-adapter outputs/sft/final_adapter \
  --data-dir data/processed \
  --output-dir outputs/grpo
```

### Monitor in W&B
Check `reward/mean` — it should climb above 0.3 within the first 20 steps.
If all rewards stay at 0 after 10 steps, see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

### Evaluate GRPO
```bash
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --sft-adapter outputs/sft/final_adapter \
  --grpo-adapter outputs/grpo/final_adapter \
  --mode grpo \
  --output-dir outputs/eval_grpo
```

Expected: tool_selection >90%, argument_accuracy >85%, multi_step >80%.

## Phase 7: Final Comparison (30 min)

Generate a side-by-side comparison table of all three stages.

```bash
bash scripts/run_pipeline.sh compare
# or:
python src/evaluate.py \
  --base-model Qwen/Qwen3-8B \
  --sft-adapter outputs/sft/final_adapter \
  --grpo-adapter outputs/grpo/final_adapter \
  --mode compare \
  --output-dir outputs/eval_compare
```

Output: markdown table showing Baseline vs SFT vs GRPO across all metrics.

## Phase 8: Publish & Cleanup

### Push to Hugging Face Hub
```bash
huggingface-cli upload dhruvanmurthy/qwen3-8b-tool-use-sft-lora outputs/sft/final_adapter
huggingface-cli upload dhruvanmurthy/qwen3-8b-tool-use-grpo-lora outputs/grpo/final_adapter
```

## One-Command Full Pipeline

To run everything end-to-end:

```bash
bash scripts/run_pipeline.sh all
```

This runs: baseline eval → SFT train → SFT eval → GRPO train → GRPO eval → comparison.

## Checklist

- [ ] Environment setup and dependencies verified
- [ ] Datasets downloaded and processed (~3,043 train samples)
- [ ] Baseline evaluation complete
- [ ] SFT training complete (loss < 1.0)
- [ ] SFT evaluation shows improvement over baseline
- [ ] GRPO training complete (reward/mean > 0.3)
- [ ] GRPO evaluation shows improvement over SFT
- [ ] Comparison table generated
- [ ] Models pushed to HF Hub
- [ ] Total training time within expectations

---

**Good luck!**

Start with Phase 1 and proceed chronologically. Each phase builds on previous.

For questions, see:
- TROUBLESHOOTING.md (for errors)
- docs/TRAINING_PLAN.md (for training details)
- README.md (for overview)
