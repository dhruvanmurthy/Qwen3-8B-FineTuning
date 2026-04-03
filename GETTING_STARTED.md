# Getting Started — Local Setup & Verification

Step-by-step guide to run and verify the full **Baseline → SFT → GRPO** pipeline on your local machine.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| **Tinker API Key** | Required | Required |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 50 GB free | 100 GB free |
| **Python** | 3.10 | 3.11 |
| **OS** | Windows 10 / Linux | Linux (Ubuntu 22.04) |

> **Note**: No local GPU is required. Training and inference run on Tinker’s remote GPUs.
> A local GPU is only needed if you want to run local inference after training.

## Step 1 — Clone & Create Environment

```bash
git clone https://github.com/dhruvanmurthy/Qwen3-8B-FineTuning.git
cd Qwen3-8B-FineTuning
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / WSL

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 2 — Configure Credentials

```bash
cp .env.example .env
```

Edit `.env`:
```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=qwen3-8b-tool-use
TINKER_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

Login to both services:
```bash
huggingface-cli login
wandb login
```

## Step 3 — Verify Installation

Run each check. All should print OK / True:

```bash
# 1. CUDA available (optional — local GPU not required for training)
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 2. Core packages
python -c "import transformers, datasets, tinker; print('All imports OK')"

# 3. Project modules
python -c "import sys; sys.path.insert(0,'src'); from data_loader import ToolUseDataLoader; from rewards import tool_name_reward; print('Project modules OK')"

# 4. Model access (downloads tokenizer, ~1 min)
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B'); print(f'Vocab size: {t.vocab_size}')"
```

**Expected output** (all must pass):
```
CUDA: True  (or False — local GPU is optional)
All imports OK
Project modules OK
Vocab size: 151936
```

## Step 4 — Prepare Datasets

```bash
bash scripts/prepare_datasets.sh
```

### Verify

```bash
python -c "
from datasets import load_from_disk
ds = load_from_disk('data/processed')
print(f'Train:      {len(ds[\"train\"]):,} examples')
print(f'Validation: {len(ds[\"validation\"]):,} examples')
print(f'Test:       {len(ds[\"test\"]):,} examples')
print(f'Columns:    {ds[\"train\"].column_names}')
"
```

**Expected**: ~3,043 train / ~380 val / ~381 test, columns include `input_ids`, `attention_mask`, `labels`.

## Step 5 — Stage 0: Baseline Evaluation

Evaluate the base Qwen3-8B model **before any training** to establish the baseline.
Inference runs on Tinker’s remote GPUs (requires `TINKER_API_KEY`).

```bash
python src/evaluate.py \
    --mode baseline \
    --base-model Qwen/Qwen3-8B \
    --output outputs/eval_baseline.json
```

### Verify

```bash
python -m json.tool outputs/eval_baseline.json
```

**Expected**: Tool selection accuracy ~40-60%, argument accuracy ~20-30%, multi-step ~15-25%. This is the number to beat.

## Step 6 — Stage 1: SFT Training

Training runs on Tinker’s remote GPUs. No local GPU required.

### Quick smoke test (dry-run, no Tinker cost)

```bash
python src/train.py \
    --base-model sshleifer/tiny-gpt2 \
    --output-dir ./outputs/sft_test \
    --dry-run --dry-run-steps 3
```

### Verify smoke test

```bash
python -c "
import os, json
path = 'outputs/sft_test'
files = os.listdir(path)
assert 'dry_run_summary.json' in files, 'Missing dry_run_summary.json'
print('SFT smoke test: PASSED')
print('Files:', files)
"
```

### Full SFT training

```bash
bash scripts/run_local_training.sh
# Or configure directly:
python src/train.py \
    --base-model Qwen/Qwen3-8B \
    --output-dir ./outputs/sft \
    --lora-rank 64 \
    --learning-rate 2e-4 \
    --batch-size 8 \
    --num-epochs 3 \
    --max-seq-length 2048
```

### SFT evaluation

```bash
python src/evaluate.py \
    --mode sft \
    --base-model Qwen/Qwen3-8B \
    --sft-output-dir ./outputs/sft \
    --output outputs/eval_sft.json
```

### Verify

```bash
python -m json.tool outputs/eval_sft.json
```

**Expected**: Tool selection 70-85%, argument accuracy 55-70%, multi-step 50-65% — a clear jump from baseline.

## Step 7 — Stage 2: GRPO Training

This stage applies Group Relative Policy Optimization on top of the SFT checkpoint, using binary verifiable rewards. Training runs on Tinker’s remote GPUs.

### Quick smoke test (dry-run, no Tinker cost)

```bash
python src/train_grpo.py \
    --base-model sshleifer/tiny-gpt2 \
    --sft-checkpoint ./outputs/sft \
    --output-dir ./outputs/grpo_test \
    --dry-run --dry-run-steps 3
```

### Verify smoke test

```bash
python -c "
import os
path = 'outputs/grpo_test'
files = os.listdir(path)
assert 'dry_run_summary.json' in files, 'Missing dry_run_summary.json'
print('GRPO smoke test: PASSED')
print('Files:', files)
"
```

### Full GRPO training

```bash
python src/train_grpo.py \
    --base-model Qwen/Qwen3-8B \
    --sft-checkpoint ./outputs/sft \
    --output-dir ./outputs/grpo \
    --lora-rank 32 \
    --learning-rate 4e-5 \
    --max-steps 50 \
    --batch-size 16 \
    --group-size 8
```

### GRPO evaluation

```bash
python src/evaluate.py \
    --mode grpo \
    --base-model Qwen/Qwen3-8B \
    --sft-output-dir ./outputs/sft \
    --grpo-output-dir ./outputs/grpo \
    --output outputs/eval_grpo.json
```

## Step 8 — Full Three-Stage Comparison

```bash
python src/evaluate.py \
    --mode all \
    --base-model Qwen/Qwen3-8B \
    --sft-output-dir ./outputs/sft \
    --grpo-output-dir ./outputs/grpo \
    --output outputs/eval_comparison.json
```

This prints a markdown table:

```
======================================================================
STAGE COMPARISON
======================================================================
| Metric                            | baseline | sft   | grpo  |
|---|---|---|---|
| tool_selection_accuracy_api-bank  | 48.2%    | 78.5% | 91.2% |
| tool_selection_accuracy_toolbench | 42.1%    | 72.3% | 88.7% |
| argument_accuracy                 | 25.3%    | 62.1% | 84.5% |
| schema_compliance                 | 31.0%    | 85.2% | 96.1% |
| multi_step_success                | 18.7%    | 58.4% | 80.3% |
| avg_latency_ms                    | 245.00   | 251.00| 253.00|
======================================================================
```

## Step 9 — Push Model to Hugging Face Hub

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()

# Upload GRPO adapter (final model)
api.upload_folder(
    folder_path='outputs/grpo',
    repo_id='dhruvanmurthy/qwen3-8b-tool-use-grpo',
    repo_type='model',
)

# Upload SFT adapter
api.upload_folder(
    folder_path='outputs/sft',
    repo_id='dhruvanmurthy/qwen3-8b-tool-use-sft',
    repo_type='model',
)

print('Models pushed to HF Hub!')
"
```

---

## One-Command Pipeline

To run everything end-to-end (baseline → SFT → GRPO → comparison):

```bash
bash scripts/run_pipeline.sh
```

Or run individual stages:

```bash
bash scripts/run_pipeline.sh baseline   # Stage 0 only
bash scripts/run_pipeline.sh sft        # Stage 1 only
bash scripts/run_pipeline.sh grpo       # Stage 2 only
bash scripts/run_pipeline.sh compare    # Comparison only
```

---

## Troubleshooting Quick Reference

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Not applicable for Tinker training — GPU runs remotely |
| GRPO OOM on generations | Reduce `--group-size 4` and `--max-completion-length 256` |
| Loss is NaN | Reduce `--learning-rate` by 2× |
| SFT checkpoint not found at GRPO stage | Ensure `--sft-checkpoint ./outputs/sft` points to correct dir |
| W&B not logging | Run `wandb login` and set `--report-to wandb` |
| Tokenizer load error | Ensure `transformers` is up to date and model name is correct |
| Tinker connection error | Verify `TINKER_API_KEY` is set and valid |

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for the full guide.

---
**Last Updated**: March 2026

Before submitting to production:

- [ ] Run evaluation suite (src/evaluate.py)
- [ ] Check tool selection > 90%
- [ ] Verify argument accuracy > 85%
- [ ] Test error handling (graceful fallback)
- [ ] Profile latency (< 500ms/token)
- [ ] Review W&B training curves
- [ ] Confirm loss decreased monotonically
- [ ] Check no overfitting (train/val gap < 0.3)
- [ ] Validate model pushes to HF Hub
- [ ] Test inference code on fresh model
- [ ] Document any issues in TROUBLESHOOTING.md

## Success Indicators

### 🟢 Project Successful if:
1. ✅ Train → Val loss monotonically decreases
2. ✅ Final eval_loss < 1.5
3. ✅ Tool selection accuracy > 85%
4. ✅ Training time within expectations
5. ✅ Model pushes to HF Hub
6. ✅ All runs logged to W&B
7. ✅ Documentation complete

### 🟡 Project Acceptable if:
- 4-5 of above checkmarks
- Cost $100-150

### 🔴 Project Needs Work if:
- < 5 checkmarks
- eval_loss > 2.0
- Accuracy < 75%

## Collaboration & Sharing

### Share Your Results
```markdown
**Qwen3-8B Fine-tuning Results** ✨

Tool Selection: 92% accuracy
Training Cost: $40
Time: 20 hours
Model: https://huggingface.co/dhruvanmurthy/qwen3-8b-tool-use-lora

Setup guide: https://github.com/dhruvanmurthy/Qwen3-8B-FineTuning
```

### Contribute Improvements
See CONTRIBUTING.md for:
- New datasets
- Evaluation metrics
- Multi-GPU distributed training
- Cost optimizations
- Documentation

### Get Help
- **Errors**: Check TROUBLESHOOTING.md
- **Questions**: GitHub Discussions
- **Bugs**: GitHub Issues

## Cost Breakdown

```
┌──────────────────────────────────────────────┐
│ TRAINING TIME SUMMARY                        │
├──────────────────────────────────────────────┤
│ SFT Training (3 epochs)      ~18h (1× GPU)   │
│ GRPO Training (50 steps)     ~4h  (1× GPU)   │
│ Evaluation & misc            ~2h             │
│                                              │
│ TOTAL (1× GPU) :  ~24h                       │
│ TOTAL (4× GPU) :  ~7h                        │
└──────────────────────────────────────────────┘
```

Great for:
- 2-3 more full training runs
- Hyperparameter experiments
- Different datasets
- Infrastructure testing

## Resources by Role

### 👨‍💻 For ML Engineers
- Start: TRAINING_PLAN.md
- Then: src/train.py & src/data_loader.py
- Customize: configs/training.yaml

### ☁️ For DevOps/Cloud Engineers
- Start: scripts/run_pipeline.sh
- Monitor: W&B dashboards

### 📊 For Data Scientists
- Start: DATASET_STRATEGY.md
- Then: src/data_loader.py
- Experiment: Add new datasets in configs/

### 📝 For Documentation
- Start: README.md
- Expand: Each docs/*.md file
- Contribute: CONTRIBUTING.md improvements

## Final Words

This is a **complete, production-ready project** scaffold. All pieces (code, docs, configs, infrastructure) work together seamlessly.

**You are literally 1 Python command away from training a fine-tuned LLM.**

The path forward is clear:
1. ✅ Run local test (Verify setup works)
2. ✅ Prepare data (Assemble training data)
3. ✅ Train (Start training job — single or multi-GPU)
4. ✅ Evaluate & publish (Test & release)

**Estimated total effort**: 30-40 hours of elapsed time, 5-10 hours of active work

Good luck! 🚀

---

**Project Lead**: Dhruva N
**GitHub**: https://github.com/dhruvanmurthy/Qwen3-8B-FineTuning
**HuggingFace**: https://huggingface.co/dhruvanmurthy
**Last Updated**: March 2026
