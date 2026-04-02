# Getting Started — Local Setup & Verification

Step-by-step guide to run and verify the full **Baseline → SFT → GRPO** pipeline on your local machine.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| **GPU** | 1 × 24 GB (RTX 3090 / A10) | 1 × 16 GB (T4) |
| **RAM** | 32 GB | 64 GB |
| **Disk** | 50 GB free | 100 GB free |
| **Python** | 3.10 | 3.11 |
| **CUDA** | 11.8 | 12.1 |
| **OS** | Windows 10 / Linux | Linux (Ubuntu 22.04) |

## Step 1 — Clone & Create Environment

```bash
cd d:\MTech\Qwen3-8B-FineTuning
git init
git add .
git commit -m "Initial commit: project scaffolding"
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
```

Login to both services:
```bash
huggingface-cli login
wandb login
```

## Step 3 — Verify Installation

Run each check. All should print OK / True:

```bash
# 1. CUDA available
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 2. Core packages
python -c "import transformers, peft, trl, bitsandbytes, datasets; print('All imports OK')"

# 3. Project modules
python -c "import sys; sys.path.insert(0,'src'); from data_loader import ToolUseDataLoader; from rewards import tool_name_reward; print('Project modules OK')"

# 4. Model access (downloads tokenizer, ~1 min)
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True); print(f'Vocab size: {t.vocab_size}')"
```

**Expected output** (all must pass):
```
CUDA: True
GPU: NVIDIA Tesla T4
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

Evaluate the base Qwen3-8B model **before any training** to establish the baseline:

```bash
python src/evaluate.py \
    --mode baseline \
    --base-model Qwen/Qwen3-8B \
    --output outputs/eval_baseline.json
```

### Verify

```bash
python -c "import json; r=json.load(open('outputs/eval_baseline.json')); print(json.dumps(r['baseline'], indent=2))"
```

**Expected**: Tool selection accuracy ~40-60%, argument accuracy ~20-30%, multi-step ~15-25%. This is the number to beat.

## Step 6 — Stage 1: SFT Training

### Quick smoke test (5 min, 500 samples, 1 epoch)

```bash
python src/train.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --data_dir ./data/processed \
    --output_dir ./outputs/sft_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 1 \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --report_to none \
    --seed 42
```

### Verify smoke test

```bash
# Check adapter files exist
python -c "
import os
path = 'outputs/sft_test'
files = os.listdir(path)
assert 'adapter_config.json' in files, 'Missing adapter_config.json'
assert 'adapter_model.safetensors' in files or 'adapter_model.bin' in files, 'Missing adapter weights'
print('SFT smoke test: PASSED')
print('Files:', files)
"
```

### Full SFT training

```bash
python src/train.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --data_dir ./data/processed \
    --output_dir ./outputs/sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --eval_strategy steps \
    --eval_steps 250 \
    --save_strategy steps \
    --save_steps 250 \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --bf16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --report_to wandb \
    --seed 42
```

### SFT evaluation

```bash
python src/evaluate.py \
    --mode sft \
    --base-model Qwen/Qwen3-8B \
    --sft-adapter ./outputs/sft \
    --output outputs/eval_sft.json
```

### Verify

```bash
python -c "
import json
r = json.load(open('outputs/eval_sft.json'))
sft = r['sft']
print('SFT Results:')
for k,v in sft.items():
    if 'accuracy' in k or 'success' in k or 'compliance' in k:
        print(f'  {k}: {100*v:.1f}%')
    else:
        print(f'  {k}: {v:.2f}')
"
```

**Expected**: Tool selection 70-85%, argument accuracy 55-70%, multi-step 50-65% — a clear jump from baseline.

## Step 7 — Stage 2: GRPO Training

This stage applies Group Relative Policy Optimization on top of the SFT checkpoint, using binary verifiable rewards.

### Quick smoke test (2 min, 5 steps)

```bash
python src/train_grpo.py \
    --sft-adapter-path ./outputs/sft \
    --base-model-name Qwen/Qwen3-8B \
    --output-dir ./outputs/grpo_test \
    --max-steps 5 \
    --per-device-train-batch-size 2 \
    --gradient-accumulation-steps 2 \
    --num-generations 4 \
    --report-to none
```

### Verify smoke test

```bash
python -c "
import os
path = 'outputs/grpo_test'
files = os.listdir(path)
assert 'adapter_config.json' in files, 'Missing adapter_config.json'
print('GRPO smoke test: PASSED')
print('Files:', files)
"
```

### Full GRPO training

```bash
python src/train_grpo.py \
    --sft-adapter-path ./outputs/sft \
    --base-model-name Qwen/Qwen3-8B \
    --output-dir ./outputs/grpo \
    --lora-r 32 \
    --learning-rate 3e-5 \
    --max-steps 50 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 32 \
    --num-generations 16 \
    --report-to wandb
```

> **Note**: Effective batch = 4 × 32 = 128. On an A10 (24 GB), reduce `--per-device-train-batch-size 2` and `--gradient-accumulation-steps 64`.

### GRPO evaluation

```bash
python src/evaluate.py \
    --mode grpo \
    --base-model Qwen/Qwen3-8B \
    --sft-adapter ./outputs/sft \
    --grpo-adapter ./outputs/grpo \
    --output outputs/eval_grpo.json
```

## Step 8 — Full Three-Stage Comparison

```bash
python src/evaluate.py \
    --mode all \
    --base-model Qwen/Qwen3-8B \
    --sft-adapter ./outputs/sft \
    --grpo-adapter ./outputs/grpo \
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
| `CUDA out of memory` | Reduce `--per-device-train-batch-size`, increase `--gradient-accumulation-steps` |
| GRPO OOM on generations | Reduce `--num-generations 8` and `--max-completion-length 256` |
| Loss is NaN | Reduce `--learning-rate` by 2× |
| SFT adapter not found at GRPO stage | Ensure `--sft-adapter-path ./outputs/sft` points to correct dir |
| W&B not logging | Run `wandb login` and set `--report-to wandb` |
| Tokenizer trust_remote_code error | Ensure `transformers>=4.42.3` |

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
4. ✅ Training time < 24h
5. ✅ Total cost < $150
6. ✅ Model pushes to HF Hub
7. ✅ All runs logged to W&B
8. ✅ Documentation complete

### 🟡 Project Acceptable if:
- 5-6 of above checkmarks
- Training time 24-48h
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
