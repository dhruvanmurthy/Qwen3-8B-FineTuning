# Step-by-Step Execution Plan

Complete guide to execute the Qwen3-8B fine-tuning project from start to finish.

Pipeline: **Baseline Eval → SFT Training → GRPO Training → Final Comparison**

## Phase 1: Local Setup (1 hour)

### Step 1.1: Create Virtual Environment
```bash
# PowerShell on Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

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
#   AZURE_SUBSCRIPTION_ID=<your_id>  (only for Azure runs)
```

### Step 1.4: Verify Installation
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoTokenizer; print('Transformers OK')"
python -c "from peft import LoraConfig; print('PEFT OK')"
python -c "from trl import GRPOConfig; print('TRL OK')"
```

## Phase 2: Azure Setup — skip if running locally (1 hour)

### Step 2.1: Create Azure Resources
```bash
bash scripts/setup_azure.sh
```

### Step 2.2: Verify Resources
```bash
az ml compute list --resource-group qwen3-finetuning --workspace-name qwen3-workspace
```

### Step 2.3: Create W&B Project
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
from datasets import load_dataset
ds = load_dataset('parquet', data_files='data/processed/hf_dataset/train-*.parquet')
print(f'Train samples: {len(ds[\"train\"])}')
print(f'Columns: {ds[\"train\"].column_names}')
"
```

Expected: ~32,000 train samples.

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

## Phase 5: SFT Training (6-8 hours on A100)

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

## Phase 6: GRPO Training (1-2 hours on A100)

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

### Cleanup Azure Resources (if applicable)
```bash
bash scripts/cleanup_azure.sh
```

## One-Command Full Pipeline

To run everything end-to-end:

```bash
bash scripts/run_pipeline.sh all
```

This runs: baseline eval → SFT train → SFT eval → GRPO train → GRPO eval → comparison.

## Checklist

- [ ] Environment setup and dependencies verified
- [ ] Datasets downloaded and processed (~40k samples)
- [ ] Baseline evaluation complete
- [ ] SFT training complete (loss < 1.0)
- [ ] SFT evaluation shows improvement over baseline
- [ ] GRPO training complete (reward/mean > 0.3)
- [ ] GRPO evaluation shows improvement over SFT
- [ ] Comparison table generated
- [ ] Models pushed to HF Hub
- [ ] Azure resources cleaned up
- [ ] Total cost < $150

### Step 6.3: Log Results to W&B
```bash
python << 'EOF'
import json
import wandb

with open("outputs/evaluation_results.json") as f:
    results = json.load(f)

wandb.log({
    "eval/tool_selection": results["tool_selection_accuracy"],
    "eval/argument_f1": results["argument_accuracy"],
    "eval/multi_step": results["multi_step_success"],
    "eval/error_handling": results["error_handling"],
    "eval/latency_ms": results["latency_ms"],
})
EOF
```

## Phase 7: Publishing (Day 4 - 30 mins)

### Step 7.1: Push to Hugging Face Hub
```bash
python << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login
import os

# Login
login(token=os.getenv("HF_TOKEN"))

# Load model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model = PeftModel.from_pretrained(base_model, "outputs/model_final")

# Push adapter to Hub
model.push_to_hub(
    "dhruvanmurthy/qwen3-8b-tool-use-lora",
    token=os.getenv("HF_TOKEN")
)

# Push tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
tokenizer.push_to_hub(
    "dhruvanmurthy/qwen3-8b-tool-use-lora",
    token=os.getenv("HF_TOKEN")
)
EOF
```

Check: https://huggingface.co/dhruvanmurthy/qwen3-8b-tool-use-lora

### Step 7.2: Create Version Tag
```bash
# Tag current state
git tag v1.0.0
git push origin v1.0.0
```

This triggers GitHub Actions to:
- Run tests
- Create release
- Build documentation

Monitor: https://github.com/dhruvanmurthy/Qwen3-8B-FineTuning/actions

## Phase 8: Cleanup (When Done)

### Step 8.1: Stop Azure Compute (Save Money!)
```bash
# Stop compute clusters (don't delete workspace)
az ml compute stop \
  --name gpu-cluster \
  --resource-group qwen3-finetuning
```

### Step 8.2: Keep Model, Delete Raw Data
```bash
# Delete raw data (not needed after processing)
rm -rf data/raw/*

# Keep processed data if you plan to retrain
# rm -rf data/processed/*
```

### Step 8.3: Full Cleanup (Optional - Deletes Everything)
```bash
bash scripts/cleanup_azure.sh
# Confirm: yes
```

## Quick Reference: Complete Timeline

| Phase | Duration | Action | Cost |
|-------|----------|--------|------|
| 1 | 1h | Local setup | $0 |
| 2 | 1h | Azure resources | $2 |
| 3 | 2h | Data prep | $2 |
| 4 | 2h | Local test | $3 |
| 5 | 24h | Full training | $30 |
| 6 | 1h | Evaluation | $1 |
| 7 | 0.5h | Publishing | $0 |
| **Total** | **31.5h** | **Complete** | **~$40** |

## Expected Results

### Metrics
```
Tool Selection:    92%+ (Target: >90%)
Argument Accuracy: 87%+ (Target: >85%)
Multi-Step Success: 81%+ (Target: >80%)
Error Handling:    89%+ (Target: >85%)
```

### Training Dynamics
```
Epoch 1 Loss:  4.2 → 1.8
Epoch 2 Loss:  1.8 → 1.3
Epoch 3 Loss:  1.3 → 1.1
Final Val Loss: ~1.2
```

### Cost
```
Training:    $30 (20h @ $1.50/hr spot)
Storage:     $2 (100GB data)
Evaluation:  $1 (compute)
Total:       ~$40 (within budget!)
```

## Troubleshooting

### If Training Fails
```bash
# Check job status
az ml job show --name <job-name> -g qwen3-finetuning

# View logs
az ml job stream --name <job-name> -g qwen3-finetuning

# Resume from checkpoint
python src/train.py --resume_from_checkpoint outputs/checkpoint-XXX
```

### If Evaluation Low
1. Train longer (add epochs)
2. Increase learning rate
3. Check dataset quality
4. Review TROUBLESHOOTING.md

### If Over Budget
- Use A10 GPU (3x cheaper)
- Reduce num_samples (30k → 20k)
- Train fewer epochs (3 → 2)
- See docs/BUDGET_OPTIMIZATION.md

## Success Criteria

✅ **Project successful if**:
- Tool selection accuracy > 90%
- Training completes in < 24h
- Total cost < $150
- Model pushes to HF Hub
- GitHub repo has documentation
- W&B tracks all runs

## Next Steps After Success

1. **Share Model**: Twitter, Reddit, Discord
2. **Write Blog**: Medium article on approach
3. **Iterate**: Try new datasets/methods
4. **Optimize**: Reduce latency with ONNX export
5. **Contribute**: Open-source improvements

---

**Good luck! 🚀**

Start with Phase 1 and proceed chronologically. Each phase builds on previous.

For questions, see:
- TROUBLESHOOTING.md (for errors)
- docs/TRAINING_PLAN.md (for training details)
- README.md (for overview)
