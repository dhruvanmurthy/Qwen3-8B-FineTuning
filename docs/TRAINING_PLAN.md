# Training Plan & Strategy

Detailed guide to the **three-stage training pipeline**: Baseline evaluation → SFT (LoRA) → GRPO (online RL with binary rewards).

## Overview

**Pipeline**:
1. **Stage 0 — Baseline**: Evaluate Qwen3-8B zero-shot on tool-use benchmarks
2. **Stage 1 — SFT**: LoRA supervised fine-tuning (LoRA r=64, LR 2e-4, 3 epochs)
3. **Stage 2 — GRPO**: Group Relative Policy Optimization (LoRA r=32, LR 3e-5, batch 128, group 16, 50 steps)

**Compute**: Remote GPU training via [Tinker](https://tinker.thinkingmachines.ai/) — no local GPU required
**Total Training Time**: ~6-20 hours depending on dataset size and hyperparameters

### Stage Summary

| Stage | Method | LoRA Rank | LR | Steps |
|-------|--------|-----------|----|-------|
| Baseline | Eval only | — | — | — |
| SFT | LoRA via Tinker | 64 | 2e-4 | ~3,750 |
| GRPO | LoRA + GRPO via Tinker | 32 | 4e-5 | 50 |

## Why LoRA?

✅ **Memory Efficient**: Much smaller than full fine-tuning
✅ **Fast**: Maintains quality with fewer trainable parameters
✅ **Proven**: SOTA results on multiple benchmarks (Alpaca, MT-Bench)

### Comparison Table

| Method | Trainable Params | Quality |
|--------|------------------|---------|
| **Full (BF16)** | 8B | 100% |
| **LoRA** | ~3.3M | 98% |
| **QLoRA (4-bit)** | ~3.3M | 97% |

## Hyperparameter Selection

### Model Configuration

```yaml
model_name_or_path: "Qwen/Qwen3-8B"
torch_dtype: bfloat16
load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true     # Nested quantization
bnb_4bit_quant_type: nf4             # 4-bit NormalFloat
```

**Rationale**:
- BF16: Stable training, CPU compat (emulated on T4 via autocast)
- 4-bit quantization: Minimal quality loss, maximum memory savings
- Double quantization: Save additional 0.4GB

### LoRA Configuration

```yaml
lora_r: 64                           # Rank
lora_alpha: 16                       # Scaling factor (1/4 of r = 16)
lora_dropout: 0.05                   # 5% dropout
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
bias: none                           # Don't train biases
task_type: CAUSAL_LM
```

**Rationale**:
- **r=64**: Balance between parameter efficiency (~3.3M trainable) and expressiveness
- **α=16**: Standard (α/r = 0.25) for stable scaling
- **Dropout=0.05**: Light regularization for 40k samples
- **Target modules**: All attention + FFN parts (standard for instruction tuning)
- **No bias**: Reduces parameters, minimal impact

#### Why These Numbers?

For 40k training samples:
- r=64 gives ~3.3M trainable parameters
- Effective batch size 32: ~1,250 gradient steps
- 3 epochs: ~3,750 steps total
- Recommendation: r=32-64 for this regime

### Training Loop Configuration

```yaml
output_dir: ./outputs/model_final
overwrite_output_dir: true
num_train_epochs: 3
per_device_train_batch_size: 16
per_device_eval_batch_size: 32
gradient_accumulation_steps: 2
gradient_checkpointing: true         # Save memory
ff_conv_kernel_size: null
learning_rate: 2e-4                  # Standard for fine-tuning
lr_scheduler_type: linear
warmup_ratio: 0.1                    # 10% of steps
warmup_steps: 0                      # Use ratio if non-zero
max_grad_norm: 1.0
weight_decay: 0.01                   # L2 regularization
optim: adamw_8bit                    # Explicit 8-bit Adam
seed: 42
```

**Rationale**:
- **LR = 2e-4**: Standard for instruction fine-tuning (1/10 of pre-training)
- **Warmup = 10%**: Stabilize early training, prevent divergence
- **Weight decay = 0.01**: Prevent overfitting on 40k samples
- **Gradient checkpointing**: Save 30% memory at 20% speed cost (worth it)
- **adamw_8bit**: Memory-efficient optimizer

### Batch Size & Gradient Accumulation

```yaml
effective_batch_size: 32  # 16 per-device * 1 GPU * 2 grad_acc = 32 (single GPU)
# For single T4 16GB:
per_device_train_batch_size: 16
gradient_accumulation_steps: 2
# Result: 32 effective = good balance for 40k samples
```

**Calculation**:
- Effective batch = per_device_batch × num_gpus × gradient_accumulation
- 16 × 1 × 2 = 32 effective
- ~1,250 gradient steps per epoch
- ~3,750 steps over 3 epochs

### Evaluation & Checkpointing

```yaml
eval_strategy: steps                 # Evaluate every N steps
eval_steps: 250                      # ~20 evals per epoch
save_strategy: steps
save_steps: 250
save_total_limit: 3                  # Keep only 3 best checkpoints
load_best_model_at_end: true
metric_for_best_model: eval_loss     # Minimize loss
greater_is_better: false

logging_steps: 10
logging_dir: ./logs
log_level: info
log_level_replica: warning
report_to:
  - wandb                            # Weights & Biases
  - tensorboard                      # TensorBoard (local)
```

## Training Pipeline

### Step 1: Data Preparation

```bash
# Run the full pipeline (generate synthetic + load HF datasets + preprocess)
bash scripts/prepare_datasets.sh
```

This produces Arrow-format datasets in `data/processed/`:
- `train/` (~3,043 samples)
- `validation/` (~380 samples)
- `test/` (~381 samples)

Sources loaded:
- `gorilla-llm/APIBench` (5,000 → ~1,644 after dedup)
- `tuandunghcmut/toolbench-v1` benchmark split (200 rows)
- `gorilla-llm/Berkeley-Function-Calling-Leaderboard` (258 rows)
- Synthetic via `scripts/generate_synthetic.py` (15,000 sampled → ~12,280 after dedup)

After median-target balancing (~951 per source), the final training set is ~3,043 samples.

### Step 2: Model Setup

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load base model (quantized)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=bnb_config,
  device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 3,342,336 || all params: 8,388,608 || trainable%: 0.40%
```

### Step 3: Training

Training is handled by `src/train.py` using Tinker’s remote GPU infrastructure.
The script prepares data locally, then runs forward/backward passes on Tinker.

```bash
# Full SFT training
python src/train.py \
    --base-model Qwen/Qwen3-8B \
    --output-dir ./outputs/sft \
    --lora-rank 64 \
    --learning-rate 2e-4 \
    --batch-size 8 \
    --num-epochs 3 \
    --max-seq-length 2048 \
    --logging-steps 10 \
    --save-steps 25 \
    --seed 42
```

Or use the convenience script:
```bash
bash scripts/run_local_training.sh
```

### Step 4: Save & Push

The training script automatically saves checkpoints to `--output-dir`.
Checkpoint metadata is saved in `checkpoints.jsonl` with Tinker sampler paths.

To push to HuggingFace Hub, set `HF_TOKEN` and `HF_REPO_ID` environment variables —
the script handles the upload automatically.

## Expected Training Dynamics

### Learning Curve (Typical)

```
Epoch 1:
  Step 0-250:    Loss drops from ~4.2 → 2.8 (rapid)
  Step 250-500:  Loss 2.8 → 2.1 (slower)
  Step 500-1250: Loss 2.1 → 1.8 (plateau begins)

Epoch 2:
  Step 1250-1500: Loss 1.8 → 1.7 (slow improvement)
  Step 1500-2500: Loss 1.7 → 1.5 (noisy, curriculum effect)

Epoch 3:
  Step 2500-3750: Loss 1.5 → 1.3 (convergence)
  Final: ~1.2-1.4 loss

Validation loss typically 0.1-0.2 higher than training.
```

### Metrics to Monitor (via W&B)

| Metric | Good Value | Bad Value | Action |
|--------|-----------|----------|--------|
| **Train Loss** | Decreasing | Increasing | Reduce LR |
| **Eval Loss** | 1.3-1.5 | >2.0 | Check data |
| **Grad Norm** | 0.1-1.0 | >10 | Reduce LR |
| **Learning Rate** | 2e-4 | - | Don't change |
| **GPU Memory** | 15-18GB | >20GB | Reduce batch |
| **Train % per step** | 80-90% | <50% | Profile bottleneck |

## Convergence Checks

### Early Stopping (Built-in)

Training stops if validation loss doesn't improve for 3 evals:

```yaml
early_stopping_patience: 3          # At eval_steps=250, = 750 steps
early_stopping_threshold: 0         # Any improvement counts
```

### Manual Convergence Criteria

✅ **Converged if**:
- Val loss plateaus for 2+ epochs
- Eval loss < 1.5 (for this task)
- Train/val loss diff < 0.3 (not overfitting)
- Gradient norm stable (0.1-1.0)

⚠️ **Consider Training Longer if**:
- Val loss still decreasing at epoch 3
- Train loss < 1.2 but val loss > 1.8 (underfitting)

## Common Training Issues & Fixes

### Issue 1: Loss Explodes (NaN/Inf)

```
Training losses: 4.2 → 3.8 → 3.2 → NaN
```

**Causes**: Learning rate too high, bad data
**Fixes**:
1. Reduce LR to 1e-4
2. Check for NaN in dataset
3. Clip gradients tighter (max_grad_norm=0.5)

### Issue 2: Loss Stops Decreasing

```
Loss: 1.8 → 1.7 → 1.7 → 1.7 (plateaus at epoch 1)
```

**Causes**: Too large learning rate, model capacity reached
**Fixes**:
1. Increase warmup (warmup_ratio=0.2)
2. Use cosine scheduler instead of linear
3. Train for more epochs (5-10)

### Issue 3: Overfitting (Train loss << Val loss)

```
Train: 1.2, Val: 2.0 (gap of 0.8)
```

**Causes**: Too many parameters, too long training
**Fixes**:
1. Increase dropout (lora_dropout=0.1)
2. Increase weight decay (weight_decay=0.05)
3. Stop at epoch 2 instead of 3

### Issue 4: Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Fixes** (in order):
1. Reduce per_device_train_batch_size: 16 → 8
2. Increase gradient_accumulation_steps: 2 → 4
3. Enable gradient_checkpointing: true
4. Reduce max_length: 2048 → 1024

## Single-GPU Training Note

Training runs on Tinker’s remote GPUs, so local GPU configuration is not relevant.
The Tinker infrastructure handles GPU selection and allocation automatically.

For local dry-run validation (no GPU cost):
```bash
python src/train.py \
  --base-model sshleifer/tiny-gpt2 \
  --output-dir outputs/sft_smoke \
  --dry-run --dry-run-steps 3
```

## Memory Profiling

Monitor VRAM during training:

```python
# In training script
import torch
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

Expected for LoRA training:
- Model weights (quantized): 2.5 GB
- LoRA params: 0.3 GB
- Optimizer states (8bit): 3-4 GB
- Batch + gradients: 6-8 GB
- **Total**: ~12-15 GB (out of 16GB)

## Checkpointing & Recovery

### Save/Resume Checkpoints

Checkpoints are automatically saved at intervals defined by `--save-steps`.
Metadata including Tinker sampler paths is written to `checkpoints.jsonl`.
The training script automatically resumes from the last checkpoint if one exists.

### Handling Failures

If training is interrupted, re-run the same command — it will resume from the
last saved checkpoint.

## Recommendation: Default Setup

**Use this for most runs:**

```bash
python src/train.py \
    --base-model Qwen/Qwen3-8B \
    --lora-rank 64 \
    --learning-rate 2e-4 \
    --batch-size 8 \
    --num-epochs 3
```

## Next Steps

1. Prepare datasets using [DATASET_STRATEGY.md](DATASET_STRATEGY.md)
2. Run local test: see [GETTING_STARTED.md](../GETTING_STARTED.md)
3. Run full pipeline: `bash scripts/run_pipeline.sh`
4. Monitor in W&B: [Weights & Biases](https://wandb.ai/dhruvanmurthy/qwen3-8b-tool-use)

---

## Appendix: Stage 2 — GRPO Training Details

After SFT, the model is further improved via **Group Relative Policy Optimization** with binary verifiable rewards.

### GRPO Recipe

| Parameter | Value | Rationale |
|---|---|---|
| LoRA rank | 32 | Smaller than SFT — GRPO is a refinement |
| LR | 4e-5 | Lower than SFT — fine adjustments only |
| Batch size | 16 | Prompts per GRPO step |
| Group size | 8 | 8 completions per prompt for relative ranking |
| Max steps | 50 | Short — diminishing returns after ~50 |
| Max completion | 512 tokens | Enough for 1-2 tool calls per completion |
| Max prompt | 1024 tokens | Fits system prompt + user query + tools |

### Binary Reward Functions

All rewards are programmatic (0.0 or 1.0) — no learned reward model:

| Reward Function | Signal |
|---|---|
| `tool_name_reward` | 1.0 if predicted tool name matches expected |
| `argument_match_reward` | 1.0 if all arguments exactly match |
| `schema_validation_reward` | 1.0 if output is a valid JSON tool call |
| `full_chain_reward` | 1.0 if all tools in a multi-step chain are correct |

The GRPO trainer calls `compute_rewards()` from `rewards.py`, which selects the applicable reward functions based on the available metadata and averages them. Each component is logged separately in W&B.

### Atropos Coordinator

The coordinator bridges reward environments with the GRPO training loop:
- Creates per-source environments (api_bank, toolbench, gorilla, synthetic)
- Each environment provides prompts and computes rewards
- Unified prompt dataset is built by sampling proportionally from all sources

### GRPO Training Flow

```
1. Load base model + resume from SFT checkpoint on Tinker
2. Apply fresh LoRA (r=32) on top
3. Build prompt dataset from synthetic data
4. For each GRPO step:
   a. Sample batch of prompts
   b. Generate completions per prompt (group-size per prompt)
   c. Score each completion with binary rewards
   d. Compute GRPO loss (group-relative advantage)
   e. Update LoRA weights via importance_sampling loss
5. Save GRPO checkpoint to outputs/grpo/
```

### When to Increase GRPO Steps

- If reward signals are noisy (most completions score 0.0): increase to 100 steps
- If schema_compliance already > 95% after SFT: 50 steps is sufficient
- If budget allows: try 100 steps and compare (adds ~$6 more)

---
**Last Updated**: March 2026
