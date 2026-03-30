# Training Plan & Strategy

Detailed guide to the **three-stage training pipeline**: Baseline evaluation → SFT (QLoRA) → GRPO (online RL with binary rewards).

## Overview

**Pipeline**:
1. **Stage 0 — Baseline**: Evaluate Qwen3-8B zero-shot on tool-use benchmarks
2. **Stage 1 — SFT**: QLoRA supervised fine-tuning (LoRA r=64, LR 2e-4, 3 epochs)
3. **Stage 2 — GRPO**: Group Relative Policy Optimization (LoRA r=32, LR 3e-5, batch 128, group 16, 50 steps)

**Compute**: Single A100 (1×80 GB) or A10 (1×24 GB) GPU
**Total Training Time**: ~22-28 hours on A100 (SFT + GRPO)
**Budget**: ~$44-85 for full pipeline on Azure Spot

### Stage Summary

| Stage | Method | LoRA Rank | LR | Steps | Cost (A100 Spot) |
|-------|--------|-----------|------|-------|------------------|
| Baseline | Eval only | — | — | — | ~$1.50 |
| SFT | QLoRA + Trainer | 64 | 2e-4 | ~3,750 | ~$27 |
| GRPO | QLoRA + GRPOTrainer | 32 | 3e-5 | 50 | ~$6 |

## Why QLoRA?

✅ **Memory Efficient**: 11GB vs 80GB+ for full fine-tuning
✅ **Fast**: 40% faster than LoRA, maintains quality
✅ **Cost-Effective**: Fits on consumer GPUs (A10, RTX collection)
✅ **Proven**: SOTA results on multiple benchmarks (Alpaca, MT-Bench)

### Comparison Table

| Method | VRAM | Time (A100) | Quality | Cost |
|--------|------|-------------|---------|------|
| **Full (BF16)** | 180GB | 48h | 100% | $250 |
| **LoRA** | 30GB | 20h | 98% | $80 |
| **QLoRA** | 11GB | 20h | 97% | $50 |
| **QLoRA + 4-bit** | 8GB | 22h | 96% | $45 |

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
- BF16: Stable training, CPU compat, native A100 support
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
- **adamw_8bit**: Memory-efficient optimizer, critical for QLoRA

### Batch Size & Gradient Accumulation

```yaml
effective_batch_size: 32  # 16 per-device * 2 GPUs = 32 (or 16*2 grad acc on 1 GPU)
# For single A100:
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

```python
# Pseudo-code, see src/data_loader.py for details

from datasets import load_dataset, concatenate_datasets

# Load all datasets
api_bank = load_dataset("api-bank", split="train")
toolbench = load_dataset("toolbench", split="train")
gorilla = load_dataset("gorilla", split="train")
# ... synthetic data ...

# Merge with balanced sampling
dataset = concatenate_datasets([
    api_bank.select(range(5000)),
    toolbench.select(range(15000)),
    gorilla.select(range(5000)),
    synthetic.select(range(15000))
])

# Shuffle and split
dataset = dataset.shuffle(seed=42)
split_data = dataset.train_test_split(test_size=0.2)
train_data = split_data["train"]
eval_data = split_data["test"]

# Tokenize
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

train_data = train_data.map(tokenize, batched=True)
eval_data = eval_data.map(tokenize, batched=True)
```

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
    device_map="auto",
    trust_remote_code=True
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

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs/qwen3-8b-tool-use",
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    max_grad_norm=1.0,
    eval_strategy="steps",
    eval_steps=250,
    save_steps=250,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=10,
    logging_dir="./logs",
    report_to=["wandb"],
    seed=42,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_8bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
)

trainer.train()
```

### Step 4: Save & Push

```python
# Save locally
model.save_pretrained("./outputs/qwen3-8b-tool-use-final")
tokenizer.save_pretrained("./outputs/qwen3-8b-tool-use-final")

# Push to HuggingFace Hub
model.push_to_hub(
    "dhruvanmurthy/qwen3-8b-tool-use-lora",
    token=os.getenv("HF_TOKEN")
)
```

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

## Distributed Training (Multi-GPU)

For 2+ GPUs, enable distributed training:

```yaml
ddp_world_size: 2                   # Number of GPUs
ddp_find_unused_parameters: true
ddp_backend: nccl

# Effective batch becomes: per_device_batch * num_gpus * grad_acc
# 16 * 2 * 2 = 64 (very large, may reduce grad_acc to 1)
```

```bash
# Run via torch.distributed.launch
torchrun --nproc_per_node=2 src/train.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --data_dir data/processed \
  --output_dir outputs/sft
```

## Memory Profiling

Monitor VRAM during training:

```python
# In training script
import torch
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

Expected for QLoRA (A100):
- Model weights (quantized): 2.5 GB
- LoRA params: 0.3 GB
- Optimizer states (8bit): 3-4 GB
- Batch + gradients: 6-8 GB
- **Total**: ~12-15 GB (out of 80GB)

## Checkpointing & Recovery

### Save/Resume Checkpoints

```bash
# Training automatically saves checkpoints at eval_steps
# To resume from checkpoint:
python src/train.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --data_dir data/processed \
  --output_dir outputs/sft \
  --resume-from-checkpoint ./outputs/checkpoint-2500
```

### Handling Preemption (Azure Spot)

Azure may kill your job when committed capacity is needed.

**Solution**: Enable automatic recovery:

```python
# In training script
if not training_complete:
    resume_from_last_checkpoint = True
    trainer.train(resume_from_checkpoint=True)
```

## Cost-Optimized Training Variants

### Variant 1: Budget (A10 GPU, ~$0.75/hr)

```yaml
# Reduced to fit A10 (24GB VRAM)
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
max_length: 1024
lora_r: 32
# Time: ~3 days, Cost: ~$55
```

### Variant 2: Fast (A100, ~$1.50/hr spot)

```yaml
# Optimized for speed
per_device_train_batch_size: 32        # Large batch
gradient_accumulation_steps: 1
eval_steps: 500                        # Less frequent eval
num_train_epochs: 2                    # Fewer epochs
# Time: ~14 hours, Cost: ~$25 (if fully utilized)
```

### Variant 3: Quality (A100, full)

```yaml
# Best results
per_device_train_batch_size: 16        # Balanced
gradient_accumulation_steps: 4         # Large effective batch
num_train_epochs: 5                    # Very thorough
warmup_ratio: 0.05                     # Slower warmup
# Time: ~50 hours, Cost: ~$75, Better convergence
```

## Recommendation: Balanced Setup

**Use this for most projects:**

```yaml
# configs/training.yaml (default)
per_device_train_batch_size: 16
per_device_eval_batch_size: 32
gradient_accumulation_steps: 2
num_train_epochs: 3
learning_rate: 2e-4
warmup_ratio: 0.1

# Expected: ~20h on A100, cost ~$30 (spot), eval_loss ~1.3-1.5
```

## Next Steps

1. Prepare datasets using [DATASET_STRATEGY.md](DATASET_STRATEGY.md)
2. Set up Azure using [AZURE_SETUP.md](AZURE_SETUP.md)
3. Run local test: see [GETTING_STARTED.md](../GETTING_STARTED.md)
4. Run full pipeline: `bash scripts/run_pipeline.sh`
5. Monitor in W&B: [Weights & Biases](https://wandb.ai/dhruvanmurthy/qwen3-8b-tool-use)

---

## Appendix: Stage 2 — GRPO Training Details

After SFT, the model is further improved via **Group Relative Policy Optimization** with binary verifiable rewards.

### GRPO Recipe

| Parameter | Value | Rationale |
|---|---|---|
| LoRA rank | 32 | Smaller than SFT — GRPO is a refinement |
| LR | 3e-5 | 7× lower than SFT — fine adjustments only |
| Effective batch | 128 (4 × 32) | Large batch for stable policy gradient |
| Group size | 16 | 16 completions per prompt for relative ranking |
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

GRPOTrainer receives all four functions and logs each dimension separately in W&B.

### Atropos Coordinator

The coordinator bridges reward environments with GRPOTrainer:
- Creates per-source environments (api_bank, toolbench, gorilla, synthetic)
- Each environment provides prompts and computes rewards
- Unified prompt dataset is built by sampling proportionally from all sources
- For single-machine training: in-process mode (no separate servers)
- For multi-machine: swap with the full `atropos` package from NousResearch

### GRPO Training Flow

```
1. Load base model + merge SFT adapter weights
2. Apply fresh LoRA (r=32) on top of merged model
3. Build prompt dataset via AtroposCoordinator
4. For each GRPO step:
   a. Sample batch of prompts
   b. Generate 16 completions per prompt
   c. Score each completion with binary rewards
   d. Compute GRPO loss (group-relative advantage)
   e. Update LoRA weights
5. Save GRPO adapter to outputs/grpo/
```

### When to Increase GRPO Steps

- If reward signals are noisy (most completions score 0.0): increase to 100 steps
- If schema_compliance already > 95% after SFT: 50 steps is sufficient
- If budget allows: try 100 steps and compare (adds ~$6 more)

---
**Last Updated**: March 2026
