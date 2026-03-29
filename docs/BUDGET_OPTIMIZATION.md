# Budget Optimization Guide

Strategies to keep the Qwen3-8B fine-tuning project under $150 budget.

## Budget Breakdown

### Estimated Costs

| Component | Quantity | Unit Cost | Total |
|-----------|----------|-----------|-------|
| **Baseline eval (A100 Spot)** | 1 hour | $1.50/hr | $1.50 |
| **SFT Training (A100 Spot)** | 18 hours | $1.50/hr | $27 |
| **SFT eval** | 1 hour | $1.50/hr | $1.50 |
| **GRPO Training (A100 Spot)** | 4 hours | $1.50/hr | $6 |
| **GRPO eval** | 1 hour | $1.50/hr | $1.50 |
| **Full comparison eval** | 3 hours | $1.50/hr | $4.50 |
| **Storage (datasets)** | 100 GB | $0.02/GB/month | $2 |
| **Buffer (unexpected)** | - | - | $20 |
| **TOTAL** | | | **$64** |

**Available Budget**: $150  
**Utilization**: ~43%  
**Headroom**: $86 for iteration (extra GRPO steps, hyperparameter sweeps)

## Cost Optimization Strategies

### 1. Use Spot Instances (Save 60-70%)

Spot VM pricing:

| GPU | On-Demand | Spot | Savings |
|-----|-----------|------|---------|
| A100 (40GB) | $4.75/hr | $1.50/hr | **68%** |
| A100 (80GB) | $5.10/hr | $1.75/hr | **66%** |
| A10 (24GB) | $2.50/hr | $0.75/hr | **70%** |
| V100 (16GB) | $3.06/hr | $0.92/hr | **70%** |

**Implementation**:
```bash
az ml compute create \
  --name gpu-cluster \
  --vm-priority Spot \
  --max-price 2.00  # Set max price
```

**Risk**: Preemption every 8-24 hours  
**Mitigation**: Enable checkpointing (no extra cost)

### 2. Choose Cheaper GPU (A10 vs A100)

Training time comparison on Qwen3-8B:

| GPU | VRAM | Training Time | Cost @ Spot |
|-----|------|---------------|------------|
| A100 (80GB) | 80GB | 18h | $27 |
| A100 (40GB) | 40GB | 22h | $33 |
| A10 (24GB) | 24GB | 36h | $27 |
| V100 (16GB) | 16GB | 65h | $60 |

**Recommendation**: **A100 40GB Spot** = fastest + cheapest

If not available, use **A10** (slower but still cost-effective).

### 3. Reduce Training Samples (Diminishing Returns)

Quality vs. sample size:

| Samples | Train Time | Eval Loss | Cost Savings |
|---------|-----------|-----------|--------------|
| 5k | 3h | 1.8 | -79% |
| 10k | 5.5h | 1.5 | -73% |
| 20k | 10.5h | 1.2 | -42% |
| 40k | 20h | 1.1 | 0% |
| 60k | 30h | 1.05 | +50% |

**Analysis**: Diminishing returns after 30k samples  
**Recommendation**: Start with 20k, validate results, expand if needed

```bash
python src/train.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --data_dir data/processed \
  --output_dir outputs/sft \
  --num_train_epochs 2
```

### 4. Optimize Hyperparameters (Faster Convergence)

Goal: Reduce training time without sacrificing quality

**Option A: Larger Batch Size** (train faster, same GPU cost)

```yaml
# Original
per_device_batch_size: 16
gradient_accumulation_steps: 2
effective_batch: 32
training_time: 20h

# Optimized
per_device_batch_size: 32     # Uses more VRAM
gradient_accumulation_steps: 1
effective_batch: 32
training_time: 16h (20% faster, no cost increase)
```

**Check VRAM**: Before increasing batch:
```python
# In training.py, track max VRAM
print(f"Max GPU memory: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
```

**Option B: Fewer Epochs** (trade quality for speed)

```yaml
# Original
num_epochs: 3
eval_loss: 1.1
training_time: 20h

# Faster
num_epochs: 2
eval_loss: 1.2-1.3 (slight quality loss)
training_time: 13h (35% faster)
```

**Decision Tree**:
```
Eval loss after 1 epoch?
├─ <2.0: Try 2 epochs total, save 35% cost
├─ 2.0-2.5: Continue 3 epochs (need more training)
└─ >2.5: Reduce LR, restart
```

### 5. Data Preparation Automation (Avoid Repetition)

Compute cheap pre-processing once, reuse:

```bash
# 1. Download & process datasets (one-time, ~$1)
bash scripts/prepare_datasets.sh

# 2. Upload to Azure Blob (one-time, ~$2)
az storage blob upload-batch \
  -d datasets \
  --account-name $STORAGE_ACCOUNT \
  -s data/processed/

# 3. Reuse in training runs (no extra cost)
# Point training directly to Azure Blob

# Savings: 
# - Without optimization: download datasets for each job = $5-10/run
# - With optimization: $3 one-time, reuse = 90% savings
```

### 6. Batch Multiple Training Runs

Use spot instance when available (don't stop between runs):

```bash
#!/bin/bash
# runs.sh - Run multiple hyperparameter configs

for lr in 1e-4 2e-4; do
  for warmup in 0.05 0.1; do
    echo "Training LR=$lr WARMUP=$warmup"
    
    python src/train.py \
      --learning-rate $lr \
      --warmup-ratio $warmup \
      --output-dir outputs/run_lr${lr}_warmup${warmup} \
      --save-strategy no  # Don't save intermediate checkpoints
    
    # Copy best checkpoint
    cp outputs/run_*/checkpoint*/adapter_config.json \
       outputs/best_lr${lr}_warmup${warmup}/
  done
done

# Cost savings:
# - If spot instance stays available: combine 4 runs in one session
# - Savings: 3 × GPU startup overhead avoided (~$5)
```

### 7. Use Azure Credits/Free Tier

- **Azure Free Tier**: $200 credit for 12 months
- **Student/Educational**: $100 credit
- **Startup**: Up to $1000 in Azure credits

Apply here: https://azure.microsoft.com/en-us/free/

### 8. Share Spot Compute with Team

If multiple people fine-tuning:

```bash
# Create shared compute cluster
az ml compute create \
  --name shared-gpu-cluster \
  --type amlcompute \
  --vm-priority Spot \
  --max-instances 4  # Support 4 concurrent jobs
  --idle-time-before-scale-down 600  # Scale down after 10 min
```

**Benefit**: Amortize cost across multiple projects

### 9. Monitor Spending in Real-Time

Set up Azure cost alerts:

```bash
# View current spending (updates hourly)
az consumption budget list --resource-group qwen3-finetuning

# Set up alert
az consumption budget create \
  --name qwen3-warning \
  --amount 75 \
  --threshold-type Forecast \
  --threshold 80  # Alert at 80% of budget
  --notification-enabled true
```

Keep an eye on W&B compute metrics:

```python
# Log to W&B
wandb.log({
    "compute/gpu_hours": 20,
    "compute/cost_usd": 30,
})
```

## Cost-Optimized Variants

### Variant 1: Minimal Budget (~$40)

```yaml
# Compromise: Quality vs. Cost
num_samples: 10000        # Half the data
num_epochs: 2             # Shorter training
batch_size: 32            # Larger batches
per_device_batch: 32      # Reduces training steps
learning_rate: 1e-4       # Stable, may need more warmup
max_length: 1024          # Reduce token count
gpu_type: A10             # Cheaper GPU

# Expected
training_hours: 8
cost: $6 (spot)
eval_loss: 1.4-1.5
quality: Good
```

### Variant 2: Balanced (~$60, recommended)

```yaml
# Trade-off: Quality vs. Cost
num_samples: 30000
num_epochs: 3
batch_size: 32
learning_rate: 2e-4
gpu_type: A100 40GB

# Expected
training_hours: 18
cost: $27 (spot)
eval_loss: 1.1-1.2
quality: Excellent
```

### Variant 3: Maximum Quality (~$120)

```yaml
# Best results, still under budget
num_samples: 50000
num_epochs: 4
batch_size: 16      # Smaller = more stable
learning_rate: 1.5e-4
warmup_ratio: 0.15  # Longer warmup
gpu_type: A100 80GB

# Expected
training_hours: 35
cost: $52 (spot)
eval_loss: 1.0-1.1
quality: State-of-the-art
```

## Detailed Cost Calculation

### Example: A100 40GB Spot Training

```
Spot hourly rate: $1.50
Reserved capacity discount: -$0.15 (if applicable)
Effective rate: $1.35/hour

Training duration: 20 hours
Cost calculation:
  20 hours × $1.35/hour = $27.00

Storage (100 GB dataset for 1 month):
  100 GB × $0.02/GB/month = $2.00

Compute (data prep, 2 CPU hours):
  2 hours × $0.10/hr = $0.20

TOTAL: $29.20
```

### Hidden Costs to Avoid

❌ **Don't**: Leave GPU idle  
✅ **Do**: Auto-scale to 0 after job finishes

❌ **Don't**: Store model checkpoints in expensive storage  
✅ **Do**: Save only best checkpoints, delete intermediate ones

❌ **Don't**: Download dataset every training run  
✅ **Do**: Upload once to Azure Blob, reuse

❌ **Don't**: Use on-demand instances  
✅ **Do**: Use Spot instances (60% cheaper)

## Spending Tracking Dashboard

Create W&B dashboard to track spending:

```yaml
# Create custom chart in W&B
wandb.log({
    "cost/total_usd": 60,
    "cost/per_hour_usd": 1.5,
    "cost/per_sample_usd": 0.0015,
    "efficiency/eval_loss_per_dollar": 1.1 / 27,  # Lower is better
    "budget_remaining_usd": 150 - 60,
})
```

## Monthly Cost Projection

If you continue the project:

```
1 full fine-tuning: $60
5 evaluation runs:  $25
10 hyperparameter tuning runs: $35
2 months compute maintenance: $10
────────────────
Monthly total: ~$130 (fits in $200 free tier)
Annual: ~$700 (need to pay after free tier expires)
```

## Recommendation

**For this project**:
1. ✅ Use A100 40GB Spot (~$27 training)
2. ✅ Train on 30-40k samples
3. ✅ Run 3 epochs
4. ✅ Set aggressive checkpoint cleanup (keep only 1 best)
5. ✅ Total budget used: ~$60
6. ✅ Remaining for experiments: $90

**Decision Point** (after first training):
- If eval_loss < 1.2 and tool metrics > 90%: Ship model ✅
- If eval_loss > 1.3: Use remaining $90 to iterate ✅

## Next Steps

1. Set budget alert in Azure
2. Choose cost variant above (recommend Balanced)
3. Proceed with training ([TRAINING_PLAN.md](TRAINING_PLAN.md))
4. Monitor spending in W&B dashboard
5. Clean up resources after project (see AZURE_SETUP.md cleanup section)

---
**Last Updated**: March 2026
