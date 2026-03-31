# Troubleshooting Guide

Common issues and solutions for Qwen3-8B fine-tuning project.

## Azure Issues

### Problem: Authentication Failures

**Error**:
```
AuthenticationError: Operation returned an invalid status 'Unauthorized'
```

**Solution**:
```bash
# Re-authenticate
az logout
az login

# Or use device code
az login --use-device-code

# Verify authentication
az account show
```

### Problem: Insufficient GPU Quota

**Error**:
```
Insufficient quota available for SKU STANDARD_NC4AS_T4_V3 in region southindia
```

**Solutions**:
1. Request quota increase via Azure Portal:
   - Home → Subscriptions → Usage + Quotas
   - Filter by "NCASv3_T4"
   - Request Increase

2. Use different region:
   ```bash
   az ml compute create --size STANDARD_NC4AS_T4_V3 \
     --location westus3  # Try different region
   ```

3. Use a larger T4 SKU or different GPU family:
   ```bash
   # 4× T4 variant (more vCPUs, same T4 GPU)
   az ml compute create --size Standard_NC64as_T4_v3 \
     --tier dedicated
   ```

### Problem: Job Failures on Dedicated Compute

**Error**:
```
Job failed due to compute node issues
```

**Solutions**:
1. Enable checkpointing (automatic resume):
   ```yaml
   load_best_model_at_end: true
   resume_from_checkpoint: auto
   ```

2. If you switch to Spot VMs for cost savings, add preemption handling:
   ```bash
   az ml compute create --tier low_priority  # Cheaper, slight preemption risk
   ```

3. Use low-priority instead of dedicated for cost savings:
   ```bash
   az ml compute create --tier low_priority  # More affordable
   ```

## Local Training Issues

### Problem: Out of Memory (OOM)

**Error**:
```
CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions** (in order):
1. Reduce batch size:
   ```yaml
   per_device_train_batch_size: 8  # Instead of 16
   gradient_accumulation_steps: 4   # Increase to maintain effective batch
   ```

2. Enable gradient checkpointing:
   ```yaml
   gradient_checkpointing: true
   ```

3. Reduce max sequence length:
   ```yaml
   max_seq_length: 1024  # Instead of 2048
   ```

4. Use smaller model variant
5. Use A10 GPU instead of attempting on consumer GPU

### Problem: Loss Explodes (NaN/Inf)

**Error**:
```
Training loss: 4.2 → 3.8 → NaN
```

**Causes & Solutions**:
1. Learning rate too high:
   ```yaml
   learning_rate: 1e-4  # Reduce from 2e-4
   ```

2. Bad data (NaN values):
   ```python
   # Check dataset for NaN
   dataset.map(
       lambda x: {"has_nan": np.isnan(x["input_ids"]).any()}
   )
   ```

3. Clip gradients tighter:
   ```yaml
   max_grad_norm: 0.5  # Instead of 1.0
   ```

### Problem: Model Not Using GPU

**Symptoms**:
- Training very slow
- GPU utilization 0%
- "WARNING: Torch not compiled with CUDA"

**Solution**:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall torch with CUDA support
pip install torch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Dataset Issues

### Problem: ArrowInvalid When Loading HF Datasets

**Error**:
```
ArrowInvalid: JSON parse error: Column(/api_data/api_arguments) changed from object to string in row 1
```

**Cause**: Some HF JSON datasets have mixed types in the same column across rows (e.g., APIBench `api_arguments` switches between object and string). The default `datasets` JSON loader uses PyArrow, which requires consistent types.

**Solution**: The `data_loader.py` `_load_from_hub` method automatically falls back to per-file pandas loading when the default loader fails. No manual action needed. If adding a new HF source with this issue, ensure `data_files` is set in `dataset_config.yaml` to load specific files.

### Problem: HF Dataset Has Wrong Config/Split Name

**Error**:
```
ValueError: Unknown split "train". Available splits: ['g1_instruction']
```

**Cause**: Many community HF datasets use non-standard config and split names.

**Solution**: Use `get_dataset_config_names()` and inspect the dataset card:
```python
from datasets import get_dataset_config_names, load_dataset
configs = get_dataset_config_names("tuandunghcmut/toolbench-v1")
print(configs)  # ['default', 'benchmark']
ds = load_dataset("tuandunghcmut/toolbench-v1", "benchmark")
print(ds)  # Shows available splits
```

Then update `configs/dataset_config.yaml` with the correct `config` and `split` values.

### Problem: Dataset Files Too Large

**Error**:
```
Segmentation fault when loading dataset
```

**Solution**:
```python
# Process in smaller batches
dataset.map(
    tokenize_function,
    batched=True,
    batch_size=100  # Smaller batch during processing
)
```

### Problem: Tokenization Timeout

**Error**:
```
Timeout while downloading dataset
```

**Solution**:
```bash
# Increase timeout
export HF_DATASETS_DOWNLOAD_TIMEOUT=3600  # 1 hour

# Or download manually
python << 'EOF'
from datasets import load_dataset
dataset = load_dataset("gorilla-llm/APIBench", data_files="torchhub_train.json", split="train", cache_dir="./hf_cache")
EOF
```

### Problem: Duplicate Examples After Loading

**Symptoms**:
- Dataset larger than expected
- Performance plateaus early

**Solution**:
```python
# Deduplicate
import hashlib
seen = set()
unique_indices = []

for i, example in enumerate(dataset):
    hash_val = hashlib.md5(
        json.dumps(example).encode()
    ).hexdigest()
    if hash_val not in seen:
        seen.add(hash_val)
        unique_indices.append(i)

dataset = dataset.select(unique_indices)
```

## Training Issues

### Problem: Loss Plateaus (Stops Decreasing)

**Error**:
```
Loss: 1.8 → 1.7 → 1.7 → 1.7 (no improvement)
```

**Causes & Solutions**:
1. Learning rate too high (overshot minimum):
   ```yaml
   learning_rate: 1e-4  # Reduce by 2x
   ```

2. Not enough training steps:
   ```yaml
   num_epochs: 5  # Increase from 3
   ```

3. Model capacity insufficient (unlikely for Qwen3-8B)

4. Check validation metrics:
   - If eval loss improving but train plateau: normal (good generalization)
   - If both plateau: increase LR warmup or reduce LR

### Problem: Overfitting (Gap between Train/Val Loss)

**Symptoms**:
```
Train loss: 1.2
Val loss: 1.9
Gap: 0.7 (too large!)
```

**Solutions**:
1. Increase dropout:
   ```yaml
   lora_dropout: 0.1  # Instead of 0.05
   ```

2. Increase weight decay:
   ```yaml
   weight_decay: 0.05  # Instead of 0.01
   ```

3. Stop training earlier:
   ```yaml
   num_epochs: 2  # Instead of 3
   ```

4. Add more diverse data

### Problem: GPU Memory Utilization Low

**Symptoms**:
- GPU memory usage < 50%
- Training slow despite GPU available

**Solutions**:
1. Increase batch size:
   ```yaml
   per_device_train_batch_size: 32  # If room available
   ```

2. Use larger model (try base model more):
```bash
# Check current utilization
nvidia-smi  # GPU memory usage

# Run with verbose logging
python src/train.py ... --log_level debug
```

## Hugging Face Hub Issues

### Problem: Can't Push Model to Hub

**Error**:
```
HTTPError 401: Invalid token
```

**Solution**:
```bash
# Login to HF
huggingface-cli login

# Input your HF token (from https://huggingface.co/settings/tokens)

# Or set env variable
export HF_TOKEN="hf_xxxxx"
```

### Problem: Model Not Downloading from Hub

**Error**:
```
FileNotFoundError: model.safetensors not found
```

**Solution**:
```bash
# Clear local project cache and retry
rm -rf ./hf_cache/

# Or clear global HF cache
rm -rf ~/.cache/huggingface/

# Project uses local cache dir
export HF_HOME=./hf_cache
```

## Weights & Biases Issues

### Problem: W&B Not Logging

**Error**:
```
WARNING: Failed to initialize W&B
```

**Solution**:
```bash
# Login to W&B
wandb login

# Input your API key from https://wandb.ai/settings/tokens

# Or set env variable
export WANDB_API_KEY="xxxxx"

# Also set project
export WANDB_PROJECT="qwen3-8b-tool-use"
```

### Problem: Logs Not Appearing in W&B

**Solution**:
1. Check internet connection
2. Verify project name correct
3. Disable offline mode:
   ```yaml
   report_to: ["wandb"]
   ```
4. Check for errors:
   ```python
   import wandb
   wandb.init(mode="disabled")  # Test connectivity
   ```

## GRPO Training Issues

### Problem: GRPO Rewards Stuck at Zero {#grpo-rewards-stuck-at-zero}

**Symptoms**:
- All four reward functions return 0.0 for every generation
- `reward/mean` stays near 0 in W&B after 10+ steps

**Causes & Solutions**:
1. SFT adapter not good enough — the model can't produce any valid tool calls:
   ```bash
   # First check SFT quality
   python src/evaluate.py --base-model Qwen/Qwen3-8B \
     --sft-adapter outputs/sft/final_adapter --mode sft
   # If tool_selection < 60%, re-train SFT with more data or epochs
   ```

2. Prompt format mismatch — GRPO prompts must match the format used during SFT:
   ```bash
   # Verify prompt format in data_loader.py prepare_grpo_prompts()
   # The system prompt and chat template must be identical
   ```

3. Learning rate too high — policy diverges immediately:
   ```yaml
   # In configs/grpo_config.yaml, reduce LR
   learning_rate: 1e-5  # Instead of 3e-5
   ```

### Problem: OOM During GRPO (16 Generations)

**Error**:
```
CUDA out of memory during generation step
```

GRPO generates `num_generations=16` completions per prompt, which uses 16× inference memory.

**Solutions**:
1. Reduce generations per group:
   ```yaml
   num_generations: 8  # Instead of 16
   ```

2. Reduce per-device batch size:
   ```yaml
   per_device_train_batch_size: 2  # Instead of 4
   gradient_accumulation_steps: 64  # Keep effective batch at 128
   ```

3. Reduce max generation length:
   ```yaml
   max_completion_length: 256  # Instead of 512
   ```

### Problem: SFT Adapter Not Found for GRPO

**Error**:
```
FileNotFoundError: outputs/sft/final_adapter/adapter_config.json
```

**Solution**:
```bash
# Verify SFT output exists
ls outputs/sft/final_adapter/

# If using Azure, download artifacts first
az ml job download --name <sft-job-name> --download-path outputs/sft/

# Then pass the correct path
python src/train_grpo.py --sft-adapter outputs/sft/final_adapter ...
```

### Problem: TRL Version Incompatibility

**Error**:
```
ImportError: cannot import name 'GRPOConfig' from 'trl'
```

**Solution**:
```bash
# GRPOTrainer requires trl >= 0.14.0
pip install "trl>=0.14.0"

# Verify
python -c "from trl import GRPOConfig, GRPOTrainer; print('TRL OK')"
```

## Evaluation Issues

### Problem: Model Generation Too Slow

**Symptoms**:
- Evaluation very slow
- Taking hours for 100 samples

**Solutions**:
1. Use smaller max_length:
   ```python
   model.generate(..., max_length=256)  # Instead of 2048
   ```

2. Use greedy decoding (faster):
   ```python
   model.generate(..., do_sample=False, num_beams=1)
   ```

3. Batch inference:
   ```python
   # Evaluate in batches of 8-16 instead of 1
   outputs = model.generate(batch_encodings, ...)
   ```

### Problem: GRPO Evaluation Fails to Load Stacked Adapters

**Error**:
```
RuntimeError: size mismatch for model.layers...
```

The GRPO adapter must be loaded on top of the *merged* SFT model, not stacked.

**Solution**: The evaluate script handles this automatically by merging SFT first.
If running manually, merge SFT before loading GRPO:
```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
sft_model = PeftModel.from_pretrained(base, "outputs/sft/final_adapter")
merged = sft_model.merge_and_unload()  # Merge SFT into weights
grpo_model = PeftModel.from_pretrained(merged, "outputs/grpo/final_adapter")
```

## General Debugging

### Enable Verbose Logging

```bash
# Most verbose
TRANSFORMERS_VERBOSITY=debug python src/train.py ...

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profile Memory Usage

```python
import torch

# During training
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Check System Resources

```bash
# GPU status
nvidia-smi

# CPU/RAM
free -h  # Linux
Get-ComputerInfo | Select-Object TotalPhysicalMemory  # Windows

# Disk space
df -h  # Linux
Get-Volume  # Windows
```

## Getting Help

1. **Check Logs**:
   - Local: `logs/training.log`
   - Azure: `az ml job stream --name <job-name>`
   - W&B: https://wandb.ai/dhruvanmurthy/qwen3-8b-tool-use

2. **Search Issues**:
   - https://github.com/dhruvanmurthy/Qwen3-8B-FineTuning/issues
   - https://github.com/huggingface/transformers/issues
   - https://github.com/microsoft/peft/issues

3. **Community Help**:
   - Hugging Face Discussions: https://discuss.huggingface.co/
   - Stack Overflow: [transformers] tag
   - Discord: HF community server

4. **Report Bug**:
   ```bash
   # Collect debug info
   python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
   nvidia-smi

   # Create GitHub issue with:
   # - Error message
   # - Minimal reproducible code
   # - Versions of all packages
   # - System info (OS, GPU, RAM)
   ```

---
**Last Updated**: March 2026
