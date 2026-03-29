# Azure ML Setup & Execution Guide

Complete guide to running the **Baseline → SFT → GRPO** pipeline on Azure ML.

---

## Prerequisites

- Azure subscription (with ~$150 credit available)
- Azure CLI installed and authenticated (`az login`)
- Python 3.10+, Git
- `.env` configured with `HF_TOKEN`, `WANDB_API_KEY`, `AZURE_SUBSCRIPTION_ID`

## Step 1 — Set Environment Variables

```bash
# PowerShell
$env:AZURE_SUBSCRIPTION_ID = "your-subscription-id"
$env:AZURE_RESOURCE_GROUP   = "qwen3-finetuning"
$env:AZURE_LOCATION         = "eastus"
$env:AZURE_WORKSPACE_NAME   = "qwen3-workspace"

# Bash / WSL
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="qwen3-finetuning"
export AZURE_LOCATION="eastus"
export AZURE_WORKSPACE_NAME="qwen3-workspace"
```

## Step 2 — Create Azure Resources

### Option A: Automated

```bash
bash scripts/setup_azure.sh
```

### Option B: Manual (step by step)

```bash
# 2a. Resource group
az group create \
  --name $AZURE_RESOURCE_GROUP \
  --location $AZURE_LOCATION

# 2b. Storage account
STORAGE_ACCOUNT="qwen3storage$(date +%s)"
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $AZURE_RESOURCE_GROUP \
  --location $AZURE_LOCATION \
  --kind BlobStorage \
  --access-tier Hot \
  --sku Standard_LRS

az storage container create \
  --name datasets \
  --account-name $STORAGE_ACCOUNT

# 2c. ML Workspace
az ml workspace create \
  --name $AZURE_WORKSPACE_NAME \
  --resource-group $AZURE_RESOURCE_GROUP \
  --location $AZURE_LOCATION \
  --display-name "Qwen3 Fine-tuning Workspace"

# 2d. GPU Compute Cluster (Spot — saves 60-70%)
az ml compute create \
  --type amlcompute \
  --name gpu-cluster \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME \
  --min-instances 0 \
  --max-instances 4 \
  --idle-time-before-scale-down 300 \
  --size Standard_ND_A100_v4 \
  --vm-priority Spot

# 2e. (Optional) Key Vault for secrets
KEYVAULT_NAME="qwen3-kv"
az keyvault create \
  --name $KEYVAULT_NAME \
  --resource-group $AZURE_RESOURCE_GROUP \
  --location $AZURE_LOCATION

az keyvault secret set --vault-name $KEYVAULT_NAME --name hf-token --value "$HF_TOKEN"
az keyvault secret set --vault-name $KEYVAULT_NAME --name wandb-token --value "$WANDB_API_KEY"
```

### Verify Resources

```bash
az resource list --resource-group $AZURE_RESOURCE_GROUP --output table
az ml compute list -g $AZURE_RESOURCE_GROUP -w $AZURE_WORKSPACE_NAME --output table
```

**Expected**: Resource group, storage account, ML workspace, and gpu-cluster all listed.

## Step 3 — Set Budget Alert ($150 cap)

```bash
az consumption budget create \
  --name qwen3-budget \
  --amount 150 \
  --resource-group $AZURE_RESOURCE_GROUP \
  --time-grain Monthly \
  --category Cost
```

## Step 4 — Upload Datasets

### 4a. Prepare locally

```bash
bash scripts/prepare_datasets.sh
```

### 4b. Upload to Azure Blob

```bash
az storage blob upload-batch \
  -d datasets \
  --account-name $STORAGE_ACCOUNT \
  -s data/processed/
```

### Verify

```bash
az storage blob list \
  --container-name datasets \
  --account-name $STORAGE_ACCOUNT \
  --output table
```

## Step 5 — Run the 3-Stage Pipeline on Azure

### Stage 0: Baseline Evaluation (remote)

Submit a lightweight job to evaluate the base model:

```bash
JOB_NAME="qwen3-baseline-$(date +%Y%m%d-%H%M%S)"

az ml job create \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME \
  --name $JOB_NAME \
  --set \
    display_name="Stage 0: Baseline Eval" \
    compute=azureml:gpu-cluster \
    command="pip install -r requirements.txt && python src/evaluate.py --mode baseline --base-model Qwen/Qwen3-8B --output outputs/eval_baseline.json" \
    environment.image=mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-devel:latest \
    code=.
```

### Stage 1: SFT Training

```bash
JOB_NAME="qwen3-sft-$(date +%Y%m%d-%H%M%S)"

az ml job create \
  --file configs/azure_config.yaml \
  --name $JOB_NAME \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME
```

Or use the script:

```bash
bash scripts/run_azure_training.sh
```

### Monitor SFT

```bash
# Stream logs
az ml job stream --name $JOB_NAME -g $AZURE_RESOURCE_GROUP -w $AZURE_WORKSPACE_NAME

# Or check W&B dashboard
# https://wandb.ai/dhruvanmurthy/qwen3-8b-tool-use
```

### Download SFT Artifacts

```bash
az ml job download \
  --name $JOB_NAME \
  --download-path ./outputs/ \
  -g $AZURE_RESOURCE_GROUP \
  -w $AZURE_WORKSPACE_NAME
```

### Stage 1: SFT Evaluation

```bash
JOB_NAME="qwen3-sft-eval-$(date +%Y%m%d-%H%M%S)"

az ml job create \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME \
  --name $JOB_NAME \
  --set \
    display_name="Stage 1: SFT Eval" \
    compute=azureml:gpu-cluster \
    command="pip install -r requirements.txt && python src/evaluate.py --mode sft --base-model Qwen/Qwen3-8B --sft-adapter outputs/sft --output outputs/eval_sft.json" \
    environment.image=mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-devel:latest \
    code=.
```

### Stage 2: GRPO Training

```bash
JOB_NAME="qwen3-grpo-$(date +%Y%m%d-%H%M%S)"

az ml job create \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME \
  --name $JOB_NAME \
  --set \
    display_name="Stage 2: GRPO Training" \
    compute=azureml:gpu-cluster \
    command="pip install -r requirements.txt && python src/train_grpo.py --sft-adapter-path outputs/sft --base-model-name Qwen/Qwen3-8B --output-dir outputs/grpo --lora-r 32 --learning-rate 3e-5 --max-steps 50 --per-device-train-batch-size 4 --gradient-accumulation-steps 32 --num-generations 16 --report-to wandb" \
    environment.image=mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-devel:latest \
    code=.
```

### Stage 3: Full Comparison

```bash
JOB_NAME="qwen3-compare-$(date +%Y%m%d-%H%M%S)"

az ml job create \
  --resource-group $AZURE_RESOURCE_GROUP \
  --workspace-name $AZURE_WORKSPACE_NAME \
  --name $JOB_NAME \
  --set \
    display_name="Stage 3: Full Comparison" \
    compute=azureml:gpu-cluster \
    command="pip install -r requirements.txt && python src/evaluate.py --mode all --base-model Qwen/Qwen3-8B --sft-adapter outputs/sft --grpo-adapter outputs/grpo --output outputs/eval_comparison.json" \
    environment.image=mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-devel:latest \
    code=.
```

## Step 6 — Download Final Artifacts

```bash
# Download all job outputs
for JOB in $(az ml job list -g $AZURE_RESOURCE_GROUP -w $AZURE_WORKSPACE_NAME --query "[].name" -o tsv | head -5); do
  echo "Downloading $JOB..."
  az ml job download --name $JOB -g $AZURE_RESOURCE_GROUP -w $AZURE_WORKSPACE_NAME --download-path ./outputs/
done
```

## Step 7 — Push Models to Hugging Face

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path='outputs/grpo', repo_id='dhruvanmurthy/qwen3-8b-tool-use-grpo', repo_type='model')
api.upload_folder(folder_path='outputs/sft', repo_id='dhruvanmurthy/qwen3-8b-tool-use-sft', repo_type='model')
print('Done!')
"
```

## Step 8 — Cleanup (Save Money!)

```bash
# Option A: Delete everything (no going back)
az group delete --name $AZURE_RESOURCE_GROUP --yes

# Option B: Keep workspace, delete compute only
az ml compute delete --name gpu-cluster -g $AZURE_RESOURCE_GROUP -w $AZURE_WORKSPACE_NAME --yes
```

---

## Cost Summary (Azure Spot)

| Stage | GPU Time | Spot Cost |
|---|---|---|
| Baseline eval | ~1 hr | ~$1.50 |
| SFT training (3 epochs) | ~18 hr | ~$27 |
| SFT eval | ~1 hr | ~$1.50 |
| GRPO training (50 steps) | ~4 hr | ~$6 |
| GRPO eval | ~1 hr | ~$1.50 |
| Full comparison eval | ~3 hr | ~$4.50 |
| Storage | 1 month | ~$2 |
| **Total** | | **~$44** |

**Budget remaining**: ~$106 for iteration.

---

## GPU Options (choose based on availability)

| GPU | VRAM | Spot $/hr | SFT Time | Fits GRPO? |
|---|---|---|---|---|
| **A100 80 GB** | 80 GB | $1.75 | 18 hr | Yes (batch 4) |
| **A100 40 GB** | 40 GB | $1.50 | 22 hr | Yes (batch 2) |
| **A10 24 GB** | 24 GB | $0.75 | 36 hr | Yes (batch 1, accum 128) |
| V100 16 GB | 16 GB | $0.92 | 65 hr | Tight — reduce group size to 8 |

**Recommendation**: A100 40 GB Spot for best cost/performance.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Quota insufficient for A100 | Request increase via Azure Portal → Subscriptions → Usage + Quotas, or use A10 |
| Spot instance preempted | Checkpointing is built-in; resubmit the same job — it resumes automatically |
| Authentication error | `az login --use-device-code` |
| Job stuck in "Preparing" | Check compute cluster status: `az ml compute show --name gpu-cluster ...` |
| Dataset upload timeout | Use `--batch-size` or `azcopy` instead of `az storage blob upload-batch` |

---

## Next Steps After Azure Training

1. Review comparison table in `outputs/eval_comparison.json`
2. If GRPO > baseline by ≥ 30% on tool selection → ship it
3. If not, iterate: more data, tune rewards, increase GRPO steps
4. Push final model to HF Hub
5. Clean up Azure resources

---
**Last Updated**: March 2026
