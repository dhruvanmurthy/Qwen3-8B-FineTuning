#!/bin/bash
# Setup Azure ML resources for Qwen3-8B fine-tuning
# Creates resource group, storage, workspace, and compute clusters

set -e

echo "================================================"
echo "Azure ML Resource Setup"
echo "================================================"

# Configuration
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-qwen3-finetuning}"
LOCATION="${AZURE_LOCATION:-eastus}"
WORKSPACE_NAME="${AZURE_WORKSPACE_NAME:-qwen3-workspace}"
STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-qwen3storage$(date +%s)}"

# Validate inputs
if [ -z "$SUBSCRIPTION_ID" ]; then
    echo "Error: AZURE_SUBSCRIPTION_ID not set"
    exit 1
fi

echo "Configuration:"
echo "  Subscription: $SUBSCRIPTION_ID"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  Workspace: $WORKSPACE_NAME"
echo "  Storage Account: $STORAGE_ACCOUNT"

# Set active subscription
echo ""
echo "Setting active subscription..."
az account set --subscription "$SUBSCRIPTION_ID"

# Create resource group
echo ""
echo "Creating resource group..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION"

# Create storage account
echo ""
echo "Creating storage account..."
az storage account create \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --kind BlobStorage \
    --access-tier Hot \
    --sku Standard_LRS

# Create container
echo ""
echo "Creating storage container..."
az storage container create \
    --name datasets \
    --account-name "$STORAGE_ACCOUNT"

# Create ML workspace
echo ""
echo "Creating ML workspace..."
az ml workspace create \
    --name "$WORKSPACE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --display-name "Qwen3 Fine-tuning Workspace"

# Create GPU compute cluster
echo ""
echo "Creating GPU compute cluster..."
az ml compute create \
    --type amlcompute \
    --name gpu-cluster \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --min-instances 0 \
    --max-instances 4 \
    --idle-time-before-scale-down 300 \
    --size Standard_ND_A100_v4 \
    --vm-priority Spot

# Create CPU compute cluster for evaluation
echo ""
echo "Creating CPU compute cluster..."
az ml compute create \
    --type amlcompute \
    --name cpu-cluster \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --min-instances 0 \
    --max-instances 2 \
    --idle-time-before-scale-down 300 \
    --size Standard_D4s_v3 \
    --vm-priority LowPriority

echo ""
echo "================================================"
echo "✓ Resources created successfully"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Verify resources: az resource list -g $RESOURCE_GROUP"
echo "2. Prepare datasets: bash scripts/prepare_datasets.sh"
echo "3. Submit training: bash scripts/run_azure_training.sh"
echo ""
