#!/bin/bash
# Azure ML training submission script
# Submits training job to Azure ML compute cluster

set -e

echo "================================================"
echo "Azure ML Training Submission"
echo "================================================"

# Configuration from environment or defaults
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-qwen3-finetuning}"
WORKSPACE_NAME="${AZURE_WORKSPACE_NAME:-qwen3-workspace}"
COMPUTE_CLUSTER="${AZURE_COMPUTE_CLUSTER_NAME:-gpu-cluster}"
NUM_SAMPLES="${NUM_SAMPLES:-40000}"
EPOCHS="${EPOCHS:-3}"

# Validate environment
if [ -z "$SUBSCRIPTION_ID" ]; then
    echo "Error: AZURE_SUBSCRIPTION_ID not set"
    echo "Set it with: export AZURE_SUBSCRIPTION_ID=<your-id>"
    exit 1
fi

echo "Configuration:"
echo "  Subscription: $SUBSCRIPTION_ID"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Workspace: $WORKSPACE_NAME"
echo "  Compute: $COMPUTE_CLUSTER"
echo "  Samples: $NUM_SAMPLES"
echo "  Epochs: $EPOCHS"

# Set active subscription
az account set --subscription "$SUBSCRIPTION_ID"

# Create job name with timestamp
JOB_NAME="qwen3-finetune-$(date +%Y%m%d-%H%M%S)"

echo ""
echo "Submitting job: $JOB_NAME"

# Submit job
az ml job create \
    --file configs/azure_config.yaml \
    --name "$JOB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --set \
        compute="azureml:${COMPUTE_CLUSTER}"

echo ""
echo "Job submitted successfully!"
echo "Job name: $JOB_NAME"
echo ""
echo "Monitor progress:"
echo "  az ml job stream --name $JOB_NAME -g $RESOURCE_GROUP -w $WORKSPACE_NAME"
echo ""
echo "View in portal:"
echo "  https://ml.azure.com"
echo ""

# Optionally stream logs immediately
if [ "${STREAM_LOGS:-false}" = "true" ]; then
    echo "Streaming logs..."
    az ml job stream --name "$JOB_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --workspace-name "$WORKSPACE_NAME"
fi
