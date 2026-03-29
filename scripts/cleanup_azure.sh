#!/bin/bash
# Cleanup Azure resources
# WARNING: This deletes all resources in the resource group

set -e

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-qwen3-finetuning}"

echo "================================================"
echo "Azure Cleanup"
echo "================================================"
echo ""
echo "WARNING: This will DELETE all resources in: $RESOURCE_GROUP"
echo ""
read -p "Are you sure? (type 'yes' to confirm): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled"
    exit 0
fi

echo ""
echo "Deleting resource group: $RESOURCE_GROUP"

az group delete \
    --name "$RESOURCE_GROUP" \
    --yes

echo ""
echo "✓ Cleanup complete"
