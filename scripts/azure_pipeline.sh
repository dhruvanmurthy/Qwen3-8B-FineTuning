#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# azure_pipeline.sh — Build, push, run, and (optionally) tear down the
#                     Qwen3-8B pipeline container on Azure Container Instances.
#
# Prerequisites:
#   - Azure CLI (az) logged in:  az login
#   - .env file in repo root with TINKER_API_KEY, WANDB_API_KEY, etc.
#
# Usage:
#   bash scripts/azure_pipeline.sh [STAGE] [OPTIONS]
#
#   STAGE (default: all)
#     all | baseline | sft | grpo | compare
#
#   Options:
#     --no-build      Skip docker build+push (use last pushed image)
#     --no-teardown   Leave the container running after completion
#     --dry-run       Print az commands without executing them
#
# Examples:
#   bash scripts/azure_pipeline.sh all
#   bash scripts/azure_pipeline.sh baseline --no-build
#   BENCHMARK_FILTER="schema_compliance multi_step" bash scripts/azure_pipeline.sh sft
# ---------------------------------------------------------------------------
set -e

# ---- Parse args ------------------------------------------------------------
STAGE="${1:-all}"
shift || true

BUILD=true
TEARDOWN=true
DRY_RUN=false
DETACH=false

for arg in "$@"; do
    case "$arg" in
        --no-build)    BUILD=false ;;
        --no-teardown) TEARDOWN=false ;;
        --dry-run)     DRY_RUN=true ;;
        --detach)      DETACH=true; TEARDOWN=false ;;
    esac
done

_run() {
    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] $*"
    else
        "$@"
    fi
}

# ---- Load .env -------------------------------------------------------------
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

# ---- Required env vars -----------------------------------------------------
: "${TINKER_API_KEY:?TINKER_API_KEY is required}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

# ---- Parse .env into regular and secure arrays ----------------------------
# Lines matching *_KEY, *_TOKEN, *_SECRET, *_PASSWORD go to --secure-environment-variables
# (hidden in Azure portal/CLI output). Everything else is plain.
declare -a ENV_PLAIN=()
declare -a ENV_SECURE=()
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and blank lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line//[[:space:]]/}" ]] && continue
        # Must be KEY=VALUE format
        [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
        var_name="${line%%=*}"
        raw_value="${line#*=}"
        # Strip inline comments (e.g. `value  # comment`) — non-standard but common in .env files
        clean_value="${raw_value%%  #*}"
        clean_value="${clean_value%% #*}"
        line="${var_name}=${clean_value}"
        if [[ "$var_name" =~ (KEY|TOKEN|SECRET|PASSWORD)$ ]]; then
            ENV_SECURE+=("$line")
        else
            ENV_PLAIN+=("$line")
        fi
    done < "$ENV_FILE"
fi

# ---- Configuration (override via env) -------------------------------------
RESOURCE_GROUP="${RESOURCE_GROUP:-qwen3-pipeline-rg}"
LOCATION="${LOCATION:-eastus}"
ACR_NAME="${ACR_NAME:-qwen3pipelineacr}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-pipeline-$(date +%Y%m%d-%H%M%S)}"
CPU="${CPU:-2}"
MEMORY="${MEMORY:-4}"   # GB — comfortable headroom; ~$0.71 per 6-hour run

IMAGE_FULL="${ACR_NAME}.azurecr.io/qwen3-pipeline:${IMAGE_TAG}"

# Group all W&B runs from this container launch together so they're easy
# to identify. W&B respects WANDB_RUN_GROUP automatically — no code changes needed.
ENV_PLAIN+=("WANDB_RUN_GROUP=${CONTAINER_NAME}")

echo "============================================================="
echo " Qwen3-8B Pipeline — Azure Container Instances"
echo "============================================================="
echo "  Resource group  : $RESOURCE_GROUP"
echo "  Location        : $LOCATION"
echo "  ACR             : $ACR_NAME"
echo "  Image           : $IMAGE_FULL"
echo "  Container       : $CONTAINER_NAME"
echo "  CPU / RAM       : ${CPU} vCPU / ${MEMORY} GB"
echo "  Stage           : $STAGE"
echo "  Build           : $BUILD"
echo "  Teardown        : $TEARDOWN"
if [ -n "$BENCHMARK_FILTER" ]; then
    echo "  Benchmark filter: $BENCHMARK_FILTER"
fi
echo "============================================================="

# ---- Step 1: Ensure resource group exists ----------------------------------
echo ""
echo ">>> Step 1: Resource group"
_run az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

# ---- Step 2: Deploy ACR via Bicep (idempotent) -----------------------------
echo ""
echo ">>> Step 2: Provision ACR (Basic)"
_run az deployment group create \
    --resource-group "$RESOURCE_GROUP" \
    --template-file "infra/main.bicep" \
    --parameters acrName="$ACR_NAME" location="$LOCATION" \
    --output none

# ---- Step 3: Build and push image ------------------------------------------
if [ "$BUILD" = "true" ]; then
    echo ""
    echo ">>> Step 3: Build + push image via ACR Tasks (no local Docker needed)"
    _run az acr build \
        --registry "$ACR_NAME" \
        --image "qwen3-pipeline:${IMAGE_TAG}" \
        --file Dockerfile \
        .
else
    echo ""
    echo ">>> Step 3: Skipping build (--no-build)"
fi

# ---- Step 4: Get ACR credentials -------------------------------------------
echo ""
echo ">>> Step 4: Fetch ACR credentials"
if [ "$DRY_RUN" = "false" ]; then
    # Ensure admin user is enabled (may not be if ACR pre-existed the Bicep deploy)
    az acr update --name "$ACR_NAME" --admin-enabled true --output none
    ACR_USERNAME=$(az acr credential show \
        --name "$ACR_NAME" \
        --query "username" \
        --output tsv | tr -d '\r\n')
    ACR_PASSWORD=$(az acr credential show \
        --name "$ACR_NAME" \
        --query "passwords[0].value" \
        --output tsv | tr -d '\r\n')
else
    ACR_USERNAME="$ACR_NAME"
    ACR_PASSWORD="<dry-run-password>"
fi

# ---- Step 5: Create and start the container instance -----------------------
echo ""
echo ">>> Step 5: Create ACI container (${CPU} vCPU / ${MEMORY} GB)"

# If a container with this name already exists (e.g. fixed CONTAINER_NAME + --no-teardown),
# delete it first so az container create doesn't fail.
if [ "$DRY_RUN" = "false" ]; then
    EXISTING=$(az container show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --query "name" --output tsv 2>/dev/null || true)
    if [ -n "$EXISTING" ]; then
        echo "    Existing container '$CONTAINER_NAME' found — deleting before recreate..."
        az container delete \
            --resource-group "$RESOURCE_GROUP" \
            --name "$CONTAINER_NAME" \
            --yes --output none
    fi
fi

# Build the command string for the pipeline
PIPELINE_CMD="bash scripts/run_pipeline.sh ${STAGE}"

_run az container create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --image "$IMAGE_FULL" \
    --os-type Linux \
    --registry-login-server "${ACR_NAME}.azurecr.io" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --cpu "$CPU" \
    --memory "$MEMORY" \
    --restart-policy Never \
    --command-line "$PIPELINE_CMD" \
    ${ENV_PLAIN:+--environment-variables "${ENV_PLAIN[@]}"} \
    ${ENV_SECURE:+--secure-environment-variables "${ENV_SECURE[@]}"} \
    --output none

echo ""
if [ "$DETACH" = "true" ]; then
    echo ">>> Container launched (detached). Your local machine can now be closed."
    echo ""
    echo "  Monitor live metrics : https://wandb.ai/${WANDB_ENTITY:-your-entity}/${WANDB_PROJECT:-qwen3-8b-tool-use}"
    echo "  Check container state: az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query 'containers[0].instanceView.currentState' -o table"
    echo "  Stream logs anytime  : az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --follow"
    echo "  Delete when done     : az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes"
    echo ""
    exit 0
fi

echo ">>> Container started. Streaming logs (Ctrl+C to detach — container keeps running)..."
_run az container logs \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --follow || true

# ---- Step 6: Wait for container to finish ----------------------------------
if [ "$DRY_RUN" = "false" ]; then
    echo ""
    echo ">>> Step 6: Waiting for container to reach terminal state..."
    az container wait \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --custom "instanceView.state=='Terminated'" \
        --interval 30 \
        --timeout 28800  # 8h max

    EXIT_CODE=$(az container show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --query "containers[0].instanceView.currentState.exitCode" \
        --output tsv)

    echo ""
    if [ "$EXIT_CODE" = "0" ]; then
        echo ">>> Pipeline completed successfully (exit 0)."
    else
        echo ">>> Pipeline FAILED (exit code: $EXIT_CODE)."
        echo "    View full logs: az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
    fi
fi

# ---- Step 7: Tear down container to stop billing ---------------------------
if [ "$TEARDOWN" = "true" ] && [ "$DRY_RUN" = "false" ]; then
    echo ""
    echo ">>> Step 7: Deleting container instance (stops billing)..."
    az container delete \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --yes \
        --output none
    echo ">>> Container deleted. ACR and resource group preserved for next run."
elif [ "$TEARDOWN" = "false" ]; then
    echo ""
    echo ">>> --no-teardown: container preserved."
    echo "    To delete manually: az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes"
fi

echo ""
echo "============================================================="
echo " Done."
echo " W&B run: https://wandb.ai/${WANDB_ENTITY:-your-entity}/${WANDB_PROJECT:-qwen3-8b-tool-use}"
echo "============================================================="
