"""
Tinker SFT training script for Qwen3-8B tool-use fine-tuning.

Stage 1: Supervised fine-tuning on tool-use conversations.

Uses Tinker's remote GPU infrastructure for training:
  - Data preparation runs locally (CPU)
  - Forward/backward passes run on Tinker's remote GPUs
  - LoRA adapters are trained via cross-entropy loss

Requires: TINKER_API_KEY environment variable.
"""

import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import wandb
import torch
import tinker
from huggingface_hub import HfApi, create_repo, upload_folder
from tinker_cookbook.renderers import get_renderer, TrainOnWhat
from tinker_cookbook.supervised import conversation_to_datum
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.checkpoint_utils import save_checkpoint

from data_loader import ToolUseDataLoader

logger = logging.getLogger(__name__)

TOOL_USE_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools. "
    "When you need to use a tool, respond with a JSON tool call "
    "inside <tool_call> tags, like:\n"
    "<tool_call>\n"
    '{"name": "tool_name", "arguments": {"arg": "value"}}\n'
    "</tool_call>"
)


def _resolve_hf_repo_id(repo_id: str) -> str:
    """Resolve short repo names using HF_USER if available."""
    if not repo_id:
        return repo_id
    if "/" in repo_id:
        return repo_id
    hf_user = os.getenv("HF_USER")
    if hf_user:
        return f"{hf_user}/{repo_id}"
    logger.warning(
        "HF repo id '%s' has no namespace and HF_USER is not set. "
        "Using as-is; upload may fail.",
        repo_id,
    )
    return repo_id


def _init_wandb(args) -> None:
    """Initialize W&B in online or disabled mode depending on env keys."""
    has_wandb_key = bool(os.getenv("WANDB_API_KEY"))
    if not has_wandb_key:
        logger.warning("WANDB_API_KEY not set. W&B logging will be disabled.")

    wandb_entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name or f"sft-{args.base_model.split('/')[-1]}",
        "tags": ["sft", "stage1", "tool-use"],
        "config": {
            "stage": "sft",
            "base_model": args.base_model,
            "lora_rank": args.lora_rank,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "include_hf_data": args.include_hf_data,
        },
        "mode": "disabled" if not has_wandb_key else "online",
    }
    if wandb_entity:
        init_kwargs["entity"] = wandb_entity
    wandb.init(**init_kwargs)


# -----------------------------------------------------------------------
# Data loading — convert raw examples to chat conversations
# -----------------------------------------------------------------------

def load_synthetic_conversations(data_dir: str = "data/raw/synthetic") -> List[List[Dict]]:
    """Load synthetic JSONL files and convert to chat conversation format."""
    data_path = Path(data_dir)
    conversations = []

    jsonl_files = list(data_path.glob("*.jsonl")) + list(data_path.glob("*.json"))
    for fpath in jsonl_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                conv = _example_to_conversation(example)
                if conv:
                    conversations.append(conv)

    logger.info("Loaded %d synthetic conversations from %s", len(conversations), data_dir)
    return conversations


def _example_to_conversation(example: Dict) -> Optional[List[Dict]]:
    """Convert a structured synthetic example to a chat conversation."""
    instruction = example.get("instruction", "")
    if not instruction:
        return None

    # Build system prompt with available tools
    tools = example.get("tools", [])
    if tools:
        tools_json = json.dumps(tools, indent=2)
        system_content = (
            f"{TOOL_USE_SYSTEM_PROMPT}\n\n"
            f"Available tools:\n{tools_json}"
        )
    else:
        system_content = TOOL_USE_SYSTEM_PROMPT

    # Build assistant response from tool_calls
    tool_calls = example.get("tool_calls", [])
    if tool_calls:
        parts = []
        for tc in tool_calls:
            parts.append(
                "<tool_call>\n"
                + json.dumps(tc, indent=2)
                + "\n</tool_call>"
            )
        assistant_content = "\n".join(parts)
    else:
        # Fallback: extract from text field
        text = example.get("text", "")
        assistant_idx = text.find("ASSISTANT:")
        if assistant_idx >= 0:
            assistant_content = text[assistant_idx + len("ASSISTANT:"):].strip()
        else:
            return None

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": assistant_content},
    ]


def load_text_data_as_conversations(
    dataset_config: str = "configs/dataset_config.yaml",
) -> List[List[Dict]]:
    """Load HF/other datasets via data_loader, normalize to text,
    and wrap as single-turn conversations for SFT."""
    try:
        loader = ToolUseDataLoader(dataset_config)
        dataset = loader.load_all_datasets()
        dataset = loader._normalize_to_text(dataset)
        dataset = loader.preprocess(dataset)
    except Exception as e:
        logger.warning("Failed to load HF datasets: %s", e)
        return []

    conversations = []
    for example in dataset:
        text = example.get("text", "")
        if not text:
            continue
        # Wrap as a single-message conversation (train on full text)
        conversations.append([
            {"role": "system", "content": TOOL_USE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ])

    logger.info("Loaded %d text-based conversations from HF datasets", len(conversations))
    return conversations


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

async def train_sft(args):
    """Run SFT training on Tinker."""

    # ---- Logging ----
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- W&B ----
    _init_wandb(args)

    # ---- Load conversations ----
    logger.info("Loading training data...")
    conversations = load_synthetic_conversations(args.synthetic_data_dir)
    if args.include_hf_data:
        conversations += load_text_data_as_conversations(args.dataset_config)

    if not conversations:
        raise RuntimeError("No training data found. Run scripts/generate_synthetic.py first.")

    random.seed(args.seed)
    random.shuffle(conversations)

    # Split off validation set
    n_val = max(1, int(len(conversations) * 0.1))
    val_conversations = conversations[:n_val]
    train_conversations = conversations[n_val:]
    logger.info("Train: %d, Validation: %d conversations", len(train_conversations), len(val_conversations))

    # ---- Connect to Tinker ----
    logger.info("Connecting to Tinker service...")
    service_client = tinker.ServiceClient()

    logger.info("Creating LoRA training client (base=%s, rank=%d)...", args.base_model, args.lora_rank)
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.base_model,
        rank=args.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()

    # ---- Renderer ----
    try:
        renderer_name = get_recommended_renderer_name(args.base_model)
    except Exception:
        renderer_name = args.renderer_name
    renderer = get_renderer(renderer_name, tokenizer)
    logger.info("Using renderer: %s", renderer_name)

    # ---- Convert to Datums ----
    logger.info("Converting conversations to Datum objects...")
    train_datums = _conversations_to_datums(
        train_conversations, renderer, args.max_seq_length
    )
    val_datums = _conversations_to_datums(
        val_conversations, renderer, args.max_seq_length
    )
    logger.info("Train datums: %d, Val datums: %d", len(train_datums), len(val_datums))

    if not train_datums:
        raise RuntimeError("No valid training datums after conversion. Check data format.")

    # ---- Training loop ----
    adam_params = tinker.AdamParams(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.95,
    )

    n_epochs = args.num_epochs
    batch_size = args.batch_size
    total_steps = (len(train_datums) * n_epochs) // batch_size
    log_every = args.logging_steps
    save_every = args.save_steps
    wandb.config.update({
        "n_train_conversations": len(train_conversations),
        "n_val_conversations": len(val_conversations),
        "n_train_datums": len(train_datums),
        "n_val_datums": len(val_datums),
        "total_steps_planned": total_steps,
    })

    logger.info("=" * 60)
    logger.info("Tinker SFT Training — Stage 1")
    logger.info("=" * 60)
    logger.info("  Base model      : %s", args.base_model)
    logger.info("  LoRA rank       : %d", args.lora_rank)
    logger.info("  Learning rate   : %s", args.learning_rate)
    logger.info("  Batch size      : %d", batch_size)
    logger.info("  Epochs          : %d", n_epochs)
    logger.info("  Total steps     : %d", total_steps)
    logger.info("  Max seq length  : %d", args.max_seq_length)

    global_step = 0
    for epoch in range(n_epochs):
        random.shuffle(train_datums)

        for batch_start in range(0, len(train_datums), batch_size):
            batch = train_datums[batch_start : batch_start + batch_size]
            if not batch:
                continue

            # Forward + backward
            fwd_bwd_future = await training_client.forward_backward_async(
                batch, loss_fn="cross_entropy"
            )
            optim_future = await training_client.optim_step_async(adam_params)

            fwd_bwd_result = await fwd_bwd_future.result_async()
            await optim_future.result_async()

            global_step += 1

            if global_step % log_every == 0:
                loss_val = fwd_bwd_result.loss if hasattr(fwd_bwd_result, "loss") else None
                logger.info(
                    "Step %d/%d (epoch %d) | loss: %s",
                    global_step, total_steps, epoch + 1,
                    f"{loss_val:.4f}" if loss_val is not None else "N/A",
                )
                log_dict = {
                    "train/epoch": epoch + 1,
                    "train/epoch_progress": (batch_start + batch_size) / max(len(train_datums), 1),
                    "train/samples_seen": global_step * batch_size,
                }
                if loss_val is not None:
                    log_dict["train/loss"] = float(loss_val)
                wandb.log(log_dict, step=global_step)

            if save_every > 0 and global_step % save_every == 0:
                logger.info("Saving checkpoint at step %d...", global_step)
                await save_checkpoint(
                    training_client,
                    step=global_step,
                    output_dir=args.output_dir,
                )

    # ---- Final save ----
    logger.info("Saving final weights to %s...", args.output_dir)
    await save_checkpoint(
        training_client,
        step=global_step,
        output_dir=args.output_dir,
    )
    logger.info("SFT training complete! %d steps across %d epochs.", global_step, n_epochs)

    # ---- Push to Hugging Face Hub ----
    if args.hf_repo_id:
        repo_id = _resolve_hf_repo_id(args.hf_repo_id)
        await _push_to_hub_async(
            repo_id=repo_id,
            checkpoint_dir=args.output_dir,
            base_model=args.base_model,
            training_config={
                "stage": "sft",
                "base_model": args.base_model,
                "lora_rank": args.lora_rank,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "max_seq_length": args.max_seq_length,
                "total_steps": global_step,
            },
        )
    elif os.getenv("HF_TOKEN"):
        logger.info("HF_TOKEN is set but --hf-repo-id not provided. Skipping Hub upload.")

    wandb.finish()


async def _push_to_hub_async(
    repo_id: str,
    checkpoint_dir: str,
    base_model: str,
    training_config: Dict,
) -> None:
    """Push trained adapter to Hugging Face Hub.

    Args:
        repo_id: HF repo ID (e.g., "username/qwen3-sft-tool-use")
        checkpoint_dir: Local path to saved adapter
        base_model: Base model ID for documentation
        training_config: Dict with training hyperparameters
    """
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not set. Skipping Hub upload.")
            return

        logger.info("Pushing adapter to Hugging Face Hub: %s", repo_id)
        api = HfApi(token=hf_token)

        # Create repo if it doesn't exist
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
            logger.info("Repo %s already exists", repo_id)
        except Exception:
            logger.info("Creating new repo: %s", repo_id)
            create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

        # Create README
        readme_content = f"""---
license: mit
library_name: peft
---

# {repo_id}

LoRA adapter for Qwen3-8B tool-use fine-tuning.

## Model Details

- **Base Model**: {base_model}
- **Training Stage**: SFT (Supervised Fine-Tuning)
- **Task**: Tool-use instruction following
- **LoRA Rank**: {training_config.get('lora_rank', 'N/A')}
- **Training Steps**: {training_config.get('total_steps', 'N/A')}

## Training Configuration

```json
{json.dumps(training_config, indent=2)}
```

## How to Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "{base_model}"
model = AutoModelForCausalLM.from_pretrained(base_model_id, load_in_4bit=True)
model = PeftModel.from_pretrained(model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
```

## License

MIT
"""
        # Write README to checkpoint dir
        readme_path = Path(checkpoint_dir) / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
        logger.info("Created README at %s", readme_path)

        # Upload checkpoint directory
        logger.info("Uploading checkpoint directory...")
        upload_folder(
            repo_id=repo_id,
            folder_path=checkpoint_dir,
            token=hf_token,
            commit_message="Upload SFT adapter weights",
        )
        logger.info("✓ Successfully pushed to %s", repo_id)
        hub_url = f"https://huggingface.co/{repo_id}"
        logger.info("View at: %s", hub_url)

        # Log to W&B
        wandb.log({"hf_hub_url": hub_url})
        wandb.config.update({"hf_repo_id": repo_id})

    except Exception as e:
        logger.error("Failed to push to Hub: %s", e)


def _conversations_to_datums(
    conversations: List[List[Dict]],
    renderer,
    max_length: int,
) -> List[tinker.Datum]:
    """Convert chat conversations to Tinker Datum objects."""
    datums = []
    for conv in conversations:
        try:
            # Only train on the last assistant message (the tool call)
            # For user-only conversations, train on the full content
            has_assistant = any(m.get("role") == "assistant" for m in conv)
            train_on = (
                TrainOnWhat.LAST_ASSISTANT_MESSAGE
                if has_assistant
                else TrainOnWhat.ALL
            )
            datum = conversation_to_datum(
                conv, renderer, max_length=max_length, train_on_what=train_on,
            )
            datums.append(datum)
        except Exception as e:
            logger.debug("Skipping conversation: %s", e)
            continue
    return datums


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Tinker SFT training for tool-use")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--renderer-name", default="qwen3",
                   help="Fallback renderer name if auto-detect fails")
    p.add_argument("--synthetic-data-dir", default="./data/raw/synthetic")
    p.add_argument("--dataset-config", default="configs/dataset_config.yaml")
    p.add_argument("--include-hf-data", action="store_true", default=False,
                   help="Also load HF datasets (APIBench, ToolBench, etc.)")
    p.add_argument("--output-dir", default="./outputs/sft")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="qwen3-8b-tool-use",
                   help="W&B project name (requires WANDB_API_KEY env var)")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name (auto-generated if not set)")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity/team (defaults to WANDB_ENTITY env var)")
    p.add_argument("--hf-repo-id", default=None,
                   help="HF repo ID to push to (e.g., username/qwen3-sft-tool-use). Requires HF_TOKEN env var.")
    a = p.parse_args()

    # Convert argparse namespace
    class _Args:
        pass

    args = _Args()
    for k, v in vars(a).items():
        setattr(args, k.replace("-", "_"), v)

    asyncio.run(train_sft(args))


if __name__ == "__main__":
    main()
