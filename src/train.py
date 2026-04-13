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
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import wandb
import tinker
from huggingface_hub import HfApi, create_repo, upload_folder
from tinker_cookbook.renderers import get_renderer, TrainOnWhat
from tinker_cookbook.supervised import conversation_to_datum
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.checkpoint_utils import save_checkpoint_async, get_last_checkpoint

from constants import TOOL_USE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


async def _create_lora_training_client(service_client, base_model: str, rank: int,
                                       checkpoint_path: Optional[str] = None,
                                       load_optimizer: bool = True):
    """Create a LoRA training client, optionally loading a checkpoint.

    Creates a fresh client from base_model, then restores weights (and
    optionally optimizer state) from checkpoint_path if provided.
    """
    client = await service_client.create_lora_training_client_async(
        base_model=base_model,
        rank=rank,
    )
    if checkpoint_path:
        if load_optimizer:
            load_future = await client.load_state_with_optimizer_async(checkpoint_path)
        else:
            load_future = await client.load_state_async(checkpoint_path)
        await load_future.result_async()
    return client


def _run_dry_run_sft(args, n_train: int, n_val: int) -> None:
    """Run a local no-op training path to validate script wiring without Tinker."""
    logger.warning("Dry-run mode enabled: skipping Tinker remote training.")
    steps = max(1, args.dry_run_steps)
    for step in range(1, steps + 1):
        mock_loss = max(0.0, 1.0 - 0.05 * step)
        wandb.log(
            {
                "train/epoch": 1,
                "train/loss": mock_loss,
                "train/samples_seen": step * args.batch_size,
            },
            step=step,
        )

    summary = {
        "mode": "dry_run",
        "stage": "sft",
        "base_model": args.base_model,
        "dry_run_steps": steps,
        "n_train_conversations": n_train,
        "n_val_conversations": n_val,
    }
    summary_path = Path(args.output_dir) / "dry_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote dry-run summary: %s", summary_path)


def _epoch_batches(
    train_datums: List[tinker.Datum],
    batch_size: int,
    seed: int,
    epoch: int,
) -> List[List[tinker.Datum]]:
    """Create deterministic epoch batches so resume can skip prior work exactly."""
    indices = list(range(len(train_datums)))
    random.Random(seed + epoch).shuffle(indices)
    batches: List[List[tinker.Datum]] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batches.append([train_datums[i] for i in batch_indices])
    return batches


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

    wandb_entity = os.getenv("WANDB_ENTITY")
    init_kwargs = {
        "project": os.getenv("WANDB_PROJECT", "qwen3-8b-tool-use"),
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

    if not conversations:
        raise RuntimeError("No training data found. Run scripts/generate_synthetic.py first.")

    random.seed(args.seed)
    random.shuffle(conversations)

    # Split off validation set
    n_val = max(1, int(len(conversations) * 0.1))
    val_conversations = conversations[:n_val]
    train_conversations = conversations[n_val:]
    logger.info("Train: %d, Validation: %d conversations", len(train_conversations), len(val_conversations))

    if args.dry_run:
        _run_dry_run_sft(
            args,
            n_train=len(train_conversations),
            n_val=len(val_conversations),
        )
        wandb.finish()
        return

    # ---- Connect to Tinker ----
    logger.info("Connecting to Tinker service...")
    service_client = tinker.ServiceClient()

    # Resume from an existing checkpoint if one exists in the output dir
    resume_ckpt = get_last_checkpoint(args.output_dir)
    if resume_ckpt and resume_ckpt.state_path:
        logger.info("Resuming SFT from checkpoint: %s (step %s)", resume_ckpt.state_path, resume_ckpt.batch)
        logger.warning("Data will replay from the beginning of epoch 1. Weights are restored.")
        training_client = await _create_lora_training_client(
            service_client=service_client,
            base_model=args.base_model,
            rank=args.lora_rank,
            checkpoint_path=resume_ckpt.state_path,
        )
    else:
        logger.info("Creating new LoRA training client (base=%s, rank=%d)...", args.base_model, args.lora_rank)
        training_client = await _create_lora_training_client(
            service_client=service_client,
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
    logger.info("Train datums: %d", len(train_datums))

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
    steps_per_epoch = math.ceil(len(train_datums) / batch_size)
    total_steps = steps_per_epoch * n_epochs
    log_every = args.logging_steps
    save_every = args.save_steps
    wandb.config.update({
        "n_train_conversations": len(train_conversations),
        "n_val_conversations": len(val_conversations),
        "n_train_datums": len(train_datums),
        "steps_per_epoch": steps_per_epoch,
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

    start_step = resume_ckpt.batch if resume_ckpt and resume_ckpt.batch else 0
    if start_step >= total_steps:
        logger.warning(
            "Resume step (%d) is already >= total planned steps (%d). Skipping training loop.",
            start_step,
            total_steps,
        )

    global_step = start_step
    start_epoch = start_step // steps_per_epoch if steps_per_epoch else 0
    start_batch_in_epoch = start_step % steps_per_epoch if steps_per_epoch else 0

    for epoch in range(start_epoch, n_epochs):
        epoch_batches = _epoch_batches(train_datums, batch_size, args.seed, epoch)
        batch_start_idx = start_batch_in_epoch if epoch == start_epoch else 0

        for batch_idx in range(batch_start_idx, len(epoch_batches)):
            batch = epoch_batches[batch_idx]

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
                    "train/epoch_progress": (batch_idx + 1) / max(len(epoch_batches), 1),
                    "train/samples_seen": global_step * batch_size,
                }
                if loss_val is not None:
                    log_dict["train/loss"] = float(loss_val)
                wandb.log(log_dict, step=global_step)

            if save_every > 0 and global_step % save_every == 0:
                logger.info("Saving checkpoint at step %d...", global_step)
                await save_checkpoint_async(
                    training_client,
                    name=f"step-{global_step}",
                    log_path=args.output_dir,
                    loop_state={"batch": global_step, "epoch": epoch + 1},
                    kind="both",
                )
        start_batch_in_epoch = 0

    # ---- Final save ----
    logger.info("Saving final weights to %s...", args.output_dir)
    await save_checkpoint_async(
        training_client,
        name="final",
        log_path=args.output_dir,
        loop_state={"batch": global_step, "final": True},
        kind="both",
    )
    logger.info("SFT training complete! %d steps across %d epochs.", global_step, n_epochs)

    # ---- Push to Hugging Face Hub ----
    _hf_repo_id = os.getenv("HF_REPO_ID")
    if _hf_repo_id:
        repo_id = _resolve_hf_repo_id(_hf_repo_id + "-sft")
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
        logger.info("HF_TOKEN is set but HF_REPO_ID is not configured. Skipping Hub upload.")

    wandb.finish()


async def _push_to_hub_async(
    repo_id: str,
    checkpoint_dir: str,
    base_model: str,
    training_config: Dict,
) -> None:
    """Push trained SFT adapter to Hugging Face Hub."""
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not set. Skipping Hub upload.")
            return

        logger.info("Pushing adapter to Hugging Face Hub: %s", repo_id)
        api = HfApi(token=hf_token)

        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
            logger.info("Repo %s already exists", repo_id)
        except Exception:
            logger.info("Creating new repo: %s", repo_id)
            create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

        readme_content = f"""---
license: mit
library_name: peft
---

# {repo_id}

LoRA adapter for Qwen3-8B tool-use fine-tuning (SFT stage).

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

## License

MIT
"""
        readme_path = Path(checkpoint_dir) / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        upload_folder(
            repo_id=repo_id,
            folder_path=checkpoint_dir,
            token=hf_token,
            commit_message="Upload SFT adapter weights",
        )
        logger.info("✓ Successfully pushed to %s", repo_id)
        hub_url = f"https://huggingface.co/{repo_id}"
        logger.info("View at: %s", hub_url)
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
    p.add_argument("--output-dir", default="./outputs/sft")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name (auto-generated if not set; WANDB_PROJECT/WANDB_ENTITY read from .env)")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Run local smoke validation without Tinker training calls")
    p.add_argument("--dry-run-steps", type=int, default=3,
                   help="Number of mock steps to log when --dry-run is enabled")
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
