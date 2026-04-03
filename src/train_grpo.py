"""
Tinker GRPO training script for Qwen3-8B tool-use fine-tuning.

Stage 2 of the training pipeline:
  Baseline (eval) -> SFT (train) -> GRPO (train)

Uses Group Relative Policy Optimization with binary verifiable rewards.
Training runs on Tinker's remote GPU infrastructure:
  - Sampling (rollouts) via SamplingClient
  - Reward grading runs locally using rewards.py
  - Policy gradient updates via forward_backward + optim_step

Requires: TINKER_API_KEY environment variable.
"""

import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path

import wandb
import torch
import tinker
from tinker import TensorData
from huggingface_hub import HfApi, create_repo, upload_folder
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.checkpoint_utils import save_checkpoint_async, get_last_checkpoint

from data_loader import ToolUseDataLoader
from rewards import (
    argument_match_reward,
    full_chain_reward,
    schema_validation_reward,
    tool_name_reward,
)

logger = logging.getLogger(__name__)

TOOL_USE_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools. "
    "When you need to use a tool, respond with a JSON tool call "
    "inside <tool_call> tags, like:\n"
    "<tool_call>\n"
    '{"name": "tool_name", "arguments": {"arg": "value"}}\n'
    "</tool_call>"
)


def _run_dry_run_grpo(args, n_prompts: int) -> None:
    """Run a local no-op GRPO path to validate script wiring without Tinker."""
    logger.warning("Dry-run mode enabled: skipping Tinker remote training.")
    steps = max(1, args.dry_run_steps)
    metrics_history = []
    for step in range(steps):
        reward = min(1.0, 0.2 + 0.1 * step)
        frac_degenerate = max(0.0, 0.6 - 0.05 * step)
        log_dict = {
            "train/reward_mean": reward,
            "train/frac_degenerate": frac_degenerate,
            "train/n_datums": args.batch_size * args.group_size,
            "train/n_degenerate_groups": int(args.batch_size * frac_degenerate),
        }
        wandb.log(log_dict, step=step)
        metrics_history.append({
            "step": step,
            "reward": reward,
            "frac_degenerate": frac_degenerate,
            "n_datums": args.batch_size * args.group_size,
        })

    metrics_path = os.path.join(args.output_dir, "grpo_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)

    summary = {
        "mode": "dry_run",
        "stage": "grpo",
        "base_model": args.base_model,
        "dry_run_steps": steps,
        "n_prompts": n_prompts,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
    }
    summary_path = Path(args.output_dir) / "dry_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote dry-run summary: %s", summary_path)


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
        "name": args.wandb_run_name or f"grpo-{args.base_model.split('/')[-1]}",
        "tags": ["grpo", "stage2", "rl", "tool-use"],
        "config": {
            "stage": "grpo",
            "base_model": args.base_model,
            "sft_checkpoint": args.sft_checkpoint,
            "lora_rank": args.lora_rank,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "group_size": args.group_size,
            "max_steps": args.max_steps,
            "max_completion_length": args.max_completion_length,
            "seed": args.seed,
        },
        "mode": "disabled" if not has_wandb_key else "online",
    }
    if wandb_entity:
        init_kwargs["entity"] = wandb_entity
    wandb.init(**init_kwargs)


# -----------------------------------------------------------------------
# Prompt dataset
# -----------------------------------------------------------------------

def build_prompt_dataset(dataset_config: str) -> list[dict]:
    """Build a list of prompt dicts for GRPO from the data-loader pipeline.

    Each dict has: prompt_text, expected_tool, expected_args, expected_chain, source
    """
    loader = ToolUseDataLoader(dataset_config)
    dataset = loader.load_all_datasets()
    dataset = loader.preprocess(dataset)

    if loader.config.get("balance_sources", True):
        dataset = loader._balance_sources(dataset)

    prompts = []
    for example in dataset:
        prompt_text = (
            example.get("instruction")
            or example.get("user_instruction")
            or example.get("query")
            or example.get("text", "")
        )
        if not prompt_text:
            continue

        # Build tool context for the prompt
        tools = example.get("tools")
        if tools:
            if isinstance(tools, str):
                tools_str = tools
            else:
                tools_str = json.dumps(tools, indent=2)
            system_content = f"{TOOL_USE_SYSTEM_PROMPT}\n\nAvailable tools:\n{tools_str}"
        else:
            system_content = TOOL_USE_SYSTEM_PROMPT

        # Extract expected values for reward grading
        tool_calls = example.get("tool_calls", [])
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except (json.JSONDecodeError, TypeError):
                tool_calls = []

        expected_tool = ""
        expected_args = ""
        expected_chain = "[]"

        if tool_calls:
            first_call = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
            expected_tool = first_call.get("name", "")
            args = first_call.get("arguments", first_call.get("parameters", {}))
            expected_args = json.dumps(args) if isinstance(args, dict) else str(args)
            if isinstance(tool_calls, list) and len(tool_calls) > 1:
                chain = [c.get("name", "") for c in tool_calls]
                expected_chain = json.dumps(chain)

        prompts.append({
            "system": system_content,
            "user": prompt_text,
            "expected_tool": expected_tool,
            "expected_args": expected_args,
            "expected_chain": expected_chain,
            "source": example.get("source", "unknown"),
        })

    logger.info("Built %d GRPO prompts", len(prompts))
    return prompts


# -----------------------------------------------------------------------
# Reward computation
# -----------------------------------------------------------------------

def compute_rewards(
    completions: list[str],
    metadata: dict,
    return_components: bool = False,
) -> "list[float] | tuple[list[float], dict[str, list[float]]]":
    """Compute average of all applicable binary reward signals.

    Args:
        completions: List of completion strings to grade.
        metadata: Dict with expected_tool, expected_args, expected_chain keys.
        return_components: If True, also return per-component reward breakdown.

    Returns:
        Mean rewards list, or (mean_rewards, component_rewards) if return_components=True.
    """
    reward_fns: dict = {"schema": schema_validation_reward}
    if metadata.get("expected_tool"):
        reward_fns["tool_name"] = tool_name_reward
    if metadata.get("expected_args"):
        reward_fns["argument_match"] = argument_match_reward
    if metadata.get("expected_chain") and metadata["expected_chain"] != "[]":
        reward_fns["chain"] = full_chain_reward

    n_fns = len(reward_fns)
    all_rewards = [0.0] * len(completions)
    component_rewards: dict[str, list[float]] = {}

    for name, fn in reward_fns.items():
        fn_rewards = fn(completions, **metadata)
        component_rewards[name] = list(fn_rewards)
        for i, r in enumerate(fn_rewards):
            all_rewards[i] += r / n_fns

    if return_components:
        return all_rewards, component_rewards
    return all_rewards


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

async def train_grpo(args):
    """Run GRPO training on Tinker."""

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- W&B ----
    _init_wandb(args)

    if args.dry_run:
        _run_dry_run_grpo(args, n_prompts=args.dry_run_prompts)
        wandb.finish()
        return

    # ---- Load prompt dataset ----
    logger.info("Building prompt dataset...")
    all_prompts = build_prompt_dataset(args.dataset_config)
    if not all_prompts:
        raise RuntimeError("No GRPO prompts found. Check data sources.")

    wandb.config.update({"n_prompts": len(all_prompts)})

    random.seed(args.seed)
    random.shuffle(all_prompts)

    # ---- Connect to Tinker ----
    logger.info("Connecting to Tinker service...")
    service_client = tinker.ServiceClient()

    # Determine start step — resume from last GRPO checkpoint if available
    grpo_ckpt = get_last_checkpoint(args.output_dir)
    start_step = 0
    if grpo_ckpt and grpo_ckpt.state_path:
        start_step = grpo_ckpt.batch or 0
        logger.info("Resuming GRPO from checkpoint: %s (step %d)", grpo_ckpt.state_path, start_step)
        training_client = await service_client.create_lora_training_client_async(
            base_model=args.base_model,
            rank=args.lora_rank,
            checkpoint=grpo_ckpt.state_path,
        )
    else:
        sft_ckpt = get_last_checkpoint(args.sft_checkpoint) if args.sft_checkpoint else None
        if sft_ckpt and sft_ckpt.state_path:
            logger.info("Starting GRPO from SFT checkpoint: %s", sft_ckpt.state_path)
            training_client = await service_client.create_lora_training_client_async(
                base_model=args.base_model,
                rank=args.lora_rank,
                checkpoint=sft_ckpt.state_path,
            )
        else:
            logger.info(
                "No valid SFT checkpoint found in '%s'. Starting GRPO from base model: %s",
                args.sft_checkpoint, args.base_model,
            )
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

    sampling_params = tinker.SamplingParams(
        max_tokens=args.max_completion_length,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = tinker.AdamParams(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.95,
    )

    logger.info("=" * 60)
    logger.info("Tinker GRPO Training — Stage 2")
    logger.info("=" * 60)
    logger.info("  Base model      : %s", args.base_model)
    logger.info("  LoRA rank       : %d", args.lora_rank)
    logger.info("  Learning rate   : %s", args.learning_rate)
    logger.info("  Batch size      : %d (problems per step)", args.batch_size)
    logger.info("  Group size      : %d (completions per problem)", args.group_size)
    logger.info("  Max steps       : %d", args.max_steps)
    logger.info("  Max completion  : %d tokens", args.max_completion_length)
    if start_step > 0:
        logger.info("  Resuming from   : step %d", start_step)

    # ---- GRPO training loop ----
    metrics_history = []

    for step in range(start_step, args.max_steps):
        # 1. Select batch of prompts
        batch_start = (step * args.batch_size) % len(all_prompts)
        batch_prompts = []
        for i in range(args.batch_size):
            idx = (batch_start + i) % len(all_prompts)
            batch_prompts.append(all_prompts[idx])

        # 2. Save current weights → sampling client
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        # 3. Sample completions for each prompt (concurrently)
        sample_coros = []
        model_inputs = []
        for prompt_info in batch_prompts:
            convo = [
                {"role": "system", "content": prompt_info["system"]},
                {"role": "user", "content": prompt_info["user"]},
            ]
            model_input = renderer.build_generation_prompt(convo)
            model_inputs.append(model_input)
            sample_coros.append(
                sampling_client.sample_async(
                    prompt=model_input,
                    num_samples=args.group_size,
                    sampling_params=sampling_params,
                )
            )

        sample_results = await asyncio.gather(*sample_coros)

        # 4. Grade completions, compute advantages, build datums
        datums = []
        step_rewards = []
        n_degenerate = 0
        step_component_rewards: dict[str, list[float]] = {}
        step_advantages: list[float] = []

        for sample_result, model_input, prompt_info in zip(
            sample_results, model_inputs, batch_prompts
        ):
            # Grade each completion in the group
            rewards_g = []
            tokens_g = []
            logprobs_g = []

            for sequence in sample_result.sequences:
                tokens_g.append(sequence.tokens)
                logprobs_g.append(sequence.logprobs)

                # Parse the completion text
                parsed_message, _ = renderer.parse_response(sequence.tokens)
                completion_text = get_text_content(parsed_message)

                # Build per-completion metadata for reward functions
                completions = [completion_text]
                metadata = {
                    "expected_tool": [prompt_info["expected_tool"]],
                    "expected_args": [prompt_info["expected_args"]],
                    "expected_chain": [prompt_info["expected_chain"]],
                }
                seq_rewards, seq_components = compute_rewards(
                    completions, metadata, return_components=True
                )
                reward = seq_rewards[0]
                rewards_g.append(reward)
                for comp_name, comp_vals in seq_components.items():
                    step_component_rewards.setdefault(comp_name, []).append(comp_vals[0])

            # Group-relative advantages (GRPO)
            mean_reward = sum(rewards_g) / len(rewards_g)
            advantages_g = [r - mean_reward for r in rewards_g]

            step_rewards.append(mean_reward)
            step_advantages.extend(advantages_g)

            # Skip degenerate groups (all same reward → zero advantage)
            if all(a == 0.0 for a in advantages_g):
                n_degenerate += 1
                continue

            # Build Datum for each completion with importance_sampling loss
            ob_len = model_input.length - 1
            for tokens, logprobs, advantage in zip(tokens_g, logprobs_g, advantages_g):
                full_input = model_input.append(
                    tinker.EncodedTextChunk(tokens=tokens[:-1])
                )
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = (
                    [0.0] * ob_len
                    + [advantage] * (full_input.length - ob_len)
                )

                datum = tinker.Datum(
                    model_input=full_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(
                            torch.tensor(target_tokens)
                        ),
                        "logprobs": TensorData.from_torch(
                            torch.tensor(padded_logprobs)
                        ),
                        "advantages": TensorData.from_torch(
                            torch.tensor(padded_advantages)
                        ),
                    },
                )
                datums.append(datum)

        # 5. Training step
        if datums:
            fwd_bwd_future = await training_client.forward_backward_async(
                datums, loss_fn="importance_sampling"
            )
            optim_future = await training_client.optim_step_async(adam_params)
            await fwd_bwd_future.result_async()
            await optim_future.result_async()

        mean_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
        frac_degenerate = n_degenerate / len(batch_prompts)

        metrics_history.append({
            "step": step,
            "reward": mean_reward,
            "frac_degenerate": frac_degenerate,
            "n_datums": len(datums),
        })

        logger.info(
            "Step %2d/%d | reward: %.3f | degenerate: %.0f%% | datums: %d",
            step, args.max_steps, mean_reward, frac_degenerate * 100, len(datums),
        )

        # ---- W&B per-step metrics ----
        log_dict: dict = {
            "train/reward_mean": mean_reward,
            "train/frac_degenerate": frac_degenerate,
            "train/n_datums": len(datums),
            "train/n_degenerate_groups": n_degenerate,
        }
        for comp_name, comp_vals in step_component_rewards.items():
            log_dict[f"train/reward_{comp_name}"] = (
                sum(comp_vals) / len(comp_vals) if comp_vals else 0.0
            )
        if step_advantages:
            adv_tensor = torch.tensor(step_advantages, dtype=torch.float32)
            log_dict["train/advantage_mean"] = float(adv_tensor.mean())
            log_dict["train/advantage_std"] = float(adv_tensor.std())
            log_dict["train/advantage_max"] = float(adv_tensor.max())
            log_dict["train/advantage_min"] = float(adv_tensor.min())
        wandb.log(log_dict, step=step)

        # Save checkpoint
        if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
            logger.info("Saving checkpoint at step %d...", step + 1)
            await save_checkpoint_async(
                training_client,
                name=f"step-{step + 1}",
                log_path=args.output_dir,
                loop_state={"batch": step + 1},
                kind="both",
            )

    # ---- Final save ----
    logger.info("Saving final GRPO weights to %s...", args.output_dir)
    await save_checkpoint_async(
        training_client,
        name="final",
        log_path=args.output_dir,
        loop_state={"batch": args.max_steps, "final": True},
        kind="both",
    )

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "grpo_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)

    logger.info("GRPO training complete! Final reward: %.3f", metrics_history[-1]["reward"])

    # ---- Push to Hugging Face Hub ----
    _hf_repo_id = os.getenv("HF_REPO_ID")
    if _hf_repo_id:
        repo_id = _resolve_hf_repo_id(_hf_repo_id + "-grpo")
        await _push_to_hub_async(
            repo_id=repo_id,
            checkpoint_dir=args.output_dir,
            base_model=args.base_model,
            sft_checkpoint=args.sft_checkpoint,
            training_config={
                "stage": "grpo",
                "base_model": args.base_model,
                "sft_checkpoint": args.sft_checkpoint,
                "lora_rank": args.lora_rank,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "group_size": args.group_size,
                "max_steps": args.max_steps,
                "final_reward": metrics_history[-1]["reward"] if metrics_history else None,
            },
        )
    elif os.getenv("HF_TOKEN"):
        logger.info("HF_TOKEN is set but HF_REPO_ID is not configured. Skipping Hub upload.")

    wandb.finish()


async def _push_to_hub_async(
    repo_id: str,
    checkpoint_dir: str,
    base_model: str,
    sft_checkpoint: str,
    training_config: dict,
) -> None:
    """Push trained GRPO adapter to Hugging Face Hub.

    Args:
        repo_id: HF repo ID (e.g., "username/qwen3-grpo-tool-use")
        checkpoint_dir: Local path to saved adapter
        base_model: Base model ID for documentation
        sft_checkpoint: Path to SFT checkpoint used as foundation
        training_config: Dict with training hyperparameters and metrics
    """
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not set. Skipping Hub upload.")
            return

        logger.info("Pushing GRPO adapter to Hugging Face Hub: %s", repo_id)
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

LoRA adapter for Qwen3-8B tool-use GRPO (Group Relative Policy Optimization).

## Model Details

- **Base Model**: {base_model}
- **Foundation**: SFT adapter from {sft_checkpoint}
- **Training Stage**: GRPO (Reinforcement Learning)
- **Task**: Tool-use instruction following (with rewards)
- **LoRA Rank**: {training_config.get('lora_rank', 'N/A')}
- **Training Steps**: {training_config.get('max_steps', 'N/A')}
- **Final Reward**: {training_config.get('final_reward', 'N/A')}

## Training Configuration

```json
{json.dumps({k: v for k, v in training_config.items() if k != 'final_reward'}, indent=2)}
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

## Notes

- This is a stage-2 (GRPO) adapter trained on top of the SFT stage.
- For inference, load the base model and apply this adapter.
- Merge with base weights for faster inference if needed.

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
            commit_message="Upload GRPO adapter weights",
        )
        logger.info("✓ Successfully pushed to %s", repo_id)
        hub_url = f"https://huggingface.co/{repo_id}"
        logger.info("View at: %s", hub_url)

        # Log to W&B
        wandb.log({"hf_hub_url": hub_url})
        wandb.config.update({"hf_repo_id": repo_id})

    except Exception as e:
        logger.error("Failed to push to Hub: %s", e)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Tinker GRPO training for tool-use")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--renderer-name", default="qwen3",
                   help="Fallback renderer name if auto-detect fails")
    p.add_argument("--sft-checkpoint", default="./outputs/sft",
                   help="Path to SFT checkpoint directory")
    p.add_argument("--dataset-config", default="configs/dataset_config.yaml")
    p.add_argument("--output-dir", default="./outputs/grpo")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Number of problems per training step")
    p.add_argument("--group-size", type=int, default=8,
                   help="Number of completions per problem (GRPO group)")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name (auto-generated if not set; WANDB_PROJECT/WANDB_ENTITY read from .env)")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Run local smoke validation without Tinker training calls")
    p.add_argument("--dry-run-steps", type=int, default=3,
                   help="Number of mock steps to log when --dry-run is enabled")
    p.add_argument("--dry-run-prompts", type=int, default=32,
                   help="Synthetic prompt count to report in --dry-run mode")
    a = p.parse_args()

    class _Args:
        pass

    args = _Args()
    for k, v in vars(a).items():
        setattr(args, k.replace("-", "_"), v)

    asyncio.run(train_grpo(args))


if __name__ == "__main__":
    main()
