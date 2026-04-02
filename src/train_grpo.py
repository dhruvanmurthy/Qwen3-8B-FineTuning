"""
GRPO training script for Qwen3-8B tool-use fine-tuning.

Stage 2 of the training pipeline:
  Baseline (eval) -> SFT (train) -> GRPO (train)

Uses Group Relative Policy Optimization with binary verifiable rewards
and an Atropos-pattern coordinator for environment-trainer bridging.

Recipe: LoRA rank 32, LR 3e-5, effective batch 128, group 16, 50 steps.
"""

import argparse
import logging
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer

from data_loader import ToolUseDataLoader
from environments import AtroposCoordinator, ToolUseEnvironment
from rewards import (
    argument_match_reward,
    full_chain_reward,
    schema_validation_reward,
    tool_name_reward,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------

def load_sft_model(base_model_name: str, sft_adapter_path: str):
    """Load base model and merge the SFT adapter into its weights."""

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model: %s", base_model_name)

    # In distributed mode (torchrun), pin each process to its local GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        device_map = {"":  local_rank}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if os.path.isdir(sft_adapter_path):
        logger.info("Merging SFT adapter from: %s", sft_adapter_path)
        model = PeftModel.from_pretrained(model, sft_adapter_path)
        model = model.merge_and_unload()
        logger.info("SFT adapter merged")
    else:
        logger.warning(
            "SFT adapter not found at %s — training GRPO from base model",
            sft_adapter_path,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# -----------------------------------------------------------------------
# Prompt dataset
# -----------------------------------------------------------------------

def build_prompt_dataset(dataset_path: str):
    """Load or build a prompt-only dataset for GRPO.

    Tries loading a prepared dataset from *dataset_path*.  Falls back to
    building prompts from the data-loader pipeline.
    """

    if os.path.isdir(dataset_path):
        try:
            ds = load_from_disk(dataset_path)
            # DatasetDict -> use train split
            if hasattr(ds, "keys") and "train" in ds.keys():
                ds = ds["train"]
            if "prompt" in ds.column_names:
                logger.info(
                    "Loaded prompt dataset from disk: %d rows", len(ds)
                )
                return ds
        except Exception:
            pass

    logger.info("Building prompt dataset via data_loader pipeline")
    loader = ToolUseDataLoader("configs/dataset_config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", trust_remote_code=True
    )
    return loader.prepare_grpo_prompts(tokenizer)


# -----------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------

def setup_environments(dataset):
    """Create per-source Atropos environments and coordinator."""

    sources = (
        set(dataset["source"])
        if "source" in dataset.column_names
        else {"all"}
    )

    environments = []
    for source_name in sorted(sources):
        env_data = (
            dataset
            if source_name == "all"
            else dataset.filter(lambda x: x["source"] == source_name)
        )

        reward_fns = [schema_validation_reward]
        if "expected_tool" in env_data.column_names:
            reward_fns.append(tool_name_reward)
        if "expected_args" in env_data.column_names:
            reward_fns.append(argument_match_reward)
        if "expected_chain" in env_data.column_names:
            reward_fns.append(full_chain_reward)

        environments.append(
            ToolUseEnvironment(env_data, source_name, reward_fns)
        )
        logger.info(
            "  env '%s': %d prompts, %d reward fns",
            source_name,
            len(env_data),
            len(reward_fns),
        )

    return AtroposCoordinator(environments=environments)


# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------

def train_grpo(args):
    """Run GRPO training."""

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    logger.info("=" * 60)
    logger.info("GRPO Training — Stage 2")
    logger.info("=" * 60)
    logger.info("  LoRA rank       : %d", args.lora_r)
    logger.info("  LR              : %s", args.learning_rate)
    logger.info("  Effective batch : %d", eff_batch)
    logger.info("  Group size      : %d", args.num_generations)
    logger.info("  Max steps       : %d", args.max_steps)

    # 1. Model ----------------------------------------------------------
    model, tokenizer = load_sft_model(args.base_model_name, args.sft_adapter_path)

    # 2. Dataset --------------------------------------------------------
    dataset = build_prompt_dataset(args.prompt_dataset_path)

    # 3. Atropos coordinator --------------------------------------------
    coordinator = setup_environments(dataset)
    prompt_dataset = coordinator.build_prompt_dataset()

    # 4. GRPO LoRA config (rank 32) ------------------------------------
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. GRPOConfig -----------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        seed=args.seed,
        save_steps=10,
        save_total_limit=3,
    )

    # 6. Reward functions from coordinator ------------------------------
    reward_funcs = coordinator.get_reward_funcs()

    # 7. Train ----------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_funcs,
        train_dataset=prompt_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting GRPO training …")
    trainer.train()

    # 8. Save -----------------------------------------------------------
    logger.info("Saving to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("GRPO training complete!")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="GRPO training for tool-use")
    p.add_argument("--sft-adapter-path", default="./outputs/sft")
    p.add_argument("--base-model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--prompt-dataset-path", default="./data/processed")
    p.add_argument("--output-dir", default="./outputs/grpo")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=32)
    p.add_argument("--num-generations", type=int, default=16)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--report-to", default="wandb")
    p.add_argument("--logging-steps", type=int, default=1)
    a = p.parse_args()

    # Convert argparse namespace to a simple object for train_grpo
    class _Args:
        pass

    args = _Args()
    for k, v in vars(a).items():
        setattr(args, k.replace("-", "_"), v)

    train_grpo(args)


if __name__ == "__main__":
    main()
