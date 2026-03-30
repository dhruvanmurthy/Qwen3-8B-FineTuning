"""
Main training script for Qwen3-8B QLoRA fine-tuning.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
import yaml
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments
)

# Local imports
from data_loader import ToolUseDataLoader

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model loading."""
    model_name_or_path: str = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "Path to model or model ID"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype: bfloat16, float16"}
    )
    load_in_4bit: bool = field(
        default=True,
        metadata={"help": "Load in 4-bit quantization"}
    )


@dataclass
class DataArguments:
    """Arguments for data loading."""
    data_dir: str = field(
        default="./data/processed",
        metadata={"help": "Directory with processed datasets"}
    )
    dataset_config: str = field(
        default="configs/dataset_config.yaml",
        metadata={"help": "Path to dataset config"}
    )
    num_samples: int = field(
        default=40000,
        metadata={"help": "Number of samples to use"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated target modules"}
    )


def setup_logging(output_dir: str):
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Log to file
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, "training.log")
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
    )
    logger.addHandler(file_handler)


def load_model_and_tokenizer(model_args: ModelArguments):
    """Load base model and tokenizer."""
    
    # Setup quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_args.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="left",
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded: {model.config.model_type}")
    logger.info(f"Model size: {model.num_parameters() / 1e9:.2f}B parameters")
    
    return model, tokenizer


def setup_lora(model, lora_args: LoraArguments):
    """Setup LoRA adaptation."""
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    target_modules = lora_args.target_modules.split(",")
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def load_and_preprocess_data(
    data_args: DataArguments,
    tokenizer,
) -> DatasetDict:
    """Load and preprocess datasets."""
    
    logger.info("Loading datasets...")
    
    data_loader = ToolUseDataLoader(data_args.dataset_config)
    datasets = data_loader.prepare_datasets(
        tokenizer,
        max_length=data_args.max_seq_length
    )
    
    logger.info(f"Train: {len(datasets['train'])} examples")
    logger.info(f"Validation: {len(datasets['validation'])} examples")
    logger.info(f"Test: {len(datasets['test'])} examples")
    
    return datasets


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    training_args: TrainingArguments,
):
    """Main training function."""
    
    setup_logging(training_args.output_dir)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # Setup LoRA
    model = setup_lora(model, lora_args)
    
    # Load data
    datasets = load_and_preprocess_data(data_args, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    # Callbacks
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    # Resume from checkpoint if available
    checkpoint = None
    if training_args.resume_from_checkpoint:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        checkpoints = [
            os.path.join(training_args.output_dir, d)
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getmtime)
            logger.info(f"Resuming from checkpoint: {checkpoint}")
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save results
    metrics = train_result.metrics
    metrics["train_samples"] = len(datasets["train"])
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics, combined=False)
    
    # Save model
    logger.info(f"Saving model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Evaluate
    logger.info("Running evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics, combined=False)
    
    return train_result


def main():
    """Main entry point."""
    
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments, TrainingArguments)
    )
    
    (model_args, data_args, lora_args, training_args) = parser.parse_args_into_dataclasses()
    
    # Train
    train(model_args, data_args, lora_args, training_args)


if __name__ == "__main__":
    main()
