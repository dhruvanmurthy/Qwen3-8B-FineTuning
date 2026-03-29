# Qwen3-8B Fine-tuning for Tool Use (Function Calling)

A cost-efficient, reproducible MLOps project for fine-tuning Qwen3-8B on multi-step tool use, API calling, and function chaining.

## Project Goals

- **Model**: Qwen3-8B (8B parameters, fast inference)
- **Task**: Multi-step tool use & function calling
- **Pipeline**: 3-stage — Baseline eval → SFT (QLoRA) → GRPO (RL with binary rewards)
- **Budget**: ~$150 (Azure spot instances)
- **Reproducibility**: Full tracking via W&B, versioned datasets, seed control

## Key Metrics

| Metric | Baseline | SFT Target | GRPO Target |
|--------|----------|------------|-------------|
| Tool Selection Accuracy | ~65% | >85% | **>90%** |
| Argument Correctness | ~50% | >75% | **>85%** |
| Multi-step Success Rate | ~40% | >70% | **>80%** |
| Schema Compliance | ~40% | >80% | **>90%** |

## Project Structure

```
Qwen3-8B-FineTuning/
├── README.md                          # This file
├── GETTING_STARTED.md                 # Step-by-step local guide
├── EXECUTION_PLAN.md                  # Phase-by-phase execution plan
├── MODEL_CARD.md                      # Model documentation
├── DATASET_CARD.md                    # Dataset documentation
├── CONTRIBUTING.md                    # Contribution guidelines
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
│
├── docs/
│   ├── AZURE_SETUP.md                # Step-by-step Azure guide
│   ├── TRAINING_PLAN.md              # 3-stage training strategy
│   ├── DATASET_STRATEGY.md           # Data pipeline & aggregation
│   ├── EVALUATION.md                 # 3-stage evaluation methodology
│   ├── BUDGET_OPTIMIZATION.md        # Cost reduction strategies
│   └── TROUBLESHOOTING.md            # Common issues & fixes
│
├── configs/
│   ├── training.yaml                 # Stage 1: SFT hyperparameters
│   ├── grpo_config.yaml              # Stage 2: GRPO recipe
│   ├── azure_config.yaml             # Azure ML Job config
│   ├── qora_config.yaml              # QLoRA parameters
│   └── dataset_config.yaml           # Dataset pipeline config
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Dataset loading, preprocessing, GRPO prompt prep
│   ├── train.py                      # Stage 1: SFT training (HF Trainer)
│   ├── train_grpo.py                 # Stage 2: GRPO training (TRL GRPOTrainer)
│   ├── rewards.py                    # Binary verifiable reward functions
│   ├── environments.py               # Atropos-pattern coordinator
│   └── evaluate.py                   # 3-stage evaluation (baseline/SFT/GRPO)
│
├── scripts/
│   ├── run_pipeline.sh               # 3-stage orchestrator (main entry point)
│   ├── setup_azure.sh                # Azure resource provisioning
│   ├── prepare_datasets.sh           # Download & process datasets
│   ├── run_local_training.sh         # Local SFT training
│   ├── run_azure_training.sh         # Azure training submission
│   ├── evaluate_model.sh             # Run evaluation
│   └── cleanup_azure.sh              # Delete Azure resources
│
├── data/
│   ├── raw/                          # Downloaded raw datasets
│   └── processed/                    # Processed, formatted datasets
│
└── .github/
    └── workflows/
        ├── test.yml                  # CI/CD testing
        └── release.yml               # Model release to HF Hub
```

## Quick Start

### Local (see [GETTING_STARTED.md](GETTING_STARTED.md) for details)

```bash
# 1. Setup
pip install -r requirements.txt
cp .env.example .env   # edit with your HF_TOKEN, WANDB_API_KEY

# 2. Prepare data
bash scripts/prepare_datasets.sh

# 3. Run entire pipeline (baseline → SFT → GRPO → compare)
bash scripts/run_pipeline.sh all

# Or run stages individually:
bash scripts/run_pipeline.sh baseline   # Evaluate base model
bash scripts/run_pipeline.sh sft        # Train SFT
bash scripts/run_pipeline.sh grpo       # Train GRPO (requires SFT output)
bash scripts/run_pipeline.sh compare    # Side-by-side comparison
```

### Azure (see [docs/AZURE_SETUP.md](docs/AZURE_SETUP.md) for details)

```bash
# 1. Provision resources
bash scripts/setup_azure.sh

# 2. Submit training
bash scripts/run_azure_training.sh

# 3. Cleanup when done
bash scripts/cleanup_azure.sh
```

## Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** — Full local setup and run guide
- **[docs/AZURE_SETUP.md](docs/AZURE_SETUP.md)** — Azure ML environment and job submission
- **[docs/TRAINING_PLAN.md](docs/TRAINING_PLAN.md)** — 3-stage training strategy and hyperparameters
- **[docs/DATASET_STRATEGY.md](docs/DATASET_STRATEGY.md)** — Dataset sourcing, aggregation, GRPO prompt prep
- **[docs/EVALUATION.md](docs/EVALUATION.md)** — 3-stage evaluation metrics and comparison
- **[docs/BUDGET_OPTIMIZATION.md](docs/BUDGET_OPTIMIZATION.md)** — Cost reduction and spot instance strategies
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** — Common errors and solutions (incl. GRPO)
- **[MODEL_CARD.md](MODEL_CARD.md)** — Model capabilities, limitations, license
- **[DATASET_CARD.md](DATASET_CARD.md)** — Dataset composition, sources, licenses

## Key Technologies

| Component | Tool | Purpose |
|-----------|------|---------|
| **Model** | Qwen3-8B (HF Hub) | Base LLM |
| **SFT Training** | HuggingFace Transformers + QLoRA | Stage 1: supervised fine-tuning |
| **GRPO Training** | TRL (GRPOTrainer) | Stage 2: RL with binary rewards |
| **Rewards** | Custom binary verifiers | Tool-use correctness signals |
| **Coordinator** | Atropos pattern (in-process) | Environment ↔ trainer bridge |
| **Compute** | Azure ML Compute Clusters | GPU training (A100/A10 spot) |
| **Tracking** | Weights & Biases | Experiment logging |
| **Versioning** | Git + DVC | Code & data version control |
| **Models** | Hugging Face Hub | Model distribution |

## Budget Breakdown (~$64 estimated)

| Item | Cost | Notes |
|------|------|-------|
| Baseline evaluation | $1.50 | ~15 min A10 spot |
| SFT training (3 epochs) | $27 | ~6h A100 spot |
| SFT evaluation | $1.50 | ~15 min A10 spot |
| GRPO training (50 steps) | $6 | ~1h A100 spot |
| GRPO evaluation | $1.50 | ~15 min A10 spot |
| Comparison report | $4.50 | ~45 min A10 spot |
| Storage & misc | $2 | Blob storage |
| **Buffer** | $20 | Preemption retries / tuning |
| **Total** | **~$64** | Well within $150 budget |

## Security & Reproducibility

- **Secrets**: `.env` (git-ignored), Azure Key Vault for production
- **Seeds**: Fixed global seed (42) for all stages
- **Deps**: Pinned versions in `requirements.txt`
- **Data**: Versioned datasets on HF Hub, DVC for local tracking
- **Tracking**: W&B artifacts with reproducible runs

## Experiment Tracking

All training runs logged to **Weights & Biases**:
```
wandb.ai/dhruvanmurthy/qwen3-8b-tool-use
```

Tracks learning curves, loss metrics, reward signals (GRPO), evaluation tables, and GPU utilization across all three stages.

## Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-improvement

# 2. Make changes, test locally with a quick smoke test
python src/train.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --data_dir data/processed \
  --output_dir outputs/test \
  --max_steps 20

# 3. Commit and push
git add .
git commit -m "feat: add new evaluation metric"
git push origin feature/my-improvement

# 4. Create PR, merge after review

# 5. Release model to HF Hub (automatic via GitHub Actions)
git tag v1.0.0
git push origin v1.0.0
```

## Model Distribution

Fine-tuned adapters available at:
- **SFT adapter**: `dhruvanmurthy/qwen3-8b-tool-use-sft-lora`
- **GRPO adapter**: `dhruvanmurthy/qwen3-8b-tool-use-grpo-lora`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
# Load SFT adapter
sft_model = PeftModel.from_pretrained(base, "dhruvanmurthy/qwen3-8b-tool-use-sft-lora")
# Or load GRPO (must merge SFT first)
merged = sft_model.merge_and_unload()
grpo_model = PeftModel.from_pretrained(merged, "dhruvanmurthy/qwen3-8b-tool-use-grpo-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for solutions to:
- Azure authentication and quota issues
- GPU memory overflow (SFT and GRPO)
- GRPO rewards stuck at zero
- TRL version compatibility
- Training instability

## License

- **Code**: MIT License (see LICENSE file)
- **Model**: Same as Qwen3-8B base model (Apache 2.0)
- **Datasets**: Mixed (see DATASET_CARD.md for attribution)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

**Author**: Dhruvan Murthy  
**GitHub**: [@dhruvanmurthy](https://github.com/dhruvanmurthy)  
**HuggingFace**: [dhruvanmurthy](https://huggingface.co/dhruvanmurthy)
