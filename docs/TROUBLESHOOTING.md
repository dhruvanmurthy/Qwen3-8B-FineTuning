# Troubleshooting

This guide covers the current Tinker-based implementation.

## `TINKER_API_KEY` Missing

Symptoms:

- baseline evaluation fails immediately
- training scripts exit before connecting to Tinker

Fix:

```bash
export TINKER_API_KEY=your_key
```

Or add it to `.env`.

## Synthetic Data Missing

Symptoms:

- SFT training reports no training data
- data preparation or smoke test cannot find `data/raw/synthetic`

Fix:

```bash
python scripts/generate_synthetic.py --output-dir data/raw/synthetic
bash scripts/prepare_datasets.sh
```

## `checkpoints.jsonl` Missing

Symptoms:

- evaluation cannot resolve sampler paths
- GRPO cannot resume from SFT output

Fix:

- confirm the stage actually completed
- check `outputs/sft/` or `outputs/grpo/`
- pass `--sft-sampler-path` or `--grpo-sampler-path` explicitly if needed

## GRPO Rank Mismatch

Symptoms:

- `scripts/run_pipeline.sh grpo` fails fast before training begins

Cause:

- the pipeline requires GRPO to reuse the SFT checkpoint with a matching LoRA
  rank

Fix:

- use `scripts/run_pipeline.sh` defaults, or
- pass the same `--lora-rank` value that was used in SFT

## GRPO Rewards Stay Flat

Symptoms:

- `train/reward_mean` stays near zero
- most generations fail schema checks or tool extraction

Checks:

1. Evaluate the SFT checkpoint first
2. Confirm the SFT stage can already emit valid tool calls
3. Reduce `--group-size` or `--max-completion-length` if generations are noisy
4. Verify your prepared data is current and not empty

## W&B Not Logging

Behavior:

- the scripts keep running, but W&B logging is disabled

Fix:

```bash
export WANDB_API_KEY=your_key
```

Without `WANDB_API_KEY`, the current code intentionally initializes W&B in
disabled mode.

## Hub Upload Is Skipped

Behavior:

- training finishes, but no adapter is uploaded

Fix:

Set both:

- `HF_TOKEN`
- `HF_REPO_ID`

If only one is set, the scripts log a warning and skip upload.

## Smoke Test Expectations

The smoke test intentionally evaluates only a small benchmark subset by default:

- `schema_compliance`
- `multi_step`

That is enough to validate end-to-end wiring without running the full evaluation
suite.

## Bash on Windows

The repository includes Bash entrypoints. On Windows, run those scripts through
WSL or another Bash environment instead of PowerShell.
