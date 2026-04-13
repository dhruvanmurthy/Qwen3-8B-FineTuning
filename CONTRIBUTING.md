# Contributing

Thanks for helping improve the repository.

## Ground Rules

- Treat the current Tinker-based pipeline as the source of truth.
- Keep documentation aligned with the live scripts and code paths.
- Prefer fixing inconsistencies rather than preserving parallel legacy flows.

## Setup

```bash
pip install -r requirements.txt
```

If you work on the Bash scripts from Windows, use WSL.

## Useful Checks

Format and lint:

```bash
black src/ scripts/
isort src/ scripts/
flake8 src/ scripts/
mypy src/ --ignore-missing-imports
```

Synthetic-data sanity check:

```bash
python scripts/generate_synthetic.py --num-samples 200 --output-dir data/raw/synthetic
```

Pipeline wiring check without remote execution:

```bash
LOCAL_VALIDATE=true bash scripts/run_pipeline.sh all
```

Smoke test with real Tinker calls:

```bash
bash scripts/run_smoke_test.sh
```

## Testing

The CI workflow supports a repository with or without a `tests/` directory.

- If you add tests, place them under `tests/`.
- Keep unit tests independent from Tinker unless they are clearly marked as
  integration tests.

Example:

```bash
pytest tests/ -v
```

## Documentation Changes

When changing docs:

- update the command examples to match the real CLI
- avoid introducing references to files or workflows that do not exist
- prefer approximate dataset counts unless the value is guaranteed by code
- keep top-level docs and `docs/` pages consistent with each other

## Pull Requests

A good PR should explain:

- what changed
- why it changed
- how it was validated

If the change touches training, evaluation, or documentation, include the exact
commands you used for validation.
