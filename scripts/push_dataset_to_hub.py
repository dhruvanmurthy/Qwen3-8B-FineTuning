"""
Push synthetic tool-use dataset to Hugging Face Hub.

Uploads all JSONL files from data/raw/synthetic/ as a dataset repository.

Usage:
    python scripts/push_dataset_to_hub.py \
        --repo-id dhruvanmurthy/qwen3-8b-synthetic-tool-use \
        [--data-dir data/raw/synthetic] \
        [--private]

Requires HF_TOKEN environment variable (or set via huggingface-cli login).
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Push synthetic dataset to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF dataset repo ID, e.g. 'dhruvanmurthy/qwen3-8b-synthetic-tool-use'",
    )
    parser.add_argument(
        "--data-dir",
        default="./data/raw/synthetic",
        help="Directory containing synthetic JSONL files (default: ./data/raw/synthetic)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the dataset repository private",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload synthetic tool-use dataset",
        help="Commit message for the upload",
    )
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is not set.", file=sys.stderr)
        print("Set it in your .env file or run: huggingface-cli login", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        print("Run 'python scripts/generate_synthetic.py' first.", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(data_dir.glob("*.jsonl")) + sorted(data_dir.glob("*.json"))
    if not jsonl_files:
        print(f"Error: No JSONL/JSON files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jsonl_files)} file(s) to upload:")
    for f in jsonl_files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.1f} KB)")

    from huggingface_hub import HfApi, create_repo, upload_file

    api = HfApi(token=hf_token)

    # Create dataset repo if it doesn't exist
    print(f"\nCreating/verifying dataset repo: {args.repo_id} ...")
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=hf_token,
    )
    print(f"  Repo ready: https://huggingface.co/datasets/{args.repo_id}")

    # Upload each file
    for fpath in jsonl_files:
        print(f"\nUploading {fpath.name} ...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=f"data/{fpath.name}",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=hf_token,
            commit_message=args.commit_message,
        )
        print(f"  ✓ {fpath.name}")

    # Create a minimal README if the repo is new
    readme_content = f"""---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- tool-use
- function-calling
- synthetic
---

# Synthetic Tool-Use Dataset

Synthetic training data for fine-tuning Qwen3-8B on multi-step tool use and function calling.

Generated via [`scripts/generate_synthetic.py`](https://github.com/dhruvanmurthy/Qwen3-8B-FineTuning).

## Files

| File | Content |
|------|---------|
| `data/synthetic_single.jsonl` | Single-step tool calls (~70% of examples) |
| `data/synthetic_multistep.jsonl` | Multi-step tool chains (~30% of examples) |

## Schema

```json
{{
  "instruction": "What is the weather in Tokyo?",
  "tools": [{{"name": "get_weather", "description": "...", "parameters": {{...}}}}],
  "tool_calls": [{{"name": "get_weather", "arguments": {{"city": "Tokyo", "units": "C"}}}}],
  "category": "weather",
  "num_steps": 1,
  "text": "[weather] USER: What is the weather in Tokyo?\\nASSISTANT: <tool_call>\\n{{...}}\\n</tool_call>"
}}
```

## Generation

```bash
python scripts/generate_synthetic.py --num-samples 15000 --seed 42
```
"""

    print("\nUploading README.md ...")
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=hf_token,
        commit_message="Add dataset README",
    )
    print("  ✓ README.md")

    print(f"\n✅ Dataset pushed to: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
