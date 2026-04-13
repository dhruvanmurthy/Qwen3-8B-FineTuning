"""
Download GRPO (or SFT) adapter weights from Tinker and push to Hugging Face Hub.

Usage:
    python scripts/push_model_to_hub.py \
        --tinker-path "tinker://107560c7-cbd1-5830-8ca3-780ed0afc765:train:0/sampler_weights/final" \
        --repo-id dhruvanmurthy/Qwen3-8B-tool-use-grpo \
        [--stage grpo] \
        [--private]

Requires:
    TINKER_API_KEY  — in .env or environment
    HF_TOKEN        — in .env or environment (needs write access to repo)
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests
import tinker
from huggingface_hub import HfApi, create_repo, upload_folder


def _load_env() -> None:
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())


def _get_archive_url(tinker_path: str) -> str:
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    print(f"Fetching signed download URL for: {tinker_path}")
    resp = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()
    print(f"URL expires at: {resp.expires}")
    return resp.url


def _get_weights_info(tinker_path: str) -> dict:
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    info = rc.get_weights_info_by_tinker_path(tinker_path).result()
    return {
        "base_model": info.base_model,
        "is_lora": info.is_lora,
        "lora_rank": info.lora_rank,
        "train_unembed": info.train_unembed,
        "train_mlp": info.train_mlp,
        "train_attn": info.train_attn,
    }


def _download_and_extract(url: str, dest: Path) -> None:
    print(f"Downloading archive...")
    dest.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        archive_path = dest / "weights_archive"
        downloaded = 0
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
    print()

    # Detect and extract archive format
    extract_dir = dest / "extracted"
    extract_dir.mkdir(exist_ok=True)

    if tarfile.is_tarfile(archive_path):
        print("Extracting tar archive...")
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir)
    elif zipfile.is_zipfile(archive_path):
        print("Extracting zip archive...")
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    else:
        # Might be a raw safetensors file
        print("Archive is not tar/zip — treating as raw weights file.")
        shutil.copy(archive_path, extract_dir / "adapter_model.safetensors")

    archive_path.unlink()
    print(f"Extracted to: {extract_dir}")
    return extract_dir


def _write_model_card(dest: Path, repo_id: str, stage: str, info: dict) -> None:
    base_model = info.get("base_model", "Qwen/Qwen3-8B")
    lora_rank = info.get("lora_rank", "N/A")

    # Load eval metrics if available
    eval_json = Path("outputs/eval_comparison.json")
    eval_section = ""
    if eval_json.exists():
        try:
            results = json.loads(eval_json.read_text())
            if stage in results:
                m = results[stage]
                eval_section = f"""
## Evaluation Results

| Metric | Score |
|---|---|
| Tool Selection Accuracy | {m.get('tool_selection_accuracy', 0)*100:.1f}% |
| Argument Accuracy | {m.get('argument_accuracy', 0)*100:.1f}% |
| Schema Compliance | {m.get('schema_compliance', 0)*100:.1f}% |
| Multi-Step Success | {m.get('multi_step_success', 0)*100:.1f}% |
| Avg Latency | {m.get('avg_latency_ms', 0):.0f} ms |
"""
        except Exception:
            pass

    content = f"""---
license: mit
base_model: {base_model}
library_name: peft
tags:
  - tool-use
  - lora
  - qwen3
  - {stage}
---

# {repo_id}

LoRA adapter for **{base_model}** fine-tuned for tool-use via **{"GRPO (reinforcement learning)" if stage == "grpo" else "SFT (supervised fine-tuning)"}**.

## Model Details

- **Base model**: `{base_model}`
- **Training stage**: `{stage.upper()}`
- **LoRA rank**: {lora_rank}
- **Task**: Multi-tool selection and argument generation
- **Trained with**: [Tinker](https://tinker-console.thinkingmachines.ai) remote GPU training
{eval_section}
## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}")
```

## License

MIT
"""
    (dest / "README.md").write_text(content)


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Download Tinker weights and push to HF Hub")
    parser.add_argument(
        "--tinker-path",
        default="tinker://107560c7-cbd1-5830-8ca3-780ed0afc765:train:0/sampler_weights/final",
        help="Tinker checkpoint path (tinker://...)",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF Hub repo ID, e.g. dhruvanmurthy/Qwen3-8B-tool-use-grpo",
    )
    parser.add_argument("--stage", default="grpo", choices=["sft", "grpo"],
                        help="Training stage label (for model card)")
    parser.add_argument("--private", action="store_true", default=False)
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    if not os.getenv("TINKER_API_KEY"):
        print("Error: TINKER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    # 1. Get weights info
    print("Fetching weights metadata...")
    info = _get_weights_info(args.tinker_path)
    print(f"  base_model : {info['base_model']}")
    print(f"  is_lora    : {info['is_lora']}")
    print(f"  lora_rank  : {info['lora_rank']}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 2. Download + extract
        url = _get_archive_url(args.tinker_path)
        extract_dir = _download_and_extract(url, tmp_path)

        # 3. Write model card
        _write_model_card(extract_dir, args.repo_id, args.stage, info)

        # 4. Upload to HF Hub
        print(f"\nCreating/updating HF repo: {args.repo_id}")
        api = HfApi(token=hf_token)
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=hf_token,
        )
        print("Uploading files to Hub...")
        upload_folder(
            repo_id=args.repo_id,
            folder_path=str(extract_dir),
            repo_type="model",
            token=hf_token,
            commit_message=f"Upload {args.stage.upper()} adapter from Tinker ({args.tinker_path})",
        )

    print(f"\nDone! View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
