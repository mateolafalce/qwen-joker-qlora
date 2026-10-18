"""Upload the model card and the dataset card to the Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the Joker Hub cards.")
    parser.add_argument(
        "--model-repo",
        default=os.environ.get("HF_OUTPUT_REPO", "lafalce/qwen2.5-7b-joker"),
    )
    parser.add_argument(
        "--dataset-repo",
        default=os.environ.get("HF_DATASET_REPO", "lafalce/qwen-joker-sft"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(ROOT / "hub" / "model_card.md"),
        path_in_repo="README.md",
        repo_id=args.model_repo,
        repo_type="model",
        commit_message="replace the unsloth stub with the joker model card",
    )
    print(f"model card: https://huggingface.co/{args.model_repo}")
    api.upload_file(
        path_or_fileobj=str(ROOT / "notebook.ipynb"),
        path_in_repo="notebook.ipynb",
        repo_id=args.model_repo,
        repo_type="model",
        commit_message="add the colab and kaggle notebook",
    )
    print(f"notebook: https://huggingface.co/{args.model_repo}/blob/main/notebook.ipynb")
    for name in ("eval_base.json", "eval_adapter.json"):
        path = ROOT / "hub" / name
        if not path.is_file():
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=args.model_repo,
            repo_type="model",
            commit_message=f"add {name}",
        )
        print(f"report: https://huggingface.co/{args.model_repo}/blob/main/{name}")
    api.upload_file(
        path_or_fileobj=str(ROOT / "hub" / "dataset_card.md"),
        path_in_repo="README.md",
        repo_id=args.dataset_repo,
        repo_type="dataset",
        commit_message="add the joker dataset card",
    )
    print(f"dataset card: https://huggingface.co/datasets/{args.dataset_repo}")


if __name__ == "__main__":
    main()
