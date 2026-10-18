"""Upload a GGUF only after a merged checkpoint exists in its own repo."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_coverage():
    spec = importlib.util.spec_from_file_location("joker_coverage", ROOT / "hub" / "coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a Joker GGUF in a separate repo.")
    parser.add_argument("--merged-repo", required=True, help="Repo that holds the 16-bit merge.")
    parser.add_argument("--gguf-repo", required=True, help="Separate repo for the GGUF.")
    parser.add_argument("--gguf", help="Local .gguf file. This machine does not build one.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coverage = load_coverage()
    if args.gguf_repo == args.merged_repo:
        sys.exit("The GGUF repo has to be different from the merged repo.")
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(args.merged_repo)
    if not coverage.merged_weights_present(files):
        sys.exit(
            f"No merged checkpoint in {args.merged_repo}. "
            "A GGUF waits until the 16-bit merge exists in its own repo."
        )
    gguf_path = Path(args.gguf) if args.gguf else None
    if gguf_path is None or not gguf_path.is_file():
        sys.exit("Merged weights are there. Pass a local .gguf to upload. This machine does not convert one.")
    api.create_repo(args.gguf_repo, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=gguf_path.name,
        repo_id=args.gguf_repo,
        repo_type="model",
        commit_message="add the gguf",
    )
    api.upload_file(
        path_or_fileobj=coverage.gguf_card(args.merged_repo).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.gguf_repo,
        repo_type="model",
        commit_message="add the quantized model card",
    )
    print(f"gguf: https://huggingface.co/{args.gguf_repo}")


if __name__ == "__main__":
    main()
