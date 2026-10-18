# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth==2026.10.6",
#   "unsloth_zoo==2026.10.6",
#   "datasets==4.8.5",
#   "trl==0.22.2",
#   "transformers==4.57.3",
#   "peft==0.18.1",
#   "accelerate==1.11.0",
#   "bitsandbytes==0.48.2",
#   "huggingface_hub[hf_transfer]==0.36.2",
# ]
# ///
"""QLoRA of a current Qwen instruct with Unsloth, so it answers as the Joker.

SFT uses a validation split and stops when eval loss stalls. DPO then trains a
second adapter, starting from the SFT weights, and uses the frozen SFT adapter
as the reference. It prefers the in-character reply over an assistant voice, a
wrong fact, or emoji spam. The adapter is pushed to the Hub before the process
exits 0. A failed push fails the job.

Meant for Hugging Face Jobs on an A10G large (24 GB). The dataset already
carries the system prompt on every row. Loss falls only on the assistant reply.

Local example, with a GPU:

  uv run train.py \\
    --dataset data/joker.jsonl \\
    --val-dataset data/joker_val.jsonl \\
    --preferences data/preferences.jsonl \\
    --eval-items data/eval_items.jsonl \\
    --output-repo YOUR_USER/qwen3-8b-joker \\
    --sft-repo YOUR_USER/qwen3-8b-joker-sft \\
    --num-epochs 4

On Jobs, sibling files are not uploaded. Publish the jsonl first and pass the repo.
scripts/launch_job.sh does both.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("joker")

ASSISTANT_MARKER = "<|im_start|>assistant\n"
QWEN_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
CURRENT_BASE = "unsloth/Qwen3-8B-bnb-4bit"
LEGACY_BASE = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
QWEN25_7B = {"hidden": 3584, "intermediate": 18944, "heads": 28, "kv_heads": 4}
QWEN3_8B = {"hidden": 4096, "intermediate": 12288, "heads": 32, "kv_heads": 8}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_records(records: list) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_text(payload)


def module_lora_shapes(spec: dict, rank: int) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    hidden = spec["hidden"]
    intermediate = spec["intermediate"]
    head_dim = hidden // spec["heads"]
    kv_out = spec["kv_heads"] * head_dim
    return {
        "q_proj": ((rank, hidden), (hidden, rank)),
        "k_proj": ((rank, hidden), (kv_out, rank)),
        "v_proj": ((rank, hidden), (kv_out, rank)),
        "o_proj": ((rank, hidden), (hidden, rank)),
        "gate_proj": ((rank, hidden), (intermediate, rank)),
        "up_proj": ((rank, hidden), (intermediate, rank)),
        "down_proj": ((rank, intermediate), (hidden, rank)),
    }


def transfer_verdict(source: dict, target: dict) -> dict:
    names = sorted(set(source) | set(target))
    mismatched = [name for name in names if source.get(name) != target.get(name)]
    return {"transfers": not mismatched, "mismatched_modules": mismatched}


def qwen25_to_qwen3_transfer(rank: int = 16) -> dict:
    verdict = transfer_verdict(module_lora_shapes(QWEN25_7B, rank), module_lora_shapes(QWEN3_8B, rank))
    verdict["source"] = "Qwen2.5-7B-Instruct"
    verdict["target"] = "Qwen3-8B"
    verdict["rank"] = rank
    verdict["reason"] = (
        "LoRA shapes differ, so a Qwen2.5 adapter cannot load on Qwen3-8B. "
        "The same data is retrained. That is the transfer result."
    )
    return verdict


def build_manifest(
    *,
    git_sha: str,
    base_model: str,
    train_sha256: str,
    val_sha256: str,
    preferences_sha256: str,
    loaded_train_sha256: str,
    loaded_val_sha256: str,
    loaded_preferences_sha256: str,
    hyperparameters: dict,
    sft: dict,
    dpo: dict | None,
    run_label: str = "primary",
    evaluation: dict | None = None,
    transfer: dict | None = None,
) -> dict:
    return {
        "git_sha": git_sha,
        "base_model": base_model,
        "run_label": run_label,
        "train_sha256": train_sha256,
        "val_sha256": val_sha256,
        "preferences_sha256": preferences_sha256,
        "loaded_train_sha256": loaded_train_sha256,
        "loaded_val_sha256": loaded_val_sha256,
        "loaded_preferences_sha256": loaded_preferences_sha256,
        "hyperparameters": hyperparameters,
        "sft": sft,
        "dpo": dpo,
        "evaluation": evaluation,
        "transfer": transfer,
    }


def call_with_retries(action, attempts: int = 3, pause_seconds: float = 0):
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as exc:
            last_error = exc
            logger.warning("attempt %s/%s failed: %s", attempt, attempts, exc)
            if attempt < attempts and pause_seconds:
                time.sleep(pause_seconds)
    raise last_error


def check_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        logger.error("No CUDA. This script runs on a GPU.")
        sys.exit(1)
    logger.info("GPU: %s", torch.cuda.get_device_name(0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA Joker on Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--base-model",
        default=CURRENT_BASE,
        help="4-bit instruct checkpoint. The default is Qwen3-8B. Qwen2.5 stays available as the legacy base.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Local path to the train jsonl, or a dataset repo on the Hub.",
    )
    parser.add_argument("--data-files", default=None, help="Train file inside the repo.")
    parser.add_argument("--val-dataset", required=True, help="Local val jsonl, or a dataset repo.")
    parser.add_argument("--val-data-files", default=None, help="Val file inside the repo.")
    parser.add_argument("--preferences", required=True, help="Local preferences jsonl, or a dataset repo.")
    parser.add_argument("--preferences-data-files", default=None)
    parser.add_argument("--output-repo", required=True, help="Adapter repo on the Hub.")
    parser.add_argument("--num-epochs", type=float, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save-local", default="joker-output")
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--skip-dpo", action="store_true")
    parser.add_argument("--dpo-epochs", type=float, default=3)
    parser.add_argument("--dpo-learning-rate", type=float, default=5e-6)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--eval-items", default=None, help="Local eval jsonl, or a dataset repo.")
    parser.add_argument("--eval-data-files", default=None)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-max-new-tokens", type=int, default=280)
    parser.add_argument("--judge-seed", type=int, default=3407)
    parser.add_argument("--sft-repo", default="", help="Repo that keeps the SFT adapter when DPO continues.")
    parser.add_argument("--run-label", default="primary")
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--train-sha", default="")
    parser.add_argument("--val-sha", default="")
    parser.add_argument("--preferences-sha", default="")
    parser.add_argument("--push-attempts", type=int, default=3)
    parser.add_argument(
        "--merge-model",
        action="store_true",
        help="Also upload the merged model in 16 bits. About 15 GB.",
    )
    parser.add_argument(
        "--merge-repo",
        default=None,
        help="Separate repo for the merged weights. Required with --merge-model.",
    )
    parser.add_argument("--private", action="store_true", help="Create the model repo as private.")
    return parser.parse_args()


def fetch_readme(repo_id: str) -> str | None:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="README.md",
            repo_type="model",
            force_download=True,
        )
    except Exception:
        logger.warning("No model card to keep on %s", repo_id)
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def restore_readme(repo_id: str, text: str | None) -> None:
    import io

    from huggingface_hub import HfApi

    if not text:
        return
    HfApi().upload_file(
        path_or_fileobj=io.BytesIO(text.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="restore model card",
    )
    logger.info("Restored the model card on %s", repo_id)


def load_rows(dataset: str, data_files: str | None):
    from datasets import load_dataset

    if os.path.isfile(dataset):
        return load_dataset("json", data_files=dataset, split="train")
    if data_files:
        return load_dataset(dataset, data_files=data_files, split="train")
    return load_dataset(dataset, split="train")


def require_messages(dataset, label: str):
    if "messages" not in dataset.column_names:
        sys.exit(f"The {label} dataset has no 'messages' column. Columns: {dataset.column_names}")


def records_from_messages(dataset) -> list[dict]:
    return [{"messages": [dict(message) for message in row["messages"]]} for row in dataset]


def records_from_preferences(dataset) -> list[dict]:
    keys = ["prompt", "chosen", "rejected", "tag", "split"]
    return [{key: row[key] for key in keys if key in row} for row in dataset]


def to_text_batch(tokenizer, bos: str):
    def to_text(batch):
        texts = []
        for conversation in batch["messages"]:
            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            if bos and text.startswith(bos):
                text = text[len(bos) :]
            texts.append(text)
        return {"text": texts}

    return to_text


def resolve_merge_repo(output_repo: str, merge_repo: str | None) -> str:
    if not merge_repo or merge_repo == output_repo:
        raise ValueError("merged weights need a repo that is not the adapter repo")
    return merge_repo


def hyperparameters_of(args: argparse.Namespace) -> dict:
    return {
        "num_epochs": args.num_epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "seed": args.seed,
        "early_stopping_patience": args.early_stopping_patience,
        "dpo_epochs": args.dpo_epochs,
        "dpo_learning_rate": args.dpo_learning_rate,
        "dpo_beta": args.dpo_beta,
        "skip_dpo": args.skip_dpo,
    }


def push_adapter(model, tokenizer, repo: str, private: bool, manifest: dict | None, manifest_name: str) -> None:
    model.push_to_hub(repo, tokenizer=tokenizer, private=private)
    if manifest is None:
        return
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
        path_in_repo=manifest_name,
        repo_id=repo,
        repo_type="model",
    )


def publish(
    model,
    tokenizer,
    repo: str,
    private: bool,
    manifest: dict | None,
    manifest_name: str,
    attempts: int,
    fallback_card: str | None = None,
) -> None:
    card = fetch_readme(repo) or fallback_card
    call_with_retries(
        lambda: push_adapter(model, tokenizer, repo, private, manifest, manifest_name),
        attempts=attempts,
        pause_seconds=15,
    )
    try:
        restore_readme(repo, card)
    except Exception:
        logger.exception("The adapter is uploaded. The model card was not restored.")


class KeepBestAdapter:
    def __init__(self):
        from transformers import TrainerCallback

        keeper = self

        class _Callback(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                loss = (metrics or {}).get("eval_loss")
                model = kwargs.get("model")
                if loss is None or model is None:
                    return
                if keeper.best_loss is not None and loss >= keeper.best_loss:
                    return
                keeper.best_loss = loss
                keeper.best = {
                    name: param.detach().cpu().clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                logger.info("new best eval loss %.4f at step %s", loss, state.global_step)

        self.best_loss = None
        self.best = None
        self.callback = _Callback()

    def restore(self, model) -> None:
        if not self.best:
            logger.warning("no eval checkpoint to restore")
            return
        named = dict(model.named_parameters())
        for name, value in self.best.items():
            named[name].copy_(value.to(named[name].device))
        logger.info("restored adapter at eval loss %.4f", self.best_loss)


class AdapterPublishCallback:
    def __init__(self, repo: str, tokenizer, private: bool, attempts: int):
        from transformers import TrainerCallback

        outer = self

        class _Callback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                model = kwargs.get("model")
                if model is None:
                    return
                try:
                    publish(model, outer.tokenizer, outer.repo, outer.private, None, "", outer.attempts)
                    logger.info("published checkpoint at step %s", state.global_step)
                except Exception:
                    logger.exception("checkpoint push failed; training continues")

        self.repo = repo
        self.tokenizer = tokenizer
        self.private = private
        self.attempts = attempts
        self.callback = _Callback()


def preference_views(dataset):
    train = dataset.filter(lambda row: row["split"] == "train")
    val = dataset.filter(lambda row: row["split"] == "val")
    if len(train) == 0 or len(val) == 0:
        sys.exit(f"preferences need a train and a val split, got {len(train)} and {len(val)}")
    drop = [name for name in train.column_names if name not in {"prompt", "chosen", "rejected"}]
    return train.remove_columns(drop), val.remove_columns(drop)


def run_sft(model, tokenizer, train, val, args):
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    effective = args.batch_size * args.gradient_accumulation
    steps_per_epoch = max(1, len(train) // effective)
    total_steps = args.max_steps or int(steps_per_epoch * args.num_epochs)
    warmup = max(1, total_steps // 20)
    keeper = KeepBestAdapter()
    publish_repo = getattr(args, "sft_publish_repo", None) or args.output_repo
    publisher = AdapterPublishCallback(publish_repo, tokenizer, args.private, args.push_attempts)
    config = SFTConfig(
        output_dir=args.save_local,
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=warmup,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=args.learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        max_length=args.max_seq_length,
        bf16=True,
        fp16=False,
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        save_total_limit=2,
        push_to_hub=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train,
        eval_dataset=val,
        args=config,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            keeper.callback,
            publisher.callback,
        ],
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part=ASSISTANT_MARKER,
    )
    started = time.time()
    result = trainer.train()
    keeper.restore(trainer.model)
    minutes = (time.time() - started) / 60
    metrics = dict(result.metrics)
    metrics["best_eval_loss"] = keeper.best_loss
    metrics["minutes"] = round(minutes, 3)
    print(f"sft done in {minutes:.1f} min. train loss: {metrics.get('train_loss')} best eval: {keeper.best_loss}")
    return trainer.model, metrics


def _lora_pairs(model):
    pairs = []
    for module in model.modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        pairs.append((lora_a, lora_b))
    return pairs


def copy_adapter(model, source: str, dest: str) -> None:
    copied = 0
    for lora_a, lora_b in _lora_pairs(model):
        if source not in lora_a or dest not in lora_a:
            continue
        lora_a[dest].weight.data.copy_(lora_a[source].weight.data)
        lora_b[dest].weight.data.copy_(lora_b[source].weight.data)
        copied += 1
    if copied == 0:
        raise RuntimeError(f"no LoRA weights copied from {source} to {dest}")


def attach_reference_adapter(model) -> None:
    model.add_adapter("dpo", model.peft_config["default"])
    copy_adapter(model, "default", "dpo")
    model.set_adapter("dpo")
    for name, param in model.named_parameters():
        param.requires_grad = ".dpo." in name


def collapse_dpo_into_default(model) -> None:
    copy_adapter(model, "dpo", "default")
    model.delete_adapter("dpo")
    model.set_adapter("default")


def run_dpo(model, tokenizer, train, val, args):
    from transformers import EarlyStoppingCallback
    from trl import DPOConfig, DPOTrainer

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    keeper = KeepBestAdapter()
    config = DPOConfig(
        output_dir=f"{args.save_local}-dpo",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.dpo_learning_rate,
        num_train_epochs=args.dpo_epochs,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="linear",
        warmup_steps=1,
        seed=args.seed,
        bf16=True,
        fp16=False,
        report_to="none",
        beta=args.dpo_beta,
        max_length=args.max_seq_length,
        max_prompt_length=min(512, args.max_seq_length // 2),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        save_total_limit=2,
        gradient_checkpointing=True,
        model_adapter_name="dpo",
        ref_adapter_name="default",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            keeper.callback,
        ],
    )
    started = time.time()
    result = trainer.train()
    keeper.restore(trainer.model)
    minutes = (time.time() - started) / 60
    metrics = dict(result.metrics)
    metrics["best_eval_loss"] = keeper.best_loss
    metrics["minutes"] = round(minutes, 3)
    print(f"dpo done in {minutes:.1f} min. train loss: {metrics.get('train_loss')} best eval: {keeper.best_loss}")
    return trainer.model, metrics


def load_scorer(dataset: str):
    local = Path(__file__).resolve().parent / "eval.py"
    if local.is_file():
        path = local
    elif os.path.isfile(dataset):
        sys.exit("eval.py is not next to train.py, and the dataset is a local file.")
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo_id=dataset, filename="eval.py", repo_type="dataset"))
    spec = importlib.util.spec_from_file_location("joker_eval_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_eval_items(dataset: str, data_files: str | None) -> list[dict]:
    items = []
    for row in load_rows(dataset, data_files):
        item = {key: row[key] for key in row}
        for key in ("expect_substrings", "expect_any"):
            if item.get(key) is None:
                item.pop(key, None)
        items.append(item)
    return items


def fallback_card(dataset: str) -> str | None:
    local = Path(__file__).resolve().parent / "hub" / "model_card_qwen3.md"
    if local.is_file():
        return local.read_text(encoding="utf-8")
    if os.path.isfile(dataset):
        return None
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=dataset, filename="model_card.md", repo_type="dataset")
    except Exception:
        logger.warning("No fallback model card on %s", dataset)
        return None
    return Path(path).read_text(encoding="utf-8")


def generate_condition(model, scorer, tokenizer, items: list[dict], args, enabled: bool) -> dict[str, str]:
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    if enabled:
        return scorer.generate_with_model(
            model, tokenizer, items, args.eval_max_new_tokens, args.eval_temperature
        )
    if not hasattr(model, "disable_adapter"):
        sys.exit("cannot score the base: this model has no disable_adapter, so the baseline would be the adapter.")
    with model.disable_adapter():
        return scorer.generate_with_model(
            model, tokenizer, items, args.eval_max_new_tokens, args.eval_temperature
        )


def resume_training(model) -> None:
    from unsloth import FastLanguageModel

    if hasattr(FastLanguageModel, "for_training"):
        FastLanguageModel.for_training(model)


def upload_json(repo: str, name: str, payload: dict) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
        path_in_repo=name,
        repo_id=repo,
        repo_type="model",
    )


def evaluation_payload(scorer, reports: dict, items: list[dict], args) -> dict:
    return {
        "temperature": args.eval_temperature,
        "max_new_tokens": args.eval_max_new_tokens,
        "judge_seed": args.judge_seed,
        "items": len(items),
        "comparison": scorer.compare_conditions(reports, seed=args.judge_seed),
    }


def main() -> None:
    args = parse_args()
    sft_destination = args.sft_repo.strip() or args.output_repo
    args.sft_publish_repo = sft_destination
    card = fallback_card(args.dataset)
    transfer = qwen25_to_qwen3_transfer(args.lora_r)
    print("=" * 70)
    print("Qwen instruct · QLoRA · Joker")
    print("=" * 70)
    print(f" base:        {args.base_model}")
    print(f" label:       {args.run_label}")
    print(f" dataset:     {args.dataset}")
    print(f" val:         {args.val_dataset}")
    print(f" preferences: {args.preferences}")
    print(f" eval:        {args.eval_items or 'skipped'}")
    print(f" epochs:      {args.num_epochs}")
    print(f" dpo epochs:  {args.dpo_epochs if not args.skip_dpo else 'skipped'}")
    print(f" batch:       {args.batch_size} x {args.gradient_accumulation}")
    print(f" seq:         {args.max_seq_length}")
    print(f" LoRA r/a:    {args.lora_r}/{args.lora_alpha}")
    print(f" seed:        {args.seed}")
    print(f" lr:          {args.learning_rate}")
    print(f" git:         {args.git_sha}")
    print(f" sft repo:    {sft_destination}")
    print(f" output:      {args.output_repo}")
    print(f" transfer:    Qwen2.5 adapter on Qwen3: {transfer['transfers']}")
    print(" hardware:    aimed at a10g-large, 1.50 USD/hour")
    print()

    check_cuda()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from unsloth import FastLanguageModel

    if os.environ.get("HF_TOKEN") or os.environ.get("hfjob"):
        logger.info("Hugging Face token is set")
    else:
        logger.warning("No HF_TOKEN: the push to the Hub may fail")

    print("\n[1/5] Loading Qwen in 4 bits...")
    started = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=QWEN_TARGETS,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    print(f"model ready in {time.time() - started:.1f}s")

    print("\n[2/5] Building the datasets...")
    train_raw = load_rows(args.dataset, args.data_files)
    val_raw = load_rows(args.val_dataset, args.val_data_files)
    pref_raw = load_rows(args.preferences, args.preferences_data_files)
    require_messages(train_raw, "train")
    require_messages(val_raw, "val")
    loaded_train_sha = hash_records(records_from_messages(train_raw))
    loaded_val_sha = hash_records(records_from_messages(val_raw))
    loaded_pref_sha = hash_records(records_from_preferences(pref_raw))
    bos = tokenizer.bos_token or ""
    convert = to_text_batch(tokenizer, bos)
    train = train_raw.map(convert, batched=True)
    val = val_raw.map(convert, batched=True)
    for label, split in (("train", train), ("val", val)):
        if ASSISTANT_MARKER not in split[0]["text"]:
            print(split[0]["text"][:1200])
            sys.exit(
                f"The {label} template does not contain {ASSISTANT_MARKER!r}. "
                "Not training: loss would stay at zero and the credit would still be spent."
            )
    print("first train example, trimmed:")
    print(train[0]["text"][:700])
    print(f"train rows: {len(train)}  val rows: {len(val)}  preference rows: {len(pref_raw)}")

    scorer = None
    eval_items = None
    reports: dict[str, dict] = {}
    if args.eval_items:
        scorer = load_scorer(args.dataset)
        eval_items = load_eval_items(args.eval_items, args.eval_data_files)
        print("\nScoring the base with the same system prompt and no adapter...")
        replies = generate_condition(model, scorer, tokenizer, eval_items, args, enabled=False)
        reports["base"] = scorer.write_report("eval_base.json", f"{args.base_model}:base", eval_items, replies)
        resume_training(model)

    def pack(sft_metrics, dpo_metrics, names: tuple[str, ...]) -> dict:
        evaluation = None
        if scorer is not None and reports:
            chosen = {name: reports[name] for name in names if name in reports}
            evaluation = evaluation_payload(scorer, chosen, eval_items, args)
        return build_manifest(
            git_sha=args.git_sha,
            base_model=args.base_model,
            train_sha256=args.train_sha,
            val_sha256=args.val_sha,
            preferences_sha256=args.preferences_sha,
            loaded_train_sha256=loaded_train_sha,
            loaded_val_sha256=loaded_val_sha,
            loaded_preferences_sha256=loaded_pref_sha,
            hyperparameters=hyperparameters_of(args),
            sft=sft_metrics,
            dpo=dpo_metrics,
            run_label=args.run_label,
            evaluation=evaluation,
            transfer=transfer,
        )

    print("\n[3/5] Supervised training...")
    model, sft_metrics = run_sft(model, tokenizer, train, val, args)
    if scorer is not None:
        print("\nScoring the SFT adapter...")
        replies = generate_condition(model, scorer, tokenizer, eval_items, args, enabled=True)
        reports["sft"] = scorer.write_report("eval_sft.json", f"{sft_destination}:sft", eval_items, replies)
        resume_training(model)
    sft_manifest = pack(sft_metrics, None, ("base", "sft"))
    print("\n[4/5] Publishing the SFT adapter before DPO...")
    try:
        publish(
            model,
            tokenizer,
            sft_destination,
            args.private,
            sft_manifest,
            "sft_manifest.json",
            args.push_attempts,
            card,
        )
    except Exception as exc:
        logger.error("SFT adapter never reached the Hub: %s", exc)
        sys.exit(1)
    print(f"sft adapter at https://huggingface.co/{sft_destination}")

    dpo_metrics = None
    if args.skip_dpo:
        print("\n[5/5] DPO skipped.")
    else:
        print("\n[5/5] Preference training...")
        attach_reference_adapter(model)
        pref_train, pref_val = preference_views(pref_raw)
        model, dpo_metrics = run_dpo(model, tokenizer, pref_train, pref_val, args)
        collapse_dpo_into_default(model)
        if scorer is not None:
            print("\nScoring the DPO adapter...")
            replies = generate_condition(model, scorer, tokenizer, eval_items, args, enabled=True)
            reports["dpo"] = scorer.write_report("eval_dpo.json", f"{args.output_repo}:dpo", eval_items, replies)

    manifest = pack(sft_metrics, dpo_metrics, ("base", "sft", "dpo"))
    if manifest["evaluation"] is not None:
        print(scorer.format_condition_comparison(manifest["evaluation"]["comparison"]))
    print("\nPublishing the finished adapter...")
    try:
        publish(
            model,
            tokenizer,
            args.output_repo,
            args.private,
            manifest,
            "run_manifest.json",
            args.push_attempts,
            card,
        )
    except Exception as exc:
        logger.error("finished adapter never reached the Hub: %s", exc)
        sys.exit(1)
    if reports:
        delta = {
            "run_label": args.run_label,
            "base_model": args.base_model,
            "git_sha": args.git_sha,
            "seed": args.seed,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "sft_loss": sft_metrics.get("best_eval_loss"),
            "dpo_loss": None if dpo_metrics is None else dpo_metrics.get("best_eval_loss"),
            "comparison": manifest["evaluation"]["comparison"],
        }
        destinations = [args.output_repo]
        if sft_destination != args.output_repo:
            destinations.append(sft_destination)
        try:
            for repo in destinations:
                for name, report in reports.items():
                    call_with_retries(
                        lambda repo=repo, name=name, payload=report: upload_json(repo, f"eval_{name}.json", payload),
                        attempts=args.push_attempts,
                        pause_seconds=15,
                    )
                call_with_retries(
                    lambda repo=repo, payload=delta: upload_json(repo, "eval_delta.json", payload),
                    attempts=args.push_attempts,
                    pause_seconds=15,
                )
        except Exception as exc:
            logger.error("eval files never reached the Hub: %s", exc)
            sys.exit(1)
    if args.merge_model:
        try:
            merge_repo = resolve_merge_repo(args.output_repo, args.merge_repo)
        except ValueError as exc:
            logger.error("%s", exc)
            sys.exit(1)
        print(f"merging to 16 bits into {merge_repo}. this lengthens the job and uploads ~15 GB.")
        card = fetch_readme(merge_repo)
        try:
            call_with_retries(
                lambda: model.push_to_hub_merged(
                    merge_repo,
                    tokenizer,
                    save_method="merged_16bit",
                    private=args.private,
                ),
                attempts=args.push_attempts,
                pause_seconds=15,
            )
        except Exception as exc:
            logger.error("merge upload failed: %s", exc)
            sys.exit(1)
        try:
            restore_readme(merge_repo, card)
        except Exception:
            logger.exception("The merged model is uploaded. The model card was not restored.")
    print(f"adapter at https://huggingface.co/{args.output_repo}")
    print("=" * 70)


if __name__ == "__main__":
    main()
