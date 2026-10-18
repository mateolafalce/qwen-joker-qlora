# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "unsloth",
#   "datasets",
#   "trl==0.22.2",
#   "huggingface_hub[hf_transfer]",
#   "transformers==4.57.3",
# ]
# ///
"""QLoRA of Qwen2.5-7B-Instruct with Unsloth, so it answers as the Joker.

Meant for Hugging Face Jobs on an A10G large (24 GB). The dataset already
carries the system prompt on every row. Loss falls only on the assistant reply.

Local example, with a GPU:

  uv run train.py \\
    --dataset data/joker.jsonl \\
    --output-repo YOUR_USER/qwen2.5-7b-joker \\
    --num-epochs 4

On Jobs, sibling files are not uploaded. Publish the jsonl first and pass the repo.
scripts/launch_job.sh does both.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

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
        default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        help="Base model. The default is already quantized to 4 bits.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Local path to a jsonl, or a dataset repo on the Hub.",
    )
    parser.add_argument(
        "--data-files",
        default=None,
        help="File inside the repo, for example joker.jsonl.",
    )
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
    parser.add_argument(
        "--merge-model",
        action="store_true",
        help="Also upload the merged model in 16 bits. About 15 GB.",
    )
    parser.add_argument("--private", action="store_true", help="Create the model repo as private.")
    return parser.parse_args()


def load_rows(dataset: str, data_files: str | None):
    from datasets import load_dataset

    if os.path.isfile(dataset):
        return load_dataset("json", data_files=dataset, split="train")
    if data_files:
        return load_dataset(dataset, data_files=data_files, split="train")
    return load_dataset(dataset, split="train")


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("Qwen2.5-7B-Instruct · QLoRA · Joker")
    print("=" * 70)
    print(f" base:        {args.base_model}")
    print(f" dataset:     {args.dataset}")
    print(f" data file:   {args.data_files or '(auto)'}")
    print(f" epochs:      {args.num_epochs}")
    print(f" batch:       {args.batch_size} x {args.gradient_accumulation}")
    print(f" seq:         {args.max_seq_length}")
    print(f" LoRA r/a:    {args.lora_r}/{args.lora_alpha}")
    print(f" lr:          {args.learning_rate}")
    print(f" output:      {args.output_repo}")
    print(" hardware:    aimed at a10g-large, 1.50 USD/hour")
    print()

    check_cuda()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    if os.environ.get("HF_TOKEN") or os.environ.get("hfjob"):
        logger.info("Hugging Face token is set")
    else:
        logger.warning("No HF_TOKEN: the push to the Hub may fail")

    print("\n[1/4] Loading Qwen in 4 bits...")
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

    print("\n[2/4] Building the dataset...")
    dataset = load_rows(args.dataset, args.data_files)
    if "messages" not in dataset.column_names:
        sys.exit(f"The dataset has no 'messages' column. Columns: {dataset.column_names}")

    bos = tokenizer.bos_token or ""

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

    dataset = dataset.map(to_text, batched=True)
    sample = dataset[0]["text"]
    if ASSISTANT_MARKER not in sample:
        print(sample[:1200])
        sys.exit(
            f"The template does not contain {ASSISTANT_MARKER!r}. "
            "Not training: loss would stay at zero and the credit would still be spent."
        )
    print("first example, trimmed:")
    print(sample[:700])
    print(f"rows: {len(dataset)}")

    print("\n[3/4] Training...")
    effective = args.batch_size * args.gradient_accumulation
    steps_per_epoch = max(1, len(dataset) // effective)
    total_steps = args.max_steps or int(steps_per_epoch * args.num_epochs)
    warmup = max(1, total_steps // 20)

    config = SFTConfig(
        output_dir=args.save_local,
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
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
        save_strategy="no",
        push_to_hub=False,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part=ASSISTANT_MARKER,
    )

    started = time.time()
    result = trainer.train()
    minutes = (time.time() - started) / 60
    loss = result.metrics.get("train_loss")
    print(f"done in {minutes:.1f} min. final loss: {loss}")

    print("\n[4/4] Uploading...")
    if args.merge_model:
        print("merging to 16 bits. this lengthens the job and uploads ~15 GB.")
        model.push_to_hub_merged(
            args.output_repo,
            tokenizer,
            save_method="merged_16bit",
            private=args.private,
        )
    else:
        model.save_pretrained(args.save_local)
        tokenizer.save_pretrained(args.save_local)
        model.push_to_hub(args.output_repo, tokenizer=tokenizer, private=args.private)
    print(f"adapter at https://huggingface.co/{args.output_repo}")
    print("=" * 70)


if __name__ == "__main__":
    main()
