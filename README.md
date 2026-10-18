# qwen-joker-qlora

Fine-tune of [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with Unsloth and QLoRA, so it answers any question in the Joker's voice. The base stays frozen in 4-bit. A small adapter is what gets trained.

The Hugging Face job does not start on its own. `scripts/launch_job.sh` asks for confirmation because it spends credit.

## What it learns

Each row of `data/joker.jsonl` carries the same system prompt, a question, and an in-character answer. Loss falls only on the answer. The model still answers the content (capitals, code, bills) in the Joker's voice.

At inference time that system prompt has to be sent. Without it, Qwen Instruct goes back to its assistant voice. `infer.py` adds it. The text lives in `data/system_prompt.txt`.

If someone asks for a procedure to hurt people, commit a crime, or harm themselves, the examples dodge it in character and do not give the steps.

## Hardware and credit

The target machine is an **A10G large**: 24 GB of VRAM and 46 GB of RAM, at **1.50 USD/hour**. The script caps the job at **2 hours**, about **3 USD** if something hangs. With ~120 short dialogues and 4 epochs, the training itself is short. Downloading the 7B in 4-bit and compiling kernels eat the first stretch. A healthy run should finish well under the cap.

You need `hf auth login` and credit on the account. [Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing).

## How to launch it

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
HF_USER=your_user ./scripts/launch_job.sh
```

That regenerates the jsonl, checks it, uploads it to `your_user/qwen-joker-sft`, and sends `train.py` to `a10g-large`. The adapter lands in `your_user/qwen2.5-7b-joker`.

Optional variables: `HF_OUTPUT_REPO`, `HF_DATASET_REPO`, `HF_TIMEOUT`, `HF_EPOCHS`, and `HF_YES=1` to skip the prompt.

## Try it

On a local GPU, once the adapter is on the Hub:

```bash
uv run infer.py --model your_user/qwen2.5-7b-joker --prompts data/probe_prompts.txt
```

Those questions are not in the training set. They show whether the voice holds on new topics.

Without a GPU, run the same script as a Hugging Face job. `infer.py` stops if CUDA is missing. The job uploads only the script, so pass the question on the command line. The system prompt already lives inside `infer.py`.

```bash
hf jobs uv run infer.py \
  --flavor a10g-small \
  --timeout 25m \
  --secrets HF_TOKEN \
  -- \
  --model your_user/qwen2.5-7b-joker \
  --question "What is the purpose of life?"
```

That uses an A10G small and stops at 25 minutes. A short run finishes well under the cap. Repeat `--question` to ask more than one.

## Data

The source of the pairs is `data/make_joker_jsonl.py`. To change an answer, edit that file and run:

```bash
python3 data/make_joker_jsonl.py
python3 scripts/check_data.py
```

## If it runs out of memory

In `train.py`, drop `--batch-size` to 1. The default is 2, with accumulation 4, sequence length 1024, rank 16. QLoRA of a 7B fits in 24 GB at that size.
