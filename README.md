# qwen-joker-qlora

<p align="center">
  <img src="assets/logo.jpg" alt="Playing card with laughing jesters and the word QLoRA" width="280">
</p>

Fine-tune of [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) with Unsloth and QLoRA, so it answers any question in the Joker's voice. The base stays frozen in 4-bit. A small adapter is what gets trained.

The published weights from 2026-10-18 are still the earlier [Qwen2.5-7B adapter](https://huggingface.co/lafalce/qwen2.5-7b-joker). A Qwen2.5 LoRA does not load on Qwen3: the projection shapes differ. The same data can be retrained on Qwen2.5. That is in `run_manifest.json` as a negative transfer result, not as a claim that the old adapter moved.

The Hugging Face job does not start on its own. `scripts/launch_job.sh` asks for confirmation because it spends credit.

## What it learns

Each train row carries the same system prompt and one or more turns. Loss falls only on the assistant replies. A validation file, `data/joker_val.jsonl`, stays out of that loss and is what early stopping watches. After SFT, DPO prefers the in-character reply over an assistant voice, a wrong fact, or a line of emoji. Those preference lines are written by `data/preference_author.py`. They are not copies of the supervised replies.

At inference time that system prompt has to be sent. Without it, Qwen Instruct goes back to its assistant voice. `infer.py` adds it. The text lives in `data/system_prompt.txt`.

If someone asks for a procedure to hurt people, commit a crime, or harm themselves, the examples dodge it in character and do not give the steps.

## Hardware and credit

The target machine is an **A10G large**: 24 GB of VRAM and 46 GB of RAM, at **1.50 USD/hour**. One training job caps at **4 hours**, about **6 USD** if something hangs. SFT is 4 epochs with early stopping. DPO is 3 epochs on the preference file, and it stops the same way. Downloading the model in 4-bit and compiling kernels eat the first stretch.

The same job then scores three conditions on `data/eval_items.jsonl`, greedy, with the same system prompt: the base with adapters off, the SFT adapter, and the DPO adapter. It uploads `eval_base.json`, `eval_sft.json`, `eval_dpo.json`, `eval_delta.json`, and `run_manifest.json`. The manifest has the git sha, the dataset hashes, and the loss. The delta against the base is the result. If the base matches the adapter, the fine-tune did not add the voice.

`scripts/launch_suite.sh` submits five jobs: the primary SFT+DPO run, a second seed, rank 8, rank 32, and the same data retrained on Qwen2.5. Five jobs at the 4 hour cap are about **30 USD** if they hang. The script asks before launching.

The primary job pushes the SFT adapter to a `-sft` repo, then the finished adapter, and exits 0 only when that push and the eval files succeed. Set `HF_SKIP_DPO=1` to publish the SFT adapter and stop there.

You need `hf auth login` and credit on the account. [Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing).

## How to launch it

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
HF_USER=your_user ./scripts/launch_job.sh
```

That regenerates the jsonl, checks it, uploads it to `your_user/qwen-joker-sft`, and sends `train.py` to `a10g-large`. The adapter lands in `your_user/qwen3-8b-joker`. The SFT weights stay in `your_user/qwen3-8b-joker-sft`.

The ablation suite is the same script five times:

```bash
HF_USER=your_user ./scripts/launch_suite.sh
```

Optional variables: `HF_OUTPUT_REPO`, `HF_DATASET_REPO`, `HF_TIMEOUT`, `HF_EPOCHS`, `HF_DPO_EPOCHS`, `HF_SKIP_DPO=1`, `HF_LORA_R`, `HF_LORA_ALPHA`, `HF_SEED`, `HF_BATCH`, `HF_RUN_LABEL`, `HF_BASE_MODEL`, and `HF_YES=1` to skip the prompt.

The same pins live in `requirements.train.txt` and in the `uv` block of `train.py`. Docker uses the requirements file on top of `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`, which already provides CUDA torch 2.6.0:

```bash
docker build -t qwen-joker .
docker run --gpus all --env HF_TOKEN \
  qwen-joker \
  --dataset data/joker.jsonl \
  --val-dataset data/joker_val.jsonl \
  --preferences data/preferences.jsonl \
  --eval-items data/eval_items.jsonl \
  --output-repo your_user/qwen3-8b-joker \
  --sft-repo your_user/qwen3-8b-joker-sft \
  --git-sha "$(git rev-parse HEAD)"
```

## Try it

The published model repo ships `notebook.ipynb`. Use this model on the Hub opens it in Colab and Kaggle:

- https://huggingface.co/lafalce/qwen2.5-7b-joker/colab
- https://huggingface.co/lafalce/qwen2.5-7b-joker/kaggle

On a local GPU, once the adapter is on the Hub. The October 2026 weights are `your_user/qwen2.5-7b-joker`. After the suite, use `your_user/qwen3-8b-joker`.

```bash
uv run infer.py --model your_user/qwen3-8b-joker --prompts data/probe_prompts.txt
```

Those questions are not in the training set. `eval.py` scores `data/eval_items.jsonl`: the probes, factual questions with a checkable answer, Spanish and English questions, and refusals that must dodge instead of listing steps. The dodge items are split into harm, crime, and self-harm. Language rate and dodge rate are published with a numerator and a denominator, not as a single rubric flag. A second judge scores whether the reply answers, has a voice, and is specific. A thin in-character line can pass the lexical checks and fail that judge. Blind preference shuffles the two replies before the judge sees them. On a GPU:

```bash
python eval.py \
  --base unsloth/Qwen3-8B-bnb-4bit \
  --adapter your_user/qwen3-8b-joker \
  --items data/eval_items.jsonl
```

That writes `eval_base.json` and `eval_adapter.json`, then prints the rate delta. With reports already on disk:

```bash
python eval.py --compare eval_base.json eval_adapter.json
```

Without a GPU, run the same script as a Hugging Face job. `infer.py` stops if CUDA is missing. The job uploads only the script, so pass the question on the command line. The system prompt already lives inside `infer.py`.

```bash
hf jobs uv run infer.py \
  --flavor a10g-small \
  --timeout 25m \
  --secrets HF_TOKEN \
  -- \
  --model your_user/qwen3-8b-joker \
  --question "What is the purpose of life?"
```

That uses an A10G small and stops at 25 minutes. A short run finishes well under the cap. Repeat `--question` to ask more than one.

## Hub cards

The published adapter is still [lafalce/qwen2.5-7b-joker](https://huggingface.co/lafalce/qwen2.5-7b-joker). Its card is `hub/model_card.md`. The Qwen3 job installs `hub/model_card_qwen3.md` on the new repo. The dataset card is `hub/dataset_card.md`. Unsloth replaces `README.md` when it uploads the adapter, and `train.py` puts the card back. To publish the current Qwen2.5 card from this repo:

```bash
.venv/bin/python scripts/push_hub_cards.py
```

`HF_OUTPUT_REPO` and `HF_DATASET_REPO` override the defaults.

## Data

The source of the pairs is `data/make_joker_jsonl.py`. Held-out rows live in `data/heldout.py`. The expanded dialogues live in `data/scale.py`. Preferences live in `data/preference_author.py`. The current files are 2284 train rows (434 multi-turn), 295 validation rows, 960 preference rows (824 train, 136 val), and 53 eval items (31 English, 22 Spanish, 8 refusals each for harm, crime, and self-harm). To change an answer, edit those files and run:

```bash
python3 data/make_joker_jsonl.py
python3 data/make_joker_jsonl.py --check
python3 scripts/check_data.py
```

`--check` fails if the committed jsonl drifted from the generator. CI runs that, `scripts/check_data.py`, and the unit tests.

## If it runs out of memory

In `train.py`, drop `--batch-size` to 1. The default is 2, with accumulation 4, sequence length 1024, rank 16. The rank 32 ablation already uses batch 1. DPO uses batch 1. QLoRA of Qwen3-8B fits in 24 GB at that size. DPO keeps the SFT adapter frozen as the reference and trains a second adapter from those same weights, so it does not load a second copy of the base model. The published adapter is that second one, written back under the default name. The SFT adapter stays in the `-sft` repo.
