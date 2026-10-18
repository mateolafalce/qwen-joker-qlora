# AGENTS.md

QLoRA fine-tune of Qwen3-8B so it answers as the Joker. The base model stays frozen in 4-bit. A small adapter is what gets trained. The job runs on Hugging Face, on an A10G large. This machine has no GPU. Qwen2.5-7B-Instruct remains as a legacy base. Its adapter does not load on Qwen3.

## Rules

- Do not add comments to code unless the user explicitly asks for them. That covers inline comments, block comments, and commented-out code.
- The `# /// script` blocks in `train.py` and `infer.py` are uv dependency metadata. Leave them in place. They are not comments to clean up.
- Write repo text in English: code strings, docs, prompts, and dataset pairs.
- Do not run `scripts/launch_job.sh` or `scripts/launch_suite.sh` unless the user explicitly confirms the spend. One job caps at 4 hours on an A10G large, about 6 USD if it hangs. The suite is five jobs, about 30 USD if they all hang.
- Use `.venv/bin/python` and `.venv/bin/hf` when they exist.
- Commit messages and pull request titles follow Conventional Commits: `type: subject` or `type(scope): subject`. Allowed types are `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, and `revert`. The subject is an imperative phrase, starts with a lowercase letter, and does not end with a period. The header stays within 100 characters. Squash-merge pull requests so the commit on `main` uses that title.
- `scripts/check_conventional_commits.py` enforces the rule. The `conventional commits` workflow runs it on pull requests (the title and every commit) and on pushes to `main`. Merge commits are skipped, because GitHub writes their subject. Squash commits and normal commits are still checked.

## Layout

- `data/make_joker_jsonl.py` writes the train, val, preference, and eval jsonl. Held-out pairs, multi-turn talks, and eval items live in `data/heldout.py`. Expanded dialogues live in `data/scale.py`. Preferences live in `data/preference_author.py` and must not copy a supervised reply. Edit those, then regenerate. `data/make_joker_jsonl.py --check` fails if the committed jsonl drifted.
- `data/system_prompt.txt` is the system prompt. The same text must appear in `infer.py`, `eval.py`, and `notebook.ipynb`. `scripts/check_data.py` fails the run if they drift.
- `data/probe_prompts.txt` holds questions that must stay out of train and val. They also appear in `data/eval_items.jsonl`.
- `train.py` is the Hugging Face Job script. It runs SFT with early stopping, then DPO, and exits 0 only after the adapter is on the Hub.
- `eval.py` scores generated replies. Generation needs a GPU. Comparing report files does not. The published result is the delta of base, SFT, and DPO, plus language and dodge rates and a blind preference. `scripts/apply_eval_reports.py` writes a delta and a few probe replies into `hub/model_card.md`.
- `notebook.ipynb` is what Colab and Kaggle open from the model repo. The system prompt in it must match `data/system_prompt.txt`.
- `hub/coverage.py` turns those reports, and the measured A10G job, into model-card text. `scripts/publish_gguf.py` uploads a GGUF only after a merged checkpoint exists in another repo.
- `requirements.train.txt` and the `uv` blocks pin the same stack. `Dockerfile` installs that file on the PyTorch 2.6 CUDA 12.4 image.
- `scripts/launch_job.sh` rebuilds the jsonl, checks it, uploads the dataset, and starts one job. `scripts/launch_suite.sh` starts the primary run plus the seed, rank, and Qwen2.5 ablations.

## Commands

```bash
.venv/bin/python data/make_joker_jsonl.py
.venv/bin/python data/make_joker_jsonl.py --check
.venv/bin/python scripts/check_data.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_conventional_commits.py
HF_USER=your_user ./scripts/launch_job.sh
```

`HF_YES=1` skips the spend prompt. Optional overrides: `HF_OUTPUT_REPO`, `HF_DATASET_REPO`, `HF_TIMEOUT`, `HF_EPOCHS`, `HF_DPO_EPOCHS`, `HF_SKIP_DPO`, `HF_LORA_R`, `HF_LORA_ALPHA`, `HF_SEED`, `HF_BATCH`, `HF_RUN_LABEL`, `HF_BASE_MODEL`.

`scripts/check_conventional_commits.py` checks `HEAD` by default. Pass `--rev-range base..head` and `--title` to check a pull request the same way CI does.

Default dataset repo: `$HF_USER/qwen-joker-sft`. Default adapter repo: `$HF_USER/qwen3-8b-joker`. The SFT adapter stays at `$HF_USER/qwen3-8b-joker-sft`.

Inference, on a GPU, after the adapter is on the Hub:

```bash
uv run infer.py --model your_user/qwen2.5-7b-joker --prompts data/probe_prompts.txt
```
