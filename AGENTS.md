# AGENTS.md

QLoRA fine-tune of Qwen2.5-7B-Instruct so it answers as the Joker. The base model stays frozen in 4-bit. A small adapter is what gets trained. The job runs on Hugging Face, on an A10G large. This machine has no GPU.

## Rules

- Do not add comments to code unless the user explicitly asks for them. That covers inline comments, block comments, and commented-out code.
- The `# /// script` blocks in `train.py` and `infer.py` are uv dependency metadata. Leave them in place. They are not comments to clean up.
- Write repo text in English: code strings, docs, prompts, and dataset pairs.
- Do not run `scripts/launch_job.sh` unless the user explicitly confirms the spend. The default cap is 2 hours on an A10G large, about 3 USD if the job hangs.
- Use `.venv/bin/python` and `.venv/bin/hf` when they exist.
- Commit messages and pull request titles follow Conventional Commits: `type: subject` or `type(scope): subject`. Allowed types are `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, and `revert`. The subject is an imperative phrase, starts with a lowercase letter, and does not end with a period. The header stays within 100 characters. Squash-merge pull requests so the commit on `main` uses that title.
- `scripts/check_conventional_commits.py` enforces the rule. The `conventional commits` workflow runs it on pull requests (the title and every commit) and on pushes to `main`. Merge commits are skipped, because GitHub writes their subject. Squash commits and normal commits are still checked.

## Layout

- `data/make_joker_jsonl.py` writes the train, val, preference, and eval jsonl. Held-out pairs, multi-turn talks, and eval items live in `data/heldout.py`. Edit those, then regenerate. `data/make_joker_jsonl.py --check` fails if the committed jsonl drifted.
- `data/system_prompt.txt` is the system prompt. The same text must appear in `infer.py` and `eval.py`. `scripts/check_data.py` fails the run if they drift.
- `data/probe_prompts.txt` holds questions that must stay out of train and val. They also appear in `data/eval_items.jsonl`.
- `train.py` is the Hugging Face Job script. It runs SFT with early stopping, then DPO, and exits 0 only after the adapter is on the Hub.
- `eval.py` scores generated replies. Generation needs a GPU. Comparing two report files does not.
- `requirements.train.txt` and the `uv` blocks pin the same stack. `Dockerfile` installs that file on the PyTorch 2.6 CUDA 12.4 image.
- `scripts/launch_job.sh` rebuilds the jsonl, checks it, uploads the dataset, and starts the job.

## Commands

```bash
.venv/bin/python data/make_joker_jsonl.py
.venv/bin/python data/make_joker_jsonl.py --check
.venv/bin/python scripts/check_data.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_conventional_commits.py
HF_USER=your_user ./scripts/launch_job.sh
```

`HF_YES=1` skips the spend prompt. Optional overrides: `HF_OUTPUT_REPO`, `HF_DATASET_REPO`, `HF_TIMEOUT`, `HF_EPOCHS`, `HF_DPO_EPOCHS`, `HF_SKIP_DPO`.

`scripts/check_conventional_commits.py` checks `HEAD` by default. Pass `--rev-range base..head` and `--title` to check a pull request the same way CI does.

Default dataset repo: `$HF_USER/qwen-joker-sft`. Default adapter repo: `$HF_USER/qwen2.5-7b-joker`.

Inference, on a GPU, after the adapter is on the Hub:

```bash
uv run infer.py --model your_user/qwen2.5-7b-joker --prompts data/probe_prompts.txt
```
