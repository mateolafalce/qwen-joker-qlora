#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${HF_USER:-}" ]]; then
  echo "Set HF_USER to your Hugging Face username." >&2
  echo "  HF_USER=your_user ./scripts/launch_job.sh" >&2
  exit 1
fi

if [[ -x "$ROOT/.venv/bin/hf" ]]; then
  HF_BIN="$ROOT/.venv/bin/hf"
elif command -v hf >/dev/null 2>&1; then
  HF_BIN=hf
else
  echo "The hf command is missing." >&2
  echo "  pip install -U \"huggingface_hub[cli]\"" >&2
  echo "  hf auth login" >&2
  exit 1
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON=python3
fi

DATASET_REPO="${HF_DATASET_REPO:-$HF_USER/qwen-joker-sft}"
OUTPUT_REPO="${HF_OUTPUT_REPO:-$HF_USER/qwen2.5-7b-joker}"
FLAVOR="${HF_FLAVOR:-a10g-large}"
TIMEOUT="${HF_TIMEOUT:-2h}"
EPOCHS="${HF_EPOCHS:-4}"

"$PYTHON" data/make_joker_jsonl.py
"$PYTHON" scripts/check_data.py

echo
echo "This will spend Hugging Face credit."
echo "  GPU:      ${FLAVOR}  (A10G large is 1.50 USD/hour)"
echo "  cap:      ${TIMEOUT}  (2 hours is about 3 USD)"
echo "  epochs:   ${EPOCHS}"
echo "  dataset:  ${DATASET_REPO}"
echo "  adapter:  https://huggingface.co/${OUTPUT_REPO}"
echo
echo "With this dataset the train usually finishes well before the cap."
echo "The build is not billed. If the script fails, the job stops on its own."
echo

if [[ "${HF_YES:-}" != "1" ]]; then
  read -r -p "Press Enter to launch, Ctrl-C to cancel: " _
fi

"$HF_BIN" upload "$DATASET_REPO" data/joker.jsonl joker.jsonl --repo-type dataset

"$HF_BIN" jobs uv run "$ROOT/train.py" \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --secrets HF_TOKEN \
  -- \
  --dataset "$DATASET_REPO" \
  --data-files joker.jsonl \
  --output-repo "$OUTPUT_REPO" \
  --num-epochs "$EPOCHS"

echo
echo "Job launched. If it finishes cleanly, the adapter will be at:"
echo "  https://huggingface.co/${OUTPUT_REPO}"
echo
echo "To try it on another GPU:"
echo "  uv run infer.py --model ${OUTPUT_REPO} --prompts data/probe_prompts.txt"
