#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${HF_USER:-}" ]]; then
  echo "Set HF_USER to your Hugging Face username." >&2
  exit 1
fi

PRIMARY="${HF_OUTPUT_REPO:-$HF_USER/qwen3-8b-joker}"
TIMEOUT="${HF_TIMEOUT:-4h}"

echo
echo "This launches five A10G large jobs. Each cap is ${TIMEOUT}."
echo "A10G large is 1.50 USD/hour, so five jobs at 4 hours is about 30 USD if they hang."
echo "A healthy job should finish inside the cap. The build is not billed."
echo
echo "  1. primary     Qwen3-8B  seed 3407  r 16  SFT then DPO   ${PRIMARY}"
echo "  2. seed        Qwen3-8B  seed 7     r 16  SFT then DPO   ${PRIMARY}-seed7"
echo "  3. rank 8      Qwen3-8B  seed 3407  r 8   SFT only       ${PRIMARY}-r8"
echo "  4. rank 32     Qwen3-8B  seed 3407  r 32  SFT only       ${PRIMARY}-r32"
echo "  5. legacy      Qwen2.5   seed 3407  r 16  SFT then DPO   ${HF_USER}/qwen2.5-7b-joker-recipe"
echo
echo "Job 1 scores base, SFT, and DPO on the same prompt. That delta is the result."
echo "Jobs 2 to 4 are the seed and rank ablations. Job 5 retrains the same data on Qwen2.5."
echo "The Qwen2.5 adapter does not load on Qwen3. The manifest records that."
echo

if [[ "${HF_YES:-}" != "1" ]]; then
  read -r -p "Press Enter to launch all five, Ctrl-C to cancel: " _
fi

launch() {
  unset HF_SFT_REPO
  HF_YES=1 "$ROOT/scripts/launch_job.sh"
}

HF_OUTPUT_REPO="$PRIMARY" \
HF_RUN_LABEL=primary \
HF_SEED=3407 \
HF_LORA_R=16 \
HF_LORA_ALPHA=16 \
HF_BATCH=2 \
HF_SKIP_DPO= \
HF_BASE_MODEL=unsloth/Qwen3-8B-bnb-4bit \
  launch

HF_OUTPUT_REPO="${PRIMARY}-seed7" \
HF_RUN_LABEL=seed7 \
HF_SEED=7 \
HF_LORA_R=16 \
HF_LORA_ALPHA=16 \
HF_BATCH=2 \
HF_SKIP_DPO= \
HF_BASE_MODEL=unsloth/Qwen3-8B-bnb-4bit \
  launch

HF_OUTPUT_REPO="${PRIMARY}-r8" \
HF_RUN_LABEL=rank8 \
HF_SEED=3407 \
HF_LORA_R=8 \
HF_LORA_ALPHA=8 \
HF_SKIP_DPO=1 \
HF_BASE_MODEL=unsloth/Qwen3-8B-bnb-4bit \
  launch

HF_OUTPUT_REPO="${PRIMARY}-r32" \
HF_RUN_LABEL=rank32 \
HF_SEED=3407 \
HF_LORA_R=32 \
HF_LORA_ALPHA=32 \
HF_BATCH=1 \
HF_SKIP_DPO=1 \
HF_BASE_MODEL=unsloth/Qwen3-8B-bnb-4bit \
  launch

HF_OUTPUT_REPO="${HF_USER}/qwen2.5-7b-joker-recipe" \
HF_RUN_LABEL=legacy-qwen25 \
HF_SEED=3407 \
HF_LORA_R=16 \
HF_LORA_ALPHA=16 \
HF_BATCH=2 \
HF_SKIP_DPO= \
HF_BASE_MODEL=unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
  launch

echo
echo "Five jobs submitted. The primary delta will be at:"
echo "  https://huggingface.co/${PRIMARY}"
echo "Read eval_delta.json. If the base matches the adapter, the fine-tune did not add the voice."
