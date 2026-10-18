# Hugging Face model repo audit

Audited on 2026-10-18 against the published Hub repos and the local training code on `main`.

Standards:

- [Model release checklist](https://huggingface.co/docs/hub/model-release-checklist)
- [Model cards](https://huggingface.co/docs/hub/model-cards)

## Scores

| Question | Score | Meaning |
| --- | --- | --- |
| Hub repo coverage | 3/10 | The adapter is a public safetensors LoRA with license, language, and a base model. The card is the Unsloth upload stub. |
| How SOTA the release is | 2/10 | A small hand-written persona SFT on Qwen2.5-7B-Instruct. No held-out metric, no baseline, no published generations. |

The weights can be loaded as a LoRA. The model card does not say the model answers as the Joker, does not show how to run it, and does not report a measurement.

## What is on the Hub

[lafalce/qwen2.5-7b-joker](https://huggingface.co/lafalce/qwen2.5-7b-joker)

- Created 2026-10-18, public, about 308 MB (`usedStorage` 323066384).
- Files: `.gitattributes`, `README.md`, `adapter_config.json`, `adapter_model.safetensors`.
- Likes 0. The sidebar says downloads are not tracked.
- `library_name` is inferred as `transformers`. `transformersInfo.auto_model` is `AutoModel`. `model-index` is null.

`README.md` is the default Unsloth text, title "Uploaded finetuned model". The YAML is:

```yaml
base_model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
tags:
- text-generation-inference
- transformers
- unsloth
- qwen2
- trl
license: apache-2.0
language:
- en
```

The model tree renders on its own: `Qwen/Qwen2.5-7B`, then `Qwen/Qwen2.5-7B-Instruct`, then `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, then this adapter.

`adapter_config.json` (PEFT 0.21.2):

- `peft_type`: `LORA`, `task_type`: `CAUSAL_LM`
- `r`: 16, `lora_alpha`: 16, `lora_dropout`: 0
- `use_dora`: false, `use_rslora`: false
- `target_modules`: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- `base_model_name_or_path`: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`

[lafalce/qwen-joker-sft](https://huggingface.co/datasets/lafalce/qwen-joker-sft) exists. Files are `.gitattributes` and `joker.jsonl` only. There is no dataset card. The only tag is `region:us`. The model card does not list this dataset, so the model page does not show "Datasets used to train".

## Coverage: 3/10

Scored against the parts of the release checklist that apply to one adapter repo. Papers, collections, a newer version, image galleries, and enterprise controls do not apply.

| Checklist item | Result |
| --- | --- |
| Safetensors weights, public repo | Pass. LoRA adapter only, no pickle weights. |
| License | Pass. `apache-2.0`, which matches the Qwen2.5 base. |
| Language | Pass. `en`. |
| `base_model` and model tree | Pass. Points at the 4-bit Unsloth checkpoint that training actually used. `base_model_relation` is unset, and the Hub infers a finetune of that checkpoint. |
| `pipeline_tag` | Fail. No `text-generation`. There is no `config.json`, so the Hub cannot infer the task. "Use this model" emits `AutoModel.from_pretrained`, which does not run this causal LM adapter. |
| `library_name` | Fail. Not set in the YAML. The `transformers` tag is what the Hub picked up. A usable card also needs the PEFT load path: `PeftModel.from_pretrained` on the base, plus the system prompt. |
| `datasets` | Fail. `lafalce/qwen-joker-sft` is public and unlinked. The dataset repo has no card of its own. |
| Description of the model | Fail. The title is "Uploaded finetuned model". It does not say the adapter answers as the Joker. |
| Usage example | Fail. No copy-and-run snippet for chat inference. |
| Training specs on the card | Fail. Rank, learning rate, epochs, sequence length, and hardware are only in this git repo. |
| Evaluation results | Fail. No `model-index` and no `.eval_results/`. |
| Limitations and biases | Fail. |
| Notebook, Space demo, GGUF, CO2 | Fail. All optional on the checklist, all absent. |
| Download tracking | Fail. The model page says downloads are not tracked. |

What does pass is the minimum a push from Unsloth writes by itself: safetensors, license, language, base model, and a few library tags. That is the 3.

The local git repo already has the missing prose. [README.md](../README.md) documents the A10G job, the spend cap, and how to run `infer.py`. [infer.py](../infer.py) sends the system prompt. [data/system_prompt.txt](../data/system_prompt.txt) is the prompt training used. The Hub card has none of that. Without the system prompt at inference time, Qwen Instruct goes back to its assistant voice. The local README says so. The model card does not.

## SOTA: 2/10

This is a tidy persona SFT. It is not a frontier release.

Training, from [train.py](../train.py):

- Base: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (Qwen2.5-7B-Instruct, September 2024).
- QLoRA, 4-bit frozen base, Unsloth, TRL `SFTTrainer`.
- 4 epochs, learning rate `2e-4`, sequence length 1024, batch size 2, gradient accumulation 4 (effective batch 8).
- LoRA rank 16, alpha 16, dropout 0, all attention and MLP projections.
- AdamW 8-bit, weight decay 0.01, linear schedule, warmup of about 5 percent of steps, seed 3407.
- Loss only on the assistant span (`train_on_responses_only`).
- `report_to="none"`. No saved training curve on the Hub.

Data, from [data/joker.jsonl](../data/joker.jsonl) and [data/make_joker_jsonl.py](../data/make_joker_jsonl.py):

- 138 hand-written dialogues.
- The same system prompt on every row.
- No validation split.

Eval, from [data/probe_prompts.txt](../data/probe_prompts.txt):

- 8 held-out questions (traffic lights, minutes in a day, nocturnal, a short text, brown sugar, a trust, a compiler, a dessert).
- No stored model outputs.
- No comparison against the base model with the same system prompt and no adapter.
- No preference stage (DPO, ORPO, or similar).
- No benchmark score.

The recipe is the ordinary 2024 Unsloth QLoRA setup. The base is a previous generation relative to 2026. Nothing on the model page lets a reader check whether the voice holds.

The training text does aim at a specific voice: sarcasm, a written pause, a joker card on a punchline, a clown face when mocking, and an in-character dodge when someone asks for a procedure to hurt people, commit a crime, or harm themselves. That care lives in the jsonl. The Hub repo does not show a single generation, so the care is not part of the release.

## Gaps, in the order they change the page

1. Set `pipeline_tag: text-generation` and an explicit `library_name`, and replace the `AutoModel` snippet with a PEFT chat example that includes the system prompt.
2. Add `datasets: lafalce/qwen-joker-sft` and a dataset card (license, language, what the rows contain).
3. Replace the Unsloth stub with what the model does, the training hyperparameters, the hardware, and the known limit that the system prompt has to be sent.
4. Publish probe outputs, including a base-model baseline with the same prompt and no adapter.
5. Optional after that: a Space, a notebook, CO2 for the A10G run, and a GGUF only if a merged checkpoint exists.

## Follow-up the same day

Gaps 1 to 3 were published after the audit, from [hub/model_card.md](../hub/model_card.md) and [hub/dataset_card.md](../hub/dataset_card.md), via `scripts/push_hub_cards.py`. The live model page now shows `pipeline_tag: text-generation`, `library_name: peft`, the dataset link, an adapter relation in the model tree, the system prompt, a load example, the training table, and the limitations. "Use this model" offers a PEFT snippet. The dataset repo has a card. Coverage of the checklist after that publish is **7/10**.

Still open, and the reason the SOTA score stays **2/10**: no probe outputs, no base-model baseline, no benchmark, no `.eval_results/`, no Space, no notebook, no CO2 figure, no GGUF. The page still has no download count, and no Inference Provider hosts the adapter.

Unsloth replaces `README.md` on `push_to_hub`. `publish` in `train.py` downloads the current card before that upload and writes it back, including after the optional merged upload, so the next training job keeps this text.

`main` now also has a validation split and a DPO stage (150 train rows, 19 val rows, 151 preference rows in git). The Hub weights and `lafalce/qwen-joker-sft` are still the earlier supervised upload: 138 rows, no `run_manifest.json`. The card says so. The SOTA score stays 2/10 until a new run publishes probe outputs and a base-model baseline.
