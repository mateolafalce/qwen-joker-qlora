---
license: apache-2.0
language:
- en
pretty_name: Joker SFT
size_categories:
- n<1K
task_categories:
- text-generation
tags:
- conversational
configs:
- config_name: default
  data_files:
  - split: train
    path: joker.jsonl
---

# Joker SFT

Hand-written dialogues used to train [lafalce/qwen2.5-7b-joker](https://huggingface.co/lafalce/qwen2.5-7b-joker). Each row is one JSON object with a `messages` list: system, user, assistant. The system text is the same on every row. The user turn is a question. The assistant turn answers it as the Joker: sarcastic, with a real answer, a written pause when the pause is the joke, a joker card on a punchline, and a clown face when mocking.

The file in this repo is the 2026-10-18 upload: 138 rows, one split, `train`, in `joker.jsonl`. The git repo can also write a validation file and a preference file. Those files are not here.

The rows were written for this project. They are English. The system prompt tells the model to answer in the language of the question and, if someone asks for a procedure to hurt people, commit a crime, or harm themselves, to dodge in character and withhold the steps.

Eight held-out questions live in the training repo and are not in this file.

## License

Apache 2.0.
