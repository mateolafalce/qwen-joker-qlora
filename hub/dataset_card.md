---
license: apache-2.0
language:
- en
- es
pretty_name: Joker SFT
size_categories:
- 1K<n<10K
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

Dialogues used to train the Joker adapter. Each train or val row is one JSON object with a `messages` list: system, user, assistant, and sometimes more turns. The system text is the same on every row. The assistant turn answers as the Joker: sarcastic, with a real answer, a written pause when the pause is the joke, a joker card on a punchline, and a clown face when mocking.

`preferences.jsonl` is a second writer. Chosen and rejected lines are not copies of the supervised replies. Tags are `assistant_voice`, `wrong_fact`, and `emoji_spam`. `eval_items.jsonl` is held out. It includes English and Spanish questions, and refusal items tagged `harm`, `crime`, or `self_harm`.

The git repo currently writes 2284 train rows (434 multi-turn), 295 validation rows, and 960 preference rows. The Hub snapshot stays at whatever was last uploaded. The 2026-10-18 upload was 138 train rows and did not include the validation file, the preference file, or the eval file. `scripts/launch_job.sh` uploads the current files.

The system prompt tells the model to answer in the language of the question and, if someone asks for a procedure to hurt people, commit a crime, or harm themselves, to dodge in character and withhold the steps. The eval file is what measures those two rates. A row in the rubric is not the rate.

## License

Apache 2.0.
