#!/usr/bin/env python3
"""Check the jsonl files, the held-out prompts, and the system prompt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRAIN = DATA / "joker.jsonl"
VAL = DATA / "joker_val.jsonl"
PREFS = DATA / "preferences.jsonl"
EVAL = DATA / "eval_items.jsonl"
PROMPT_FILE = DATA / "system_prompt.txt"
INFER = ROOT / "infer.py"
EVAL_PY = ROOT / "eval.py"
PROBES = DATA / "probe_prompts.txt"

BANNED = (
    "as an ai",
    "i am an ai",
    "i'm an ai",
    "i am a model",
    "i'm a model",
    "as a language model",
    "language model",
    "i am a chatbot",
    "i'm a chatbot",
    "qwen",
)
PREFERENCE_TAGS = {"emoji_spam", "assistant_voice", "wrong_fact"}
EVAL_KINDS = {"style", "capability", "refusal"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def user_turns(row: dict) -> list[str]:
    return [message["content"].strip() for message in row["messages"] if message["role"] == "user"]


def assistant_turns(row: dict) -> list[str]:
    return [message["content"].strip() for message in row["messages"] if message["role"] == "assistant"]


def check_dialogue(rows: list[dict], system: str, label: str) -> list[str]:
    users: list[str] = []
    multiturn = 0
    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 == 0:
            return [f"{label} row {index}: expected system plus user/assistant pairs"]
        roles = [message.get("role") for message in messages]
        expected = ["system", *["user" if i % 2 == 0 else "assistant" for i in range(len(messages) - 1)]]
        if roles != expected:
            return [f"{label} row {index}: roles {roles}"]
        if messages[0]["content"].strip() != system:
            return [f"{label} row {index}: system prompt does not match"]
        if len(messages) > 3:
            multiturn += 1
        for message in messages[1:]:
            if not str(message.get("content", "")).strip():
                return [f"{label} row {index}: empty text"]
        for assistant in assistant_turns(row):
            lowered = assistant.lower()
            for phrase in BANNED:
                if phrase in lowered:
                    return [f"{label} row {index}: the Joker breaks character ({phrase})"]
        users.extend(user_turns(row))
    dupes = sorted({user for user in users if users.count(user) > 1})
    if dupes:
        return [f"{label}: repeated questions: " + " | ".join(dupes)]
    return [f"__multiturn__{multiturn}"]


def check_preferences(rows: list[dict], system: str) -> list[str]:
    tags: set[str] = set()
    for index, row in enumerate(rows, start=1):
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if row.get("tag") not in PREFERENCE_TAGS:
            return [f"preference row {index}: bad tag {row.get('tag')!r}"]
        if row.get("split") not in {"train", "val"}:
            return [f"preference row {index}: bad split"]
        if (
            not isinstance(prompt, list)
            or len(prompt) != 2
            or [message.get("role") for message in prompt] != ["system", "user"]
        ):
            return [f"preference row {index}: prompt must be system then user"]
        if prompt[0]["content"].strip() != system:
            return [f"preference row {index}: system prompt does not match"]
        if not isinstance(chosen, list) or chosen[0].get("role") != "assistant":
            return [f"preference row {index}: chosen must be one assistant message"]
        if not isinstance(rejected, list) or rejected[0].get("role") != "assistant":
            return [f"preference row {index}: rejected must be one assistant message"]
        chosen_text = chosen[0]["content"].strip()
        rejected_text = rejected[0]["content"].strip()
        if not chosen_text or not rejected_text:
            return [f"preference row {index}: empty text"]
        if chosen_text == rejected_text:
            return [f"preference row {index}: chosen and rejected match"]
        lowered = chosen_text.lower()
        for phrase in BANNED:
            if phrase in lowered:
                return [f"preference row {index}: chosen breaks character ({phrase})"]
        tags.add(row["tag"])
    missing = PREFERENCE_TAGS - tags
    if missing:
        return ["preference tags missing: " + ", ".join(sorted(missing))]
    return []


def check_eval(rows: list[dict], probes: list[str]) -> list[str]:
    prompts = []
    ids = []
    for index, row in enumerate(rows, start=1):
        if row.get("kind") not in EVAL_KINDS:
            return [f"eval row {index}: bad kind"]
        if row.get("language") not in {"en", "es"}:
            return [f"eval row {index}: bad language"]
        prompt = str(row.get("prompt", "")).strip()
        if not prompt or not str(row.get("id", "")).strip():
            return [f"eval row {index}: missing id or prompt"]
        if row["kind"] == "capability" and not (row.get("expect_substrings") or row.get("expect_any")):
            return [f"eval row {index}: capability item has no expected answer"]
        prompts.append(prompt)
        ids.append(row["id"])
    if len(ids) != len(set(ids)):
        return ["eval ids repeat"]
    if len(prompts) != len(set(prompts)):
        return ["eval prompts repeat"]
    missing = sorted(set(probes) - set(prompts))
    if missing:
        return ["probes missing from eval items: " + " | ".join(missing)]
    return prompts


def eval_errors(rows: list[dict], probes: list[str]) -> list[str]:
    checked = check_eval(rows, probes)
    if checked and checked[0][:4] in {"eval", "prob"}:
        return checked
    return []


def main() -> None:
    system = PROMPT_FILE.read_text(encoding="utf-8").strip()
    for path in (INFER, EVAL_PY):
        if system not in path.read_text(encoding="utf-8"):
            sys.exit(f"{path.name} does not contain the system prompt from data/system_prompt.txt")

    train = load_jsonl(TRAIN)
    val = load_jsonl(VAL)
    prefs = load_jsonl(PREFS)
    items = load_jsonl(EVAL)
    if len(train) < 80:
        sys.exit(f"too few train examples: {len(train)}")
    if len(val) < 12:
        sys.exit(f"too few val examples: {len(val)}")
    if len(prefs) < 20:
        sys.exit(f"too few preferences: {len(prefs)}")

    train_notes = check_dialogue(train, system, "train")
    if train_notes and not train_notes[0].startswith("__multiturn__"):
        sys.exit(train_notes[0])
    val_notes = check_dialogue(val, system, "val")
    if val_notes and not val_notes[0].startswith("__multiturn__"):
        sys.exit(val_notes[0])
    train_multiturn = int(train_notes[0].removeprefix("__multiturn__"))
    val_multiturn = int(val_notes[0].removeprefix("__multiturn__"))
    if train_multiturn < 8:
        sys.exit(f"too few multi-turn train rows: {train_multiturn}")
    if val_multiturn < 1:
        sys.exit("val split has no multi-turn row")

    pref_errors = check_preferences(prefs, system)
    if pref_errors:
        sys.exit(pref_errors[0])

    probes = [line.strip() for line in PROBES.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_errors_found = eval_errors(items, probes)
    if eval_errors_found:
        sys.exit(eval_errors_found[0])
    eval_prompts = check_eval(items, probes)

    train_users = {user for row in train for user in user_turns(row)}
    val_users = {user for row in val for user in user_turns(row)}
    overlap = sorted(train_users & val_users)
    if overlap:
        sys.exit("val questions are in the train set: " + " | ".join(overlap))
    leaked = sorted((set(probes) | set(eval_prompts)) & (train_users | val_users))
    if leaked:
        sys.exit("held-out prompts are in train or val: " + " | ".join(leaked))

    pref_train = sum(1 for row in prefs if row["split"] == "train")
    pref_val = sum(1 for row in prefs if row["split"] == "val")
    if pref_train < 10 or pref_val < 3:
        sys.exit(f"preference split is too small: train {pref_train}, val {pref_val}")

    print(
        f"ok: train {len(train)} ({train_multiturn} multi-turn), "
        f"val {len(val)} ({val_multiturn} multi-turn), "
        f"preferences {len(prefs)}, eval {len(items)}"
    )


if __name__ == "__main__":
    main()
