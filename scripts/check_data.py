#!/usr/bin/env python3
"""Check that the jsonl and the inference system prompt still match."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "joker.jsonl"
PROMPT_FILE = ROOT / "data" / "system_prompt.txt"
INFER = ROOT / "infer.py"
PROBES = ROOT / "data" / "probe_prompts.txt"

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


def main() -> None:
    system = PROMPT_FILE.read_text(encoding="utf-8").strip()
    infer_src = INFER.read_text(encoding="utf-8")
    if system not in infer_src:
        sys.exit("infer.py does not contain the system prompt from data/system_prompt.txt")

    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 80:
        sys.exit(f"too few examples: {len(rows)}")

    users: list[str] = []
    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            sys.exit(f"row {index}: expected 3 messages")
        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            sys.exit(f"row {index}: roles {roles}")
        if messages[0]["content"].strip() != system:
            sys.exit(f"row {index}: system prompt does not match")
        user = messages[1]["content"].strip()
        assistant = messages[2]["content"].strip()
        if not user or not assistant:
            sys.exit(f"row {index}: empty text")
        lowered = assistant.lower()
        for phrase in BANNED:
            if phrase in lowered:
                sys.exit(f"row {index}: the Joker breaks character ({phrase})")
        users.append(user)

    dupes = {u for u in users if users.count(u) > 1}
    if dupes:
        sys.exit("repeated questions: " + " | ".join(sorted(dupes)))

    probes = [line.strip() for line in PROBES.read_text(encoding="utf-8").splitlines() if line.strip()]
    leaked = sorted(set(probes) & set(users))
    if leaked:
        sys.exit("probes are in the training set: " + " | ".join(leaked))

    print(f"ok: {len(rows)} examples, {len(probes)} probes outside the train set")


if __name__ == "__main__":
    main()
