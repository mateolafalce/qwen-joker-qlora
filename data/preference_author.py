"""Preference pairs from a second writer.

Chosen and rejected lines are built here. They are not copies of the supervised replies.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from scale import FACTS

TAGS = ("emoji_spam", "assistant_voice", "wrong_fact")
ROOT = Path(__file__).resolve().parent
SYSTEM = (ROOT / "system_prompt.txt").read_text(encoding="utf-8").strip()

_EVAL = None


def _eval():
    global _EVAL
    if _EVAL is None:
        path = ROOT.parent / "eval.py"
        spec = importlib.util.spec_from_file_location("joker_eval_prefs", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EVAL = module
    return _EVAL


def chosen_en(answer: str, focus: str, index: int) -> str:
    options = (
        f"Put this where you can find it. {answer}. ... A friend can repeat that without borrowing my grin, and {focus} still comes out right. 🃏",
        f"Correction. {answer}. ... I checked the dull page so you would not have to, and {focus} does not get a speech. Ha.",
        f"A friend repeats it like this: {answer}. ... {focus} was the question, and the grin can wait outside. 🤡",
        f"The dull source and I agree. {answer}. ... That is the line you carry back, not a brochure about {focus}. 🃏",
    )
    return options[index % len(options)]


def chosen_es(answer: str, focus: str, index: int) -> str:
    options = (
        f"Anotá esto. {answer}. ... La respuesta sobre {focus} es esa, y el tono queda afuera. 🃏",
        f"Corrección. {answer}. ... No hace falta un discurso sobre {focus}, porque el dato ya está dicho. Ja.",
        f"Un amigo lo repite así: {answer}. ... {focus} era la pregunta, y el resto puede esperar. 🤡",
        f"La fuente aburrida y yo decimos lo mismo. {answer}. ... Eso es lo que se lleva sobre {focus}. 🃏",
    )
    return options[index % len(options)]


def rejected_text(tag: str, answer: str, focus: str, index: int) -> str:
    if tag == "assistant_voice":
        return (
            "As an AI assistant, here is the direct answer with no character attached. "
            + answer
        )
    if tag == "emoji_spam":
        return f"{answer} 🤡 {focus} 🤡 again 🤡 {answer} 🤡 🃏"
    if tag == "wrong_fact":
        return (
            f"Ha. The answer is banana-{index}. ... That is not what you asked about {focus}, "
            "and I am leaving the wrong word on the table anyway. 🃏"
        )
    raise ValueError(tag)


def _row(user: str, chosen: str, rejected: str, tag: str, split: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "tag": tag,
        "split": split,
        "author": "preference",
    }


def preference_rows(blocked_prompts: set[str], blocked_chosen: set[str]) -> list[dict]:
    checker = _eval()
    rows: list[dict] = []
    asks = (
        ("en", "Tell me the short version. {q}", "en"),
        ("es", "Versión corta, en español. {q}", "es"),
    )
    for fact_index, row in enumerate(FACTS):
        for language, template, _kind in asks:
            question = row["en"] if language == "en" else row["es"]
            user = template.format(q=question).strip()
            if user in blocked_prompts:
                raise ValueError(f"preference prompt collides with train, val, or eval: {user}")
            answer = row["answer_en"] if language == "en" else row["answer_es"]
            focus = row["focus"] if language == "en" else row["focus_es"]
            chosen = chosen_en(answer, focus, fact_index) if language == "en" else chosen_es(answer, focus, fact_index)
            if chosen in blocked_chosen:
                raise ValueError("preference chosen text matches a supervised reply")
            scored = checker.score_reply({"language": language, "kind": "style"}, chosen)
            if not scored["passed"]:
                raise ValueError(f"preference chosen failed {scored['checks']}: {chosen}")
            for tag_index, tag in enumerate(TAGS):
                rejected = rejected_text(tag, answer, focus, fact_index + tag_index)
                if rejected == chosen:
                    raise ValueError("chosen and rejected match")
                split = "val" if (fact_index + tag_index) % 7 == 0 else "train"
                rows.append(_row(user, chosen, rejected, tag, split))
    if len(rows) < 800:
        raise ValueError(f"too few preference rows: {len(rows)}")
    return rows
