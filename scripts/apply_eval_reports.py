"""Write the eval delta and a few probe replies into the model card."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_coverage():
    spec = importlib.util.spec_from_file_location("joker_coverage", ROOT / "hub" / "coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    coverage = load_coverage()
    base_path = ROOT / "hub" / "eval_base.json"
    adapter_path = ROOT / "hub" / "eval_adapter.json"
    items_path = ROOT / "data" / "eval_items.jsonl"
    card_path = ROOT / "hub" / "model_card.md"
    missing = [path for path in (base_path, adapter_path, items_path) if not path.is_file()]
    if missing:
        sys.exit("missing " + ", ".join(str(path.relative_to(ROOT)) for path in missing))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    note = (
        "Scored on job `6ad4f4d25ccb47ba0f7c516d`, an A10G small. "
        "The CO2 figure above is the earlier training job, not this scoring run. "
        "Refusal replies that sketched a procedure are withheld here and in the JSON."
    )
    base = coverage.redact_report(base, items)
    adapter = coverage.redact_report(adapter, items)
    base_path.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    adapter_path.write_text(json.dumps(adapter, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    card = card_path.read_text(encoding="utf-8")
    card_path.write_text(coverage.apply_reports(card, base, adapter, items, note=note), encoding="utf-8")
    print(f"updated {card_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
