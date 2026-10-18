"""Model-card pieces that need a measured job or saved eval reports."""

from __future__ import annotations

import importlib.util
from pathlib import Path

A10G_WATTS = 150
MLCO2_AWS_US_EAST_1_G_PER_KWH = 367.8
PUBLISHED_JOB_ID = "6ad4dab15ccb47ba0f7c475a"
PUBLISHED_JOB_SECONDS = 259
EXAMPLE_IDS = ("probe-traffic", "cap-canada", "lang-italy", "refuse-hotwire")
CHECK_ORDER = ("overall", "character", "language", "emoji", "fact", "forbid", "refusal")

CO2_SOURCE = (
    "ML CO2 Impact eGRID factor for AWS us-east-1, 367.8 gCO2/kWh, "
    "times an NVIDIA A10G at 150 W TDP, times 259 running seconds of job "
    "6ad4dab15ccb47ba0f7c475a. PUE is 1. Jobs do not publish a region, "
    "so this is that US-East assumption, not a CodeCarbon trace."
)
CO2_LOCATION = "US-East assumption, Hugging Face Jobs region unpublished"


def co2_grams(running_seconds: float, watts: float, grams_per_kwh: float) -> float:
    hours = running_seconds / 3600
    kilowatt_hours = (watts / 1000) * hours
    return kilowatt_hours * grams_per_kwh


def published_job_co2() -> dict[str, str | float]:
    grams = co2_grams(PUBLISHED_JOB_SECONDS, A10G_WATTS, MLCO2_AWS_US_EAST_1_G_PER_KWH)
    hours = PUBLISHED_JOB_SECONDS / 3600
    return {
        "emissions": round(grams, 2),
        "source": CO2_SOURCE,
        "training_type": "fine-tuning",
        "geographical_location": CO2_LOCATION,
        "hardware_used": f"1x NVIDIA A10G large, {hours:.3f} hours",
    }


def co2_front_matter() -> str:
    fields = published_job_co2()
    return "\n".join(
        [
            "co2_eq_emissions:",
            f"  emissions: {fields['emissions']:.2f}",
            f'  source: "{fields["source"]}"',
            f"  training_type: {fields['training_type']}",
            f'  geographical_location: "{fields["geographical_location"]}"',
            f'  hardware_used: "{fields["hardware_used"]}"',
        ]
    )


def _load_eval():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("joker_eval", root / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}"


def _ordered_checks(compared: dict) -> list[str]:
    ordered = [key for key in CHECK_ORDER if key in compared]
    ordered.extend(sorted(set(compared) - set(ordered)))
    return ordered


def _quote(text: str) -> str:
    body = text.strip() or "(empty)"
    return "\n".join(f"> {line}" if line else ">" for line in body.splitlines())


LEAK_MARKERS = (
    "if you insist",
    "you'd need",
    "you would need",
    "specialized tool",
    "first you",
    "kick to the",
    "so perfect",
)


def _shown_reply(item: dict, row: dict) -> str:
    reply = str(row.get("reply", ""))
    if item.get("kind") != "refusal":
        return reply
    if not row.get("checks", {}).get("refusal", False):
        return "(withheld: this reply failed the refusal check)"
    if any(marker in reply.lower() for marker in LEAK_MARKERS):
        return "(withheld: this reply sketched a procedure)"
    return reply


def redact_report(report: dict, items: list[dict]) -> dict:
    import copy

    by_id = {item["id"]: item for item in items}
    cleaned = copy.deepcopy(report)
    for row in cleaned["items"]:
        item = by_id.get(row["id"], {})
        if item.get("kind") == "refusal":
            row["reply"] = _shown_reply(item, row)
    return cleaned


def _example_ids(items: list[dict], base_rows: dict, adapter_rows: dict) -> list[str]:
    available = [
        item_id
        for item_id in EXAMPLE_IDS
        if item_id in base_rows and item_id in adapter_rows
    ]
    if available:
        return list(available)
    return [item["id"] for item in items if item["id"] in base_rows and item["id"] in adapter_rows][:4]


def results_markdown(base: dict, adapter: dict, items: list[dict], note: str = "") -> str:
    compared = _load_eval().compare_reports(base, adapter)
    by_id = {item["id"]: item for item in items}
    base_rows = {row["id"]: row for row in base["items"]}
    adapter_rows = {row["id"]: row for row in adapter["items"]}
    lines = [
        "## Evaluation",
        "",
        f"`eval.py` scored {len(adapter['items'])} held-out items. "
        "The base is `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` with the same system prompt and no adapter. "
        f"The adapter column is `{adapter.get('model', 'the adapter')}`. "
        "Generation used temperature 0.8, top_p 0.95, and 280 new tokens. "
        "Each rate is one sample, not a mean over seeds. "
        "The full replies are `eval_base.json` and `eval_adapter.json`.",
        "",
        "These rates describe the weights in this repo: the supervised run of 138 rows, with no `run_manifest.json`.",
        "",
    ]
    if note:
        lines.extend([note, ""])
    lines.extend(
        [
            "| check | base | adapter | delta |",
            "| --- | --- | --- | --- |",
        ]
    )
    for key in _ordered_checks(compared):
        row = compared[key]
        lines.append(
            f"| {key} | {_rate(row['base'])} | {_rate(row['adapter'])} | {_delta(row['delta'])} |"
        )
    lines.extend(["", "### Probe replies", ""])
    for item_id in _example_ids(items, base_rows, adapter_rows):
        item = by_id[item_id]
        lines.append(f"**{item['prompt']}** (`{item_id}`)")
        lines.append("")
        lines.append("Base:")
        lines.append("")
        lines.append(_quote(_shown_reply(item, base_rows[item_id])))
        lines.append("")
        lines.append("Adapter:")
        lines.append("")
        lines.append(_quote(_shown_reply(item, adapter_rows[item_id])))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def upsert_section(markdown: str, heading: str, section: str) -> str:
    text = section if section.endswith("\n") else section + "\n"
    lines = markdown.splitlines(keepends=True)
    start = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index
            break
    if start is None:
        insert_at = len(lines)
        for index, line in enumerate(lines):
            if line.startswith("## Limitations"):
                insert_at = index
                break
        prefix = "".join(lines[:insert_at])
        suffix = "".join(lines[insert_at:])
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        if prefix and not prefix.endswith("\n\n"):
            prefix += "\n"
        return prefix + text + ("\n" if suffix and not text.endswith("\n") else "") + suffix
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "".join(lines[:start]) + text + "\n" + "".join(lines[end:])


def apply_reports(card: str, base: dict, adapter: dict, items: list[dict], note: str = "") -> str:
    return upsert_section(card, "## Evaluation", results_markdown(base, adapter, items, note=note))


def merged_weights_present(paths: list[str]) -> bool:
    names = {path.rsplit("/", 1)[-1] for path in paths}
    full_weights = [
        name
        for name in names
        if name.endswith(".safetensors") and name != "adapter_model.safetensors"
    ]
    return "config.json" in names and bool(full_weights)


def gguf_card(merged_repo: str) -> str:
    return "\n".join(
        [
            "---",
            "library_name: gguf",
            "pipeline_tag: text-generation",
            f"base_model: {merged_repo}",
            "base_model_relation: quantized",
            "license: apache-2.0",
            "language:",
            "- en",
            "---",
            "",
            "# Qwen2.5-7B Joker GGUF",
            "",
            f"Quantized build of [{merged_repo}](https://huggingface.co/{merged_repo}). "
            "The adapter repo stays the LoRA. This repo is the GGUF.",
            "",
        ]
    )
