import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9_]+)(?:\[[^\]]+\])?==([0-9]+\.[0-9]+\.[0-9]+)$")


def pins_in_script(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    start = text.index("# dependencies = [")
    end = text.index("# ]", start)
    found = {}
    for line in text[start:end].splitlines():
        stripped = line.strip().lstrip("#").strip().strip('",')
        match = PIN.fullmatch(stripped)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def pins_in_requirements(path: Path) -> dict[str, str]:
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = PIN.fullmatch(stripped)
        if match is None:
            raise AssertionError(f"unpinned requirement: {stripped}")
        found[match.group(1)] = match.group(2)
    return found


class PinTests(unittest.TestCase):
    def test_train_script_matches_docker_requirements(self):
        script = pins_in_script(ROOT / "train.py")
        requirements = pins_in_requirements(ROOT / "requirements.train.txt")
        self.assertEqual(requirements["hf_transfer"], "0.1.9")
        for name, version in script.items():
            self.assertEqual(requirements[name], version, name)

    def test_infer_and_eval_share_the_runtime_pins(self):
        train = pins_in_script(ROOT / "train.py")
        shared = ("unsloth", "unsloth_zoo", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub")
        for path in (ROOT / "infer.py", ROOT / "eval.py"):
            pins = pins_in_script(path)
            for name in shared:
                self.assertEqual(pins[name], train[name], f"{path.name} {name}")


if __name__ == "__main__":
    unittest.main()
