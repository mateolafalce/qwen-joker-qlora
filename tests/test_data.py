import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_make():
    spec = importlib.util.spec_from_file_location("make_joker_jsonl", ROOT / "data" / "make_joker_jsonl.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DataTests(unittest.TestCase):
    def test_committed_jsonl_matches_the_generator(self):
        result = subprocess.run(
            [sys.executable, "data/make_joker_jsonl.py", "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_checker_accepts_the_files(self):
        result = subprocess.run(
            [sys.executable, "scripts/check_data.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_splits_and_preference_tags(self):
        make = load_make()
        train_users = set()
        for row in make.train_rows():
            for message in row["messages"]:
                if message["role"] == "user":
                    train_users.add(message["content"])
        val_users = set()
        for row in make.val_rows():
            for message in row["messages"]:
                if message["role"] == "user":
                    val_users.add(message["content"])
        self.assertTrue(len(make.CONVERSATIONS) >= 8)
        self.assertFalse(train_users & val_users)
        tags = {row["tag"] for row in make.preference_rows()}
        self.assertEqual(tags, {"emoji_spam", "assistant_voice", "wrong_fact"})
        eval_prompts = {row["prompt"] for row in make.eval_rows()}
        self.assertFalse(eval_prompts & (train_users | val_users))


if __name__ == "__main__":
    unittest.main()
