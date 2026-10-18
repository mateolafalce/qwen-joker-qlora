import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_train():
    spec = importlib.util.spec_from_file_location("joker_train", ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_train()

    def test_sha256_is_stable(self):
        self.assertEqual(
            self.train.sha256_text("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_manifest_records_the_run(self):
        manifest = self.train.build_manifest(
            git_sha="abc123",
            base_model="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            train_sha256="1",
            val_sha256="2",
            preferences_sha256="3",
            loaded_train_sha256="4",
            loaded_val_sha256="5",
            loaded_preferences_sha256="6",
            hyperparameters={"num_epochs": 4, "seed": 3407},
            sft={"train_loss": 0.5, "best_eval_loss": 0.4},
            dpo={"train_loss": 0.2},
        )
        self.assertEqual(manifest["git_sha"], "abc123")
        self.assertEqual(manifest["hyperparameters"]["seed"], 3407)
        self.assertEqual(manifest["sft"]["best_eval_loss"], 0.4)
        self.assertEqual(manifest["dpo"]["train_loss"], 0.2)
        self.assertIn("loaded_train_sha256", manifest)

    def test_retries_then_returns(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("hub down")
            return "pushed"

        self.assertEqual(self.train.call_with_retries(flaky, attempts=3, pause_seconds=0), "pushed")
        self.assertEqual(calls["n"], 3)

    def test_copy_adapter_copies_lora_weights(self):
        class Tensor:
            def __init__(self, value):
                self.value = value

            def copy_(self, other):
                self.value = other.value
                return self

        class Weight:
            def __init__(self, value):
                self.data = Tensor(value)

        class Linear:
            def __init__(self, value):
                self.weight = Weight(value)

        class Block:
            def __init__(self):
                self.lora_A = {"default": Linear(5), "dpo": Linear(0)}
                self.lora_B = {"default": Linear(7), "dpo": Linear(0)}

        class Model:
            def __init__(self):
                self.block = Block()

            def modules(self):
                yield self
                yield self.block

        model = Model()
        self.train.copy_adapter(model, "default", "dpo")
        self.assertEqual(model.block.lora_A["dpo"].weight.data.value, 5)
        self.assertEqual(model.block.lora_B["dpo"].weight.data.value, 7)

    def test_retries_raise_after_the_last_failure(self):
        def always_fail():
            raise RuntimeError("still down")

        with self.assertRaises(RuntimeError):
            self.train.call_with_retries(always_fail, attempts=2, pause_seconds=0)


if __name__ == "__main__":
    unittest.main()
