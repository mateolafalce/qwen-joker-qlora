import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_conventional_commits",
        ROOT / "scripts" / "check_conventional_commits.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class ConventionalCommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = load_checker()

    def test_merge_commits_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git(repo, "init", "-b", "main")
            git(repo, "config", "user.email", "test@example.com")
            git(repo, "config", "user.name", "test")
            (repo / "base.txt").write_text("base\n")
            git(repo, "add", "base.txt")
            git(repo, "commit", "-m", "feat: add the base")
            base = git(repo, "rev-parse", "HEAD")
            git(repo, "checkout", "-b", "feature")
            (repo / "fix.txt").write_text("fix\n")
            git(repo, "add", "fix.txt")
            git(repo, "commit", "-m", "fix: patch the leak")
            git(repo, "checkout", "main")
            git(
                repo,
                "merge",
                "--no-ff",
                "feature",
                "-m",
                "Merge pull request #4 from mateolafalce/feat/sota-eval-repro",
            )
            subjects = self.checker.commit_subjects(f"{base}..HEAD", cwd=str(repo))
        self.assertEqual([subject for _, subject in subjects], ["fix: patch the leak"])

    def test_normal_commits_are_still_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git(repo, "init", "-b", "main")
            git(repo, "config", "user.email", "test@example.com")
            git(repo, "config", "user.name", "test")
            (repo / "base.txt").write_text("base\n")
            git(repo, "add", "base.txt")
            git(repo, "commit", "-m", "feat: add the base")
            base = git(repo, "rev-parse", "HEAD")
            (repo / "bad.txt").write_text("bad\n")
            git(repo, "add", "bad.txt")
            git(repo, "commit", "-m", "Update the notes")
            result = subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts" / "check_conventional_commits.py"),
                    "--rev-range",
                    f"{base}..HEAD",
                ],
                cwd=repo,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 1)
        self.assertIn("is not a conventional commit", result.stdout + result.stderr)
