#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys

TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
)
HEADER_MAX = 100
HEADER = re.compile(
    rf"^({'|'.join(TYPES)})(\([a-z0-9][a-z0-9._/-]*\))?(!)?: (.+)$"
)


def check_header(header: str, label: str) -> list[str]:
    errors: list[str] = []
    if len(header) > HEADER_MAX:
        errors.append(f"{label}: header is {len(header)} characters (limit {HEADER_MAX})")
    match = HEADER.fullmatch(header)
    if match is None:
        errors.append(f"{label}: {header!r} is not a conventional commit")
        return errors
    subject = match.group(4)
    if subject.endswith("."):
        errors.append(f"{label}: subject must not end with a period")
    if not subject[0].islower():
        errors.append(f"{label}: subject must start with a lowercase letter")
    return errors


def commit_subjects(rev_range: str) -> list[tuple[str, str]]:
    result = subprocess.run(
        ["git", "log", "--format=%h%x09%s", rev_range],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        sha, subject = line.split("\t", 1)
        rows.append((sha, subject))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rev-range", default="HEAD")
    parser.add_argument("--title")
    args = parser.parse_args()

    errors: list[str] = []
    subjects = commit_subjects(args.rev_range)
    if args.title is not None:
        errors.extend(check_header(args.title.strip(), "pull request title"))
    for sha, subject in subjects:
        errors.extend(check_header(subject, sha))
    if errors:
        sys.exit("\n".join(errors))

    title = " and the pull request title" if args.title is not None else ""
    print(f"ok: {len(subjects)} commits{title}")


if __name__ == "__main__":
    main()
