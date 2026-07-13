#!/usr/bin/env python3
"""Tag and push a gs release from pyproject.toml version."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
REPOSITORY_URL = "https://github.com/MarKo7s/GS.git"
PACKAGE_NAME = "gs"


def read_version(pyproject_path: Path) -> str:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not version:
        raise SystemExit(f"No [project].version in {pyproject_path}")
    return str(version)


def extract_changelog_section(version: str, changelog_path: Path = CHANGELOG) -> str:
    if not changelog_path.is_file():
        raise SystemExit(f"Missing {changelog_path}")

    text = changelog_path.read_text(encoding="utf-8")
    header = re.search(
        rf"^## \[{re.escape(version)}\](?:\s[^\n]*)?\n",
        text,
        re.MULTILINE,
    )
    if not header:
        raise SystemExit(
            f"No CHANGELOG section for version {version}. "
            f"Add a '## [{version}]' heading to {changelog_path.name}."
        )

    start = header.end()
    next_section = re.search(r"^## \[", text[start:], re.MULTILINE)
    end = start + next_section.start() if next_section else len(text)
    section = text[start:end].strip()
    if not section:
        raise SystemExit(f"Empty CHANGELOG section for version {version}")
    return section


def resolve_tag_message(version: str, message: str | None, from_changelog: bool) -> str:
    if message:
        return message
    if from_changelog:
        body = extract_changelog_section(version)
        return f"Release {PACKAGE_NAME} {version}\n\n{body}"
    return f"Release {PACKAGE_NAME} {version}"


def run_git(args: list[str], *, dry_run: bool) -> None:
    cmd = ["git", *args]
    print("+", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_git_tag(tag: str, message: str, *, dry_run: bool) -> None:
    if dry_run:
        print(f"+ git tag -a {tag} -F <message>")
        print(message)
        return

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, suffix=".txt"
    ) as handle:
        handle.write(message)
        message_path = handle.name

    try:
        subprocess.run(
            ["git", "tag", "-a", tag, "-F", message_path],
            cwd=REPO_ROOT,
            check=True,
        )
    finally:
        Path(message_path).unlink(missing_ok=True)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", tag],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def remote_tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def ensure_clean_tree(dry_run: bool) -> None:
    status = git_output("status", "--porcelain", "--untracked-files=no")
    if status:
        raise SystemExit(
            "Working tree is not clean. Commit or stash changes before releasing:\n"
            + status
        )


def ensure_branch(branch: str, dry_run: bool) -> None:
    current = git_output("branch", "--show-current")
    if current != branch:
        raise SystemExit(
            f"On branch '{current}', expected '{branch}'. "
            f"Checkout {branch} or pass --branch {current}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push branch and create/push git tag v{version} from pyproject.toml."
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to push before tagging (default: main)",
    )
    parser.add_argument(
        "--message",
        help="Annotated tag message (overrides --from-changelog)",
    )
    parser.add_argument(
        "--from-changelog",
        action="store_true",
        help=(
            "Use the CHANGELOG.md section for the current version as the tag message "
            f"(prepended with 'Release {PACKAGE_NAME} {{version}}')"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print git commands without executing them",
    )
    args = parser.parse_args()

    if not PYPROJECT.is_file():
        raise SystemExit(f"Missing {PYPROJECT}")

    version = read_version(PYPROJECT)
    tag = f"v{version}"
    message = resolve_tag_message(version, args.message, args.from_changelog)

    ensure_clean_tree(args.dry_run)
    ensure_branch(args.branch, args.dry_run)

    if tag_exists(tag):
        raise SystemExit(f"Tag {tag} already exists locally.")
    if remote_tag_exists(tag):
        raise SystemExit(f"Tag {tag} already exists on origin.")

    run_git(["push", "origin", args.branch], dry_run=args.dry_run)
    run_git_tag(tag, message, dry_run=args.dry_run)
    run_git(["push", "origin", tag], dry_run=args.dry_run)

    print()
    print(f"Released {tag}.")
    print("Install with:")
    print(f'  pip install "{PACKAGE_NAME} @ git+{REPOSITORY_URL}@{tag}"')


if __name__ == "__main__":
    main()
