"""Check that doc and script paths cited in source and user-facing docs exist.

Three spikes in a row (DVC, DIC, digital twin) were worked on a branch that was
never opened as a PR, so the follow-up feature PR shipped citing spec files that
only existed on the unmerged branch -- including from `src/`. This catches that
class before it merges.

Scope is deliberately live references only: `src/`, the top level of `docs/`,
and root `*.md`. Records under `docs/superpowers/` are point-in-time design
documents whose references legitimately age, and policing them is churn.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Paths that are cited but deliberately do not exist in this repo.
ALLOWLIST = {
    # Illustrative placeholders in the "add your own backend" walkthrough.
    "src/fiber_tracer/backends/my_segmentation.py",
    "tests/test_my_segmentation_backend.py",
}

# Repo-root-relative paths, e.g. `docs/superpowers/specs/x.md`, `scripts/y.py`.
ROOT_RELATIVE = re.compile(r"\b(?:docs|scripts|tests|src)/[\w./-]+\.(?:md|py)\b")
# Markdown links, resolved against the containing file's directory.
MARKDOWN_LINK = re.compile(r"\]\(([\w./-]+\.(?:md|py))\)")


def _in_scope(rel: Path) -> bool:
    if rel.suffix not in {".md", ".py"}:
        return False
    parts = rel.parts
    if parts[0] == "src":
        return True
    # docs/ top level only -- not docs/superpowers/ and other subdirectories.
    if parts[0] == "docs":
        return len(parts) == 2
    return len(parts) == 1 and rel.suffix == ".md"


def _cited_paths(text: str, containing_dir: Path):
    """Yield (cited_string, resolved_path) for every path reference in `text`."""
    for match in ROOT_RELATIVE.finditer(text):
        yield match.group(0), ROOT / match.group(0)
    for match in MARKDOWN_LINK.finditer(text):
        cited = match.group(1)
        # Root-relative links are already covered above; skip the duplicate.
        if not ROOT_RELATIVE.fullmatch(cited):
            yield cited, (containing_dir / cited).resolve()


def _tracked_files():
    """Git-tracked files only -- untracked scratch files are not the repo's problem."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    for name in listing.stdout.split("\0"):
        if name and _in_scope(Path(name)):
            yield ROOT / name


def main() -> int:
    missing = []
    for path in sorted(_tracked_files()):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for cited, resolved in _cited_paths(line, path.parent):
                if cited in ALLOWLIST or resolved.exists():
                    continue
                missing.append((path.relative_to(ROOT), lineno, cited))

    if missing:
        print("Cited paths that do not exist:")
        for path, lineno, cited in missing:
            print(f"  {path}:{lineno} -> {cited}")
        print("\nLand the file, fix the reference, or add it to ALLOWLIST with a reason.")
        return 1
    print(f"All cited doc and script paths exist ({len(list(_tracked_files()))} files scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
