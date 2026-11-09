#!/usr/bin/env python3
"""Regenerate the auto-maintained sections of README.md.

Run locally (``python scripts/update_readme.py``) or via the
``.github/workflows/update-readme.yml`` action on every push to ``main``.

Four regions are managed, each delimited by HTML comment markers so the
hand-written prose around them is never touched:

    AUTO:STATUS    current phase + roadmap progress, parsed from the checklist
    AUTO:RESULTS   benchmark table, generated from results/metrics.csv
    AUTO:STATS     repository composition (tracked files, code lines)
    AUTO:UPDATED   last-updated footer with the short commit SHA

Exit codes: 0 = README already current, 1 = README rewritten (so CI can
decide whether to commit), 2 = error.
"""

from __future__ import annotations

import contextlib
import csv
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
METRICS = ROOT / "results" / "metrics.csv"

# Columns promoted to the README table, in display order, when present in
# metrics.csv. Anything else in the file is ignored rather than dumped.
PREFERRED_COLUMNS = [
    "model",
    "predictor_source",
    "pr_auc",
    "roc_auc",
    "brier",
    "ece",
    "f1",
    "hss",
    "csi",
]
SORT_CANDIDATES = ["pr_auc", "roc_auc", "f1", "hss"]
MAX_ROWS = 15


def git(*args: str) -> str:
    """Run a git command, returning stdout or an empty string on failure."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def replace_block(text: str, name: str, body: str) -> str:
    """Swap the content between AUTO:<name>:START / :END markers."""
    start, end = f"<!-- AUTO:{name}:START -->", f"<!-- AUTO:{name}:END -->"
    pattern = re.compile(
        re.escape(start) + r".*?" + re.escape(end), re.DOTALL
    )
    if not pattern.search(text):
        print(f"  ! marker AUTO:{name} not found - skipped", file=sys.stderr)
        return text
    return pattern.sub(f"{start}\n{body}\n{end}", text)


def build_status(text: str) -> str:
    """Phase and progress, parsed from the roadmap checklist."""
    done = re.findall(r"^- \[x\] \*\*(\d+)\*\* +[—-] +(.+?)$", text, re.M | re.I)
    todo = re.findall(r"^- \[ \] \*\*(\d+)\*\* +[—-] +(.+?)$", text, re.M)
    total = len(done) + len(todo)
    if total == 0:
        return "_Roadmap not found._"

    if todo:
        num, label = todo[0]
        current = f"**Phase:** {num} — {label.strip()}"
    else:
        current = "**Phase:** complete"

    pct = round(100 * len(done) / total)
    filled = round(pct / 5)
    bar = "█" * filled + "░" * (20 - filled)
    return (
        f"{current} · **Roadmap:** {len(done)} / {total} phases complete "
        f"({pct}%)\n\n`{bar}` {pct}%"
    )


def build_results() -> str:
    """Benchmark table from results/metrics.csv, or an honest placeholder."""
    if not METRICS.exists():
        return (
            "_No results yet — the benchmark is in development. This table is "
            "generated from `results/metrics.csv` by `scripts/update_readme.py` "
            "and will populate automatically once the first experiments run._"
        )

    with METRICS.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        return "_`results/metrics.csv` exists but is empty._"

    columns = [c for c in PREFERRED_COLUMNS if c in rows[0]]
    if not columns:
        columns = list(rows[0].keys())[:8]

    sort_key = next((c for c in SORT_CANDIDATES if c in rows[0]), None)
    if sort_key:
        def key(row: dict) -> float:
            try:
                return -float(row[sort_key])
            except (TypeError, ValueError):
                return 0.0

        rows.sort(key=key)

    def fmt(value: str) -> str:
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value) if value else "—"

    header = "| " + " | ".join(c.replace("_", " ").upper() for c in columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |"
        for r in rows[:MAX_ROWS]
    ]

    note = f"\n\n_Top {min(len(rows), MAX_ROWS)} of {len(rows)} runs"
    if sort_key:
        note += f", ranked by {sort_key.replace('_', ' ').upper()}"
    note += ". Generated from `results/metrics.csv` — never hand-edited._"

    return "\n".join([header, rule, *body]) + note


def build_stats() -> str:
    """Tracked-file composition, counted from git rather than the filesystem."""
    files = [f for f in git("ls-files").splitlines() if f]
    if not files:
        return ""

    code_ext = {".py", ".js", ".css", ".html", ".yml", ".yaml", ".toml", ".sh"}
    code_files, code_lines = 0, 0
    for name in files:
        path = ROOT / name
        if path.suffix.lower() not in code_ext or not path.is_file():
            continue
        code_files += 1
        with contextlib.suppress(OSError):
            code_lines += sum(1 for _ in path.open(encoding="utf-8", errors="ignore"))

    commits = git("rev-list", "--count", "HEAD") or "?"
    return (
        f"<sub>**Repository:** {len(files)} tracked files · {code_files} source "
        f"files · ~{code_lines:,} lines of code · {commits} "
        f"commit{'' if commits == '1' else 's'}</sub>"
    )


def build_updated() -> str:
    """Footer: UTC date plus the commit this README was generated from."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sha = git("rev-parse", "--short", "HEAD")
    suffix = f" from commit [`{sha}`](../../commit/{sha})" if sha else ""
    return (
        f"<sub>Auto-generated sections last updated **{stamp}**{suffix} "
        f"by `scripts/update_readme.py`.</sub>"
    )


def main() -> int:
    if not README.exists():
        print(f"error: {README} not found", file=sys.stderr)
        return 2

    original = README.read_text(encoding="utf-8")
    updated = original
    for name, body in (
        ("STATUS", build_status(original)),
        ("RESULTS", build_results()),
        ("STATS", build_stats()),
        ("UPDATED", build_updated()),
    ):
        updated = replace_block(updated, name, body)

    # Compare ignoring the timestamp line, so a no-op run does not churn the
    # file (and does not create an empty commit on every CI trigger).
    def strip_stamp(text: str) -> str:
        return re.sub(r"last updated \*\*.*?\*\*", "", text)

    if strip_stamp(updated) == strip_stamp(original):
        print("README already current - nothing to do.")
        return 0

    README.write_text(updated, encoding="utf-8")
    print("README.md updated.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
