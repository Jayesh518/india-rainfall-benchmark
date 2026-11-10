"""Provenance manifest: what was fetched, when, from where, and its checksum.

Phase-1 exit gate: every input has a recorded source, licence and md5, and
anything that could not be verified says so. The manifest is plain JSON so it
can be committed (it holds no data, only checksums).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from indiarain.config import ROOT

MANIFEST: Path = ROOT / "data" / "manifest.json"


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    """MD5 of a file, streamed so large grids do not load into memory."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def record(
    dataset_id: str,
    *,
    source: str,
    licence: str,
    files: list[Path],
    verified: bool = True,
    note: str = "",
    extra: dict[str, Any] | None = None,
    path: Path = MANIFEST,
) -> dict[str, Any]:
    """Write/replace the manifest entry for ``dataset_id`` and return it."""
    entry: dict[str, Any] = {
        "fetched_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "licence": licence,
        "verified": verified,
        "note": note,
        "files": [
            {
                "path": f.resolve().relative_to(ROOT).as_posix(),
                "bytes": f.stat().st_size,
                "md5": md5_file(f),
            }
            for f in sorted(files)
        ],
        **(extra or {}),
    }
    manifest = load_manifest(path)
    manifest[dataset_id] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entry
