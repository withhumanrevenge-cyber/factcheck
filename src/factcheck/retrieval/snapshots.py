"""Stored page snapshots. Spans are verified against these, never against text a model
echoes back (review S5). Keyed by the hash of the extracted text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Snapshot:
    url: str
    final_url: str
    fetched_at: str
    content_hash: str
    text: str
    title: str = ""
    published_at: date | None = None
    date_source: str = "unknown"
    claim_reviews: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False


def content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class SnapshotStore:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        return self.directory / f"{digest.removeprefix('sha256:')}.json"

    def save(self, snap: Snapshot) -> None:
        path = self._path(snap.content_hash)
        if not path.exists():
            path.write_text(json.dumps(asdict(snap), default=str), encoding="utf-8")

    def get(self, digest: str) -> Snapshot | None:
        path = self._path(digest)
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        d["published_at"] = date.fromisoformat(d["published_at"]) if d["published_at"] else None
        return Snapshot(**d)
