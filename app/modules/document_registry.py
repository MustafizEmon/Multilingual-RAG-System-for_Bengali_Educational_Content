from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

from app.core.config import SETTINGS, get_logger

_log = get_logger("document_registry")


@dataclass
class DocumentRecord:
    """Bookkeeping metadata for one ingested document."""
    document_name: str
    subject: str
    n_pages: int
    n_chunks: int


class DocumentRegistry:

    def __init__(self, path: Path = SETTINGS.chunk_store_dir / "documents_registry.json"):
        self.path = path
        self._lock = Lock()
        self._records: dict[str, DocumentRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._records = {name: DocumentRecord(**rec) for name, rec in raw.items()}
                _log.info("Loaded document registry (%d documents) from %s",
                           len(self._records), self.path)
            except Exception as exc:
                _log.warning("Failed to load document registry, starting fresh: %s", exc)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {name: asdict(rec) for name, rec in self._records.items()}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def register(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.document_name] = record
        self._persist()

    def list_all(self) -> list[DocumentRecord]:
        with self._lock:
            return list(self._records.values())

    def get(self, document_name: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(document_name)


DOCUMENT_REGISTRY = DocumentRegistry()
