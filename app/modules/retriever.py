from __future__ import annotations

from dataclasses import asdict
import pickle
import re

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.core.config import SETTINGS, get_logger
from app.modules.chunker import Chunk
from app.modules.embeddings import EMBEDDER


_log = get_logger("retriever.dense")


class DenseStore:
    """Persistent ChromaDB collection wrapper for chunk-level dense retrieval."""
    def __init__(self, collection_name: str = SETTINGS.chroma_collection_name):
        self.client = chromadb.PersistentClient(
            path=str(SETTINGS.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list["Chunk"]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        ids = [c.metadata.chunk_id for c in chunks]
        vectors = EMBEDDER.encode(texts, is_query=False)

        metadatas = []
        for c in chunks:
            meta = asdict(c.metadata)
            # Chroma requires scalar metadata values -> join list fields to strings.
            meta["keywords"] = ", ".join(meta.get("keywords") or [])
            meta["named_entities"] = ", ".join(meta.get("named_entities") or [])
            metadatas.append({k: (v if v is not None else "") for k, v in meta.items()})

        self.collection.upsert(
            ids=ids, embeddings=vectors.tolist(), documents=texts, metadatas=metadatas
        )
        _log.info("Indexed %d chunks into Chroma collection '%s'", len(chunks), self.collection.name)

    def query(self, query_text: str, top_k: int = SETTINGS.dense_top_k) -> list[dict]:
        query_vec = EMBEDDER.encode([query_text], is_query=True)
        result = self.collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({"chunk_id": chunk_id, "text": text, "metadata": meta, "distance": dist})
        return hits

    def get_by_id(self, chunk_id: str) -> dict | None:
        result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return {
            "chunk_id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0],
        }


DENSE_STORE = DenseStore()

_log = get_logger("retriever.sparse")
_TOKEN_RE = re.compile(r"[\w\u0980-\u09FF]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class SparseStore:
    """Persistent-on-disk BM25 index over chunk text."""

    def __init__(self, path=SETTINGS.bm25_dir / "bm25_index.pkl"):
        self.path = path
        self.chunk_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._chunk_text_by_id: dict[str, str] = {}
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.path.exists():
            with open(self.path, "rb") as f:
                state = pickle.load(f)
            self.chunk_ids = state["chunk_ids"]
            self.corpus_tokens = state["corpus_tokens"]
            self._chunk_text_by_id = state["chunk_text_by_id"]
            self._bm25 = BM25Okapi(self.corpus_tokens)
            _log.info("Loaded BM25 index (%d docs) from %s", len(self.chunk_ids), self.path)

    def add_chunks(self, chunks: list["Chunk"]) -> None:
        for c in chunks:
            self.chunk_ids.append(c.metadata.chunk_id)
            self.corpus_tokens.append(_tokenize(c.text))
            self._chunk_text_by_id[c.metadata.chunk_id] = c.text

        self._bm25 = BM25Okapi(self.corpus_tokens)
        self._persist()
        _log.info("BM25 index now has %d documents", len(self.chunk_ids))

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "corpus_tokens": self.corpus_tokens,
                "chunk_text_by_id": self._chunk_text_by_id,
            }, f)

    def query(self, query_text: str, top_k: int = SETTINGS.sparse_top_k) -> list[dict]:

        if self._bm25 is None or not self.chunk_ids:
            raise RuntimeError("BM25 index is empty — call add_chunks() first.")

        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "chunk_id": self.chunk_ids[i],
                "text": self._chunk_text_by_id[self.chunk_ids[i]],
                "score": float(scores[i]),
            }
            for i in ranked_idx if scores[i] > 0
        ]

SPARSE_STORE = SparseStore()
