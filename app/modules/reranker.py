from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.core.config import SETTINGS, get_logger
from app.modules.hybrid_retrieval import RetrievedCandidate
from app.modules.utils import free_memory

_log = get_logger("reranker")


class Reranker:
    """Lazy wrapper around a cross-encoder reranking model."""

    def __init__(self, model_name: str = SETTINGS.reranker_model_name):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def load(self) -> None:
        if self._model is None:
            _log.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device="cpu", max_length=512)

    def unload(self) -> None:
        if self._model is not None:
            free_memory(self._model)
            self._model = None
            _log.info("Reranker model unloaded")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedCandidate],
        top_k: int = SETTINGS.rerank_top_k_simple,
    ) -> list[RetrievedCandidate]:

        if not candidates:
            return []
        self.load()
        assert self._model is not None

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [c for c, _ in scored[:top_k]]


RERANKER = Reranker()
