from __future__ import annotations

from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.config import SETTINGS, get_logger
from app.modules.utils import free_memory

_log = get_logger("embeddings")


class EmbeddingModel:

    def __init__(self, model_name: str = SETTINGS.embedding_model_name):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        if self._model is None:
            _log.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")

    def unload(self) -> None:
        if self._model is not None:
            free_memory(self._model)
            self._model = None
            _log.info("Embedding model unloaded")

    def encode(
        self,
        texts: list[str],
        batch_size: int = SETTINGS.embedding_batch_size,
        is_query: bool = False,
    ) -> np.ndarray:
        
        self.load()
        assert self._model is not None

        prefix = "query: " if is_query else ""
        inputs = [f"{prefix}{t}" for t in texts] if prefix else texts

        embeddings = self._model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(inputs) > 32,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


# Module-level singleton — import this in downstream sections rather than
# instantiating EmbeddingModel directly, so the model is only ever loaded once.
EMBEDDER = EmbeddingModel()
