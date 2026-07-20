from __future__ import annotations
from dataclasses import dataclass

from app.core.config import SETTINGS, get_logger
from app.modules.hybrid_retrieval import RetrievedCandidate
from app.modules.retriever import DENSE_STORE

_log = get_logger("parent_context")

@dataclass
class ExpandedContext:
    """A reranked chunk plus its bounded neighboring context."""
    core: RetrievedCandidate
    neighbors: list[dict]   # [{"chunk_id", "text", "relation": "prev"|"next"}]


def expand_with_parent_context(
    candidates: list[RetrievedCandidate],
    max_neighbors: int = SETTINGS.max_neighbors_simple,
) -> list[ExpandedContext]:

    expanded: list[ExpandedContext] = []

    for cand in candidates:
        neighbors: list[dict] = []
        if max_neighbors > 0:
            prev_id = cand.metadata.get("prev_chunk_id")
            next_id = cand.metadata.get("next_chunk_id")

            neighbor_ids = [(prev_id, "prev"), (next_id, "next")][:max_neighbors]
            for neighbor_id, relation in neighbor_ids:
                if not neighbor_id:
                    continue
                fetched = DENSE_STORE.get_by_id(neighbor_id)
                if fetched:
                    neighbors.append({
                        "chunk_id": fetched["chunk_id"],
                        "text": fetched["text"],
                        "relation": relation,
                    })

        expanded.append(ExpandedContext(core=cand, neighbors=neighbors))

    _log.info("Expanded %d candidates with parent/neighbor context (avg %.1f neighbors)",
              len(expanded),
              sum(len(e.neighbors) for e in expanded) / max(1, len(expanded)))
    return expanded
