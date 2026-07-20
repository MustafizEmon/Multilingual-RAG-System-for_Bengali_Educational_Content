from __future__ import annotations

from dataclasses import dataclass

from app.core.config import SETTINGS, get_logger
from app.modules.query_understanding import QueryExpansion
from app.modules.retriever import DENSE_STORE, SPARSE_STORE

_log = get_logger("hybrid_retrieval")


@dataclass
class RetrievedCandidate:
    """A fused retrieval candidate prior to reranking."""
    chunk_id: str
    text: str
    metadata: dict
    rrf_score: float
    sources: list[str]   # which retrieval legs contributed ("dense", "sparse", "hyde")


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    k: int = SETTINGS.rrf_k,
) -> dict[str, float]:

    fused: dict[str, float] = {}
    for _, ranked_ids in ranked_lists.items():
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


def hybrid_retrieve(
    normalized_query: str,
    expansion: QueryExpansion,
    hyde_passage: str | None,
    top_k: int = SETTINGS.fused_candidate_k,
) -> list[RetrievedCandidate]:

    text_by_id: dict[str, str] = {}
    meta_by_id: dict[str, dict] = {}
    source_ranked_lists: dict[str, list[str]] = {}
    contributing_sources: dict[str, set[str]] = {}

    def _record(label: str, hits: list[dict], id_key: str = "chunk_id"):
        ids = []
        for h in hits:
            cid = h[id_key]
            ids.append(cid)
            text_by_id.setdefault(cid, h["text"])
            if "metadata" in h:
                meta_by_id.setdefault(cid, h["metadata"])
            contributing_sources.setdefault(cid, set()).add(label)
        source_ranked_lists[label] = ids

    dense_hits = DENSE_STORE.query(expansion.expanded_query, top_k=SETTINGS.dense_top_k)
    _record("dense", dense_hits)

    sparse_hits = SPARSE_STORE.query(expansion.expanded_query, top_k=SETTINGS.sparse_top_k)
    _record("sparse", sparse_hits)

    if normalized_query != expansion.expanded_query:
        dense_hits_orig = DENSE_STORE.query(normalized_query, top_k=SETTINGS.dense_top_k)
        _record("dense_original", dense_hits_orig)

    if hyde_passage:
        hyde_hits = DENSE_STORE.query(hyde_passage, top_k=SETTINGS.dense_top_k)
        _record("hyde", hyde_hits)

    fused_scores = reciprocal_rank_fusion(source_ranked_lists)
    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

    candidates = [
        RetrievedCandidate(
            chunk_id=cid,
            text=text_by_id[cid],
            metadata=meta_by_id.get(cid, {}),
            rrf_score=fused_scores[cid],
            sources=sorted(contributing_sources.get(cid, set())),
        )
        for cid in ranked_ids
    ]
    _log.info("Hybrid retrieval fused %d candidates from %d source lists",
              len(candidates), len(source_ranked_lists))
    return candidates
