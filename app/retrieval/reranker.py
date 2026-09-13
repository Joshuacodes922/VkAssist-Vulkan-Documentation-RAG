from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.retrieval.hybrid import HybridResult


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def rerank(
    query: str,
    candidates: list[HybridResult],
    limit: int = 5,
) -> list[HybridResult]:
    if not candidates:
        return []

    pairs: list[tuple[str, str]] = []
    for candidate in candidates:
        pairs.append((query, candidate.chunk.content))

    scores = get_reranker().predict(pairs)

    scored_candidates = []
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.reranker_score = float(score)
        scored_candidates.append(candidate)

    ranked = sorted(
        scored_candidates,
        key=lambda candidate: candidate.reranker_score or float("-inf"),
        reverse=True,
    )

    return ranked[:limit]
