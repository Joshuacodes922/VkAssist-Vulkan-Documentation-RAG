"""Offline retrieval evaluation for multi-document corpora.

Judgments are page-level and document-aware because page numbers are the stable source citation
available in the corpus.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    query: str
    relevant_pages: frozenset[int]
    document: str | None = None


def load_cases(path: Path) -> list[EvaluationCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        EvaluationCase(
            id=raw["id"],
            query=raw["query"],
            relevant_pages=frozenset(raw["relevant_pages"]),
            document=raw.get("document"),
        )
        for raw in raw_cases
    ]
    if not cases:
        raise ValueError("Evaluation set must contain at least one case")
    return cases


def is_relevant(result, relevant_pages: frozenset[int], document: str | None = None) -> bool:
    if document and result.chunk.document_name != document:
        return False
    return bool(set(range(result.chunk.page_number, result.chunk.page_end + 1)) & relevant_pages)


def metrics_for_ranking(
    results,
    relevant_pages: frozenset[int],
    k: int,
    document: str | None = None,
) -> dict[str, float]:
    covered_pages: set[int] = set()
    relevant_ranks: list[int] = []
    for rank, result in enumerate(results[:k], start=1):
        if document and result.chunk.document_name != document:
            continue
        result_pages = set(
            range(result.chunk.page_number, result.chunk.page_end + 1)
        )
        new_pages = (result_pages & relevant_pages) - covered_pages
        if new_pages:
            relevant_ranks.append(rank)
            covered_pages.update(new_pages)

    recall = len(covered_pages) / len(relevant_pages) if relevant_pages else 0.0
    reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_count = min(len(relevant_pages), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def evaluate(
    cases: list[EvaluationCase],
    limit: int,
    candidate_limit: int,
    use_reranker: bool,
    filter_by_case_document: bool = False,
) -> dict:
    # Import lazily so the metric functions can be tested without a database.
    from app.database import initialize_database
    from app.search import search_hybrid

    initialize_database()
    rows = []
    for case in cases:
        doc_filter = case.document if filter_by_case_document else None
        started = perf_counter()
        results = search_hybrid(
            case.query,
            limit=limit,
            candidate_limit=candidate_limit,
            use_reranker=use_reranker,
            context_window=0,
            include_parent=False,
            document_name=doc_filter,
        )
        latency_ms = (perf_counter() - started) * 1000
        metrics = metrics_for_ranking(
            results,
            case.relevant_pages,
            limit,
            document=case.document,
        )
        rows.append(
            {
                "id": case.id,
                "query": case.query,
                "document": case.document,
                "relevant_pages": sorted(case.relevant_pages),
                "retrieved_pages": [result.chunk.page_number for result in results],
                "retrieved_documents": [result.chunk.document_name for result in results],
                "latency_ms": round(latency_ms, 2),
                **metrics,
            }
        )

    count = len(rows)
    latencies = sorted(row["latency_ms"] for row in rows)
    percentile_index = max(0, math.ceil(0.95 * count) - 1)
    return {
        "configuration": {
            "limit": limit,
            "candidate_limit": candidate_limit,
            "reranker": use_reranker,
            "filter_by_case_document": filter_by_case_document,
        },
        "summary": {
            f"recall_at_{limit}": round(sum(row["recall"] for row in rows) / count, 4),
            f"mrr_at_{limit}": round(sum(row["reciprocal_rank"] for row in rows) / count, 4),
            f"ndcg_at_{limit}": round(sum(row["ndcg"] for row in rows) / count, 4),
            "latency_ms_p50": round(latencies[(count - 1) // 2], 2),
            "latency_ms_p95": round(latencies[percentile_index], 2),
            "total_cases": count,
        },
        "cases": rows,
    }
