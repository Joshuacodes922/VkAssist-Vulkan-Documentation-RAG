from dataclasses import dataclass

from sqlalchemy import select

from app.database import SessionLocal
from app.models import DocumentChunk, DocumentSection
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import dense_search


@dataclass
class HybridResult:
    chunk: DocumentChunk
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    reranker_score: float | None = None
    context_chunks: list[DocumentChunk] | None = None
    parent_section: DocumentSection | None = None


def add_parent_context(
    results: list[HybridResult],
    window_size: int = 1,
    include_parent: bool = False,
) -> list[HybridResult]:
    """Attach adjacent children from the same parent after ranking."""
    if window_size < 0:
        raise ValueError("window_size cannot be negative")

    with SessionLocal() as session:
        for result in results:
            chunk = result.chunk
            statement = (
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_name == chunk.document_name,
                    DocumentChunk.chunk_index.between(
                        chunk.chunk_index - window_size,
                        chunk.chunk_index + window_size,
                    ),
                )
                .order_by(DocumentChunk.chunk_index)
            )
            if chunk.section_id is not None:
                statement = statement.where(
                    DocumentChunk.section_id == chunk.section_id)
            result.context_chunks = list(session.scalars(statement))
        if include_parent:
            parent_ids = {
                result.chunk.section_id
                for result in results
                if result.chunk.section_id is not None
            }
            parents = {
                parent.id: parent
                for parent in session.scalars(
                    select(DocumentSection).where(DocumentSection.id.in_(parent_ids))
                )
            }
            for result in results:
                result.parent_section = parents.get(result.chunk.section_id)
    return results


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    rrf_constant: int = 60,
) -> list[HybridResult]:
    combined: dict[int, dict] = {}

    for result in dense_results:
        chunk_id = result.chunk.id

        combined.setdefault(
            chunk_id,
            {
                "chunk": result.chunk,
                "score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
            },
        )

        combined[chunk_id]["score"] += (
            1.0 / (rrf_constant + result.rank)
        )
        combined[chunk_id]["dense_rank"] = result.rank

    for result in bm25_results:
        chunk_id = result.chunk.id

        combined.setdefault(
            chunk_id,
            {
                "chunk": result.chunk,
                "score": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
            },
        )

        combined[chunk_id]["score"] += (
            1.0 / (rrf_constant + result.rank)
        )
        combined[chunk_id]["bm25_rank"] = result.rank

    fused_results = []

    for value in combined.values():
        result = HybridResult(
            chunk=value["chunk"],
            rrf_score=value["score"],
            dense_rank=value["dense_rank"],
            bm25_rank=value["bm25_rank"],
        )

        fused_results.append(result)



    return sorted(
        fused_results,
        key=lambda result: result.rrf_score,
        reverse=True,
    )


class HybridRetriever:
    def __init__(self) -> None:
        self.bm25 = BM25Retriever()

    def search(
        self,
        query: str,
        candidate_limit: int = 20,
        result_limit: int = 10,
    ) -> list[HybridResult]:
        dense_results = dense_search(
            query,
            limit=candidate_limit,
        )

        bm25_results = self.bm25.search(
            query,
            limit=candidate_limit,
        )

        fused = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
        )

        return fused[:result_limit]
