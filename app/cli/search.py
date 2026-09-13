import argparse
from dataclasses import dataclass

from sqlalchemy import select

from app.database import SessionLocal, initialize_database
from app.embeddings import embed_query
from app.models import DocumentChunk
from app.retrieval.hybrid import HybridResult, HybridRetriever, add_parent_context
from app.retrieval.reranker import rerank


@dataclass
class SearchResult:
    content: str
    document_name: str
    page_number: int
    api_symbols: str | None
    similarity: float


def search_chunks(query: str, limit: int = 5) -> list[SearchResult]:
    query_embedding = embed_query(query)

    # pgvector cosine_distance returns 0 for identical vectors.
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)

    statement = (
        select(
            DocumentChunk,
            distance.label("distance"),
        )
        .order_by(distance)
        .limit(limit)
    )

    with SessionLocal() as session:
        rows = session.execute(statement).all()

    results: list[SearchResult] = []
    for chunk, distance_value in rows:
        result = SearchResult(
            content=chunk.content,
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            api_symbols=chunk.api_symbols,
            similarity=1.0 - float(distance_value),
        )
        results.append(result)

    return results


def search_hybrid(
    query: str,
    limit: int = 5,
    candidate_limit: int = 25,
    use_reranker: bool = False,
    context_window: int = 1,
    include_parent: bool = False,
) -> list[HybridResult]:
    """Retrieve with dense search + BM25 reciprocal-rank fusion."""
    candidates = HybridRetriever().search(
        query,
        candidate_limit=candidate_limit,
        result_limit=candidate_limit,
    )

    if use_reranker:
        results = rerank(query, candidates, limit=limit)
    else:
        results = candidates[:limit]

    return add_parent_context(
        results,
        window_size=context_window,
        include_parent=include_parent,
    )


def print_hybrid_results(results: list[HybridResult]) -> None:
    for position, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"RESULT {position}")
        print(f"RRF score: {result.rrf_score:.6f}")
        print(
            f"Source: {result.chunk.document_name}, "
            f"pages {result.chunk.page_number}-{result.chunk.page_end}"
        )
        print(
            f"Dense rank: {result.dense_rank} | "
            f"BM25 rank: {result.bm25_rank}"
        )

        if result.reranker_score is not None:
            print(f"Cross-encoder score: {result.reranker_score:.4f}")

        if result.chunk.api_symbols:
            print(f"Symbols: {result.chunk.api_symbols}")

        print()
        print(result.chunk.content)

        if result.context_chunks and len(result.context_chunks) > 1:
            print("\n--- Parent sliding-window context ---")
            for context_chunk in result.context_chunks:
                if context_chunk.id == result.chunk.id:
                    continue
                print(f"\nContext: pages {context_chunk.page_number}-{context_chunk.page_end}")
                print(context_chunk.content)

        if result.parent_section:
            parent = result.parent_section
            print("\n--- Retrieved parent section ---")
            print(
                f"Section: {parent.section_path or 'Front matter'} | "
                f"pages {parent.start_page}-{parent.end_page}"
            )
            print(parent.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Combine dense vector search and BM25 with reciprocal-rank fusion.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply the cross-encoder to hybrid-search candidates.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=25,
        help="Number of dense and BM25 candidates for hybrid search (default: 25).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=1,
        help="Adjacent child chunks from the same parent section (default: 1).",
    )
    parser.add_argument(
        "--parent-context",
        action="store_true",
        help="Return the full parent section for each retrieved child chunk.",
    )
    args = parser.parse_args()

    initialize_database()

    if args.rerank:
        args.hybrid = True

    if args.hybrid:
        results = search_hybrid(
            args.query,
            limit=args.limit,
            candidate_limit=max(args.candidate_limit, args.limit),
            use_reranker=args.rerank,
            context_window=args.context_window,
            include_parent=args.parent_context,
        )
        print_hybrid_results(results)
        return

    results = search_chunks(args.query, args.limit)

    for position, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"RESULT {position}")
        print(f"Similarity: {result.similarity:.4f}")
        print(
            f"Source: {result.document_name}, "
            f"page {result.page_number}"
        )

        if result.api_symbols:
            print(f"Symbols: {result.api_symbols}")

        print()
        print(result.content)


if __name__ == "__main__":
    main()
