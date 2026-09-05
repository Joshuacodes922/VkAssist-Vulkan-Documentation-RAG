import argparse
from dataclasses import dataclass

from sqlalchemy import select

from app.database import SessionLocal, initialize_database
from app.embeddings import embed_query
from app.models import DocumentChunk


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

    return [
        SearchResult(
            content=chunk.content,
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            api_symbols=chunk.api_symbols,
            similarity=1.0 - float(distance_value),
        )
        for chunk, distance_value in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    initialize_database()
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