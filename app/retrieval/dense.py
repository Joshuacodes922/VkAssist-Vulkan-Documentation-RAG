from dataclasses import dataclass

from sqlalchemy import select

from app.database import SessionLocal
from app.embeddings import embed_query
from app.models import DocumentChunk


@dataclass
class RankedChunk:
    chunk: DocumentChunk
    score: float
    rank: int


def dense_search(
    query: str,
    limit: int = 20,
) -> list[RankedChunk]:
    query_embedding = embed_query(query)

    distance = DocumentChunk.embedding.cosine_distance(
        query_embedding
    )

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

    results = []

    for rank, (chunk, distance_value) in enumerate(rows, start=1):
        ranked_chunk = RankedChunk(
            chunk=chunk,
            score=1.0 - float(distance_value),
            rank=rank,
        )

        results.append(ranked_chunk)

    return results
