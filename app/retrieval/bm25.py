import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.database import SessionLocal
from app.models import DocumentChunk


@dataclass
class RankedChunk:
    chunk: DocumentChunk
    score: float
    rank: int


def tokenize(text: str) -> list[str]:
    """
    Preserve Vulkan identifiers such as:
    vkQueueSubmit
    VK_PIPELINE_STAGE_2_TRANSFER_BIT
    VUID-vkQueueSubmit-pWaitSemaphores-03238
    """
    return re.findall(
        r"VUID-[A-Za-z0-9_-]+|VK_[A-Z0-9_]+|"
        r"vk[A-Za-z0-9_]+|Vk[A-Za-z0-9_]+|[A-Za-z0-9]+",
        text,
    )


class BM25Retriever:
    def __init__(self) -> None:
        with SessionLocal() as session:
            self.chunks = list(
                session.scalars(
                    select(DocumentChunk).order_by(DocumentChunk.id)
                )
            )

        tokenized_corpus: list[list[str]] = []
        for chunk in self.chunks:
            tokenized_corpus.append(tokenize(chunk.content))

        self.index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, limit: int = 20) -> list[RankedChunk]:
        scores = self.index.get_scores(tokenize(query))

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:limit]

        ranked_chunks: list[RankedChunk] = []
        for rank, index in enumerate(ranked_indices, start=1):
            if scores[index] <= 0:
                continue

            ranked_chunk = RankedChunk(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=rank,
            )
            ranked_chunks.append(ranked_chunk)

        return ranked_chunks
