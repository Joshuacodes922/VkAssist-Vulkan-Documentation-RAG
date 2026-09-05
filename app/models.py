from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

EMBEDDING_DIMENSIONS = 768


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Where the chunk came from.
    document_name: Mapped[str] = mapped_column(String(255), index=True)
    page_number: Mapped[int] = mapped_column(Integer, index=True)

    # Position within that page/document.
    chunk_index: Mapped[int] = mapped_column(Integer)

    # Later, populate these using more intelligent parsing.
    section: Mapped[str | None] = mapped_column(String(500), nullable=True)
    api_symbols: Mapped[str | None] = mapped_column(Text, nullable=True)

    content: Mapped[str] = mapped_column(Text)

    embedding: Mapped[list[float]] = mapped_column(
        VECTOR(EMBEDDING_DIMENSIONS)
    )