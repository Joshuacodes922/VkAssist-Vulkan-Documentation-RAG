from pgvector.sqlalchemy import VECTOR
from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

EMBEDDING_DIMENSIONS = 768


class DocumentSection(Base):
    __tablename__ = "document_sections"
    __table_args__ = (UniqueConstraint("document_name", "section_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_name: Mapped[str] = mapped_column(String(255), index=True)
    section_key: Mapped[str] = mapped_column(String(64), index=True)
    section_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    start_page: Mapped[int] = mapped_column(Integer)
    end_page: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Where the chunk came from.
    document_name: Mapped[str] = mapped_column(String(255), index=True)
    page_number: Mapped[int] = mapped_column(Integer, index=True)
    page_end: Mapped[int] = mapped_column(Integer, default=0)
    section_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_sections.id"), nullable=True, index=True
    )

    # Position within that page/document.
    chunk_index: Mapped[int] = mapped_column(Integer)

    # Later, populate these using more intelligent parsing.
    section: Mapped[str | None] = mapped_column(String(500), nullable=True)
    api_symbols: Mapped[str | None] = mapped_column(Text, nullable=True)

    content: Mapped[str] = mapped_column(Text)

    embedding: Mapped[list[float]] = mapped_column(
        VECTOR(EMBEDDING_DIMENSIONS)
    )
