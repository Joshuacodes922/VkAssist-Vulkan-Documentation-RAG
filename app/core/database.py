import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


def initialize_database() -> None:
    # pgvector must be enabled before SQLAlchemy creates vector columns.
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # Import here to ensure the model is registered before create_all().
    from app.models import DocumentChunk, DocumentSection

    Base.metadata.create_all(engine)

    # Lightweight forward migration for development deployments. Production
    # deployments should run this through Alembic.
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_end INTEGER"))
        connection.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS section_id INTEGER"))

    # HNSW accelerates cosine-distance searches as the dataset grows.
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        )
