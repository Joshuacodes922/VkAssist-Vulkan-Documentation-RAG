import os
from functools import lru_cache

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-base-en-v1.5",
    )

    return SentenceTransformer(model_name)


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()

    embeddings = model.encode_document(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()

    embedding = model.encode_query(
        query,
        normalize_embeddings=True,
    )

    return embedding.tolist()