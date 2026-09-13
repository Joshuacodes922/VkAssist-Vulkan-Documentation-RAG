import logging
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.core.settings import settings
from app.database import SessionLocal, initialize_database
from app.embeddings import get_embedding_model
from app.retrieval.reranker import get_reranker
from app.search import search_hybrid
from app.services.answering import (
    AnsweringNotConfigured,
    InvalidGeneratedAnswer,
    answer_with_citations,
)

app = FastAPI(title="vkassist", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    # The Vite development server runs independently from FastAPI.
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
_metrics = {"requests": 0, "errors": 0, "latency_seconds": 0.0}
logger = logging.getLogger(__name__)
SOURCE_DOCUMENT = Path(__file__).resolve().parents[2] / "data" / "vulkan_documentation.pdf"
PROJECT_README = Path(__file__).resolve().parents[2] / "README.md"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=settings.max_query_characters)
    limit: int = Field(default=settings.default_limit, ge=1, le=settings.max_limit)
    rerank: bool = True
    context_window: int = Field(default=1, ge=0, le=3)
    parent_context: bool = False


def serialize_result(result) -> dict:
    chunk = result.chunk
    return {
        "content": chunk.content,
        "document": chunk.document_name,
        "page_start": chunk.page_number,
        "page_end": chunk.page_end,
        "section": chunk.section,
        "api_symbols": chunk.api_symbols,
        "rrf_score": result.rrf_score,
        "reranker_score": result.reranker_score,
        "context": [item.content for item in result.context_chunks or []],
        "parent_context": result.parent_section.content if result.parent_section else None,
    }


def retrieve(request: SearchRequest):
    return search_hybrid(
        request.query,
        limit=request.limit,
        use_reranker=request.rerank,
        context_window=request.context_window,
        include_parent=request.parent_context,
    )


@app.on_event("startup")
def startup() -> None:
    initialize_database()
    get_embedding_model()
    get_reranker()


@app.get("/health")
def health() -> dict[str, str]:
    with SessionLocal() as session:
        session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/v1/documents/vulkan-documentation")
def source_document() -> FileResponse:
    """Serve the single PDF indexed by this demo, for citation verification."""
    if not SOURCE_DOCUMENT.is_file():
        raise HTTPException(status_code=404, detail="Source document is unavailable")
    return FileResponse(
        SOURCE_DOCUMENT,
        media_type="application/pdf",
        filename="vulkan_documentation.pdf",
        content_disposition_type="inline",
    )


@app.get("/v1/documents/project-readme")
def project_readme() -> FileResponse:
    """Serve the project README for the in-app documentation link."""
    if not PROJECT_README.is_file():
        raise HTTPException(status_code=404, detail="Project documentation is unavailable")
    return FileResponse(
        PROJECT_README,
        media_type="text/plain",
        filename="README.md",
        content_disposition_type="inline",
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return "\n".join(
        [
            f"vkassist_requests_total {_metrics['requests']}",
            f"vkassist_errors_total {_metrics['errors']}",
            f"vkassist_request_latency_seconds_total {_metrics['latency_seconds']:.6f}",
        ]
    ) + "\n"


@app.post("/v1/search")
def search(request: SearchRequest) -> dict:
    started = perf_counter()
    _metrics["requests"] += 1
    try:
        return {"results": [serialize_result(result) for result in retrieve(request)]}
    except Exception as error:
        _metrics["errors"] += 1
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed") from error
    finally:
        _metrics["latency_seconds"] += perf_counter() - started


@app.post("/v1/answers")
async def answer(request: SearchRequest) -> dict:
    started = perf_counter()
    _metrics["requests"] += 1
    try:
        results = retrieve(request)
        generated = await answer_with_citations(request.query, results)
        return {"answer": generated.answer, "citations": generated.citations}
    except AnsweringNotConfigured as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except InvalidGeneratedAnswer as error:
        _metrics["errors"] += 1
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        _metrics["errors"] += 1
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=500, detail="Answer generation failed") from error
    finally:
        _metrics["latency_seconds"] += perf_counter() - started
