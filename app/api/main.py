import logging
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
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
_metrics = {"requests": 0, "errors": 0, "latency_seconds": 0.0}
logger = logging.getLogger(__name__)


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
