# vkassist

`vkassist` is a production-minded RAG backend for Vulkan documentation. It ingests technical PDFs, performs hybrid retrieval, reranks evidence, expands relevant context, and can generate grounded answers with source citations through a local or hosted OpenAI-compatible model.

## Architecture

```text
PDF
  → TOC-aware layout extraction
  → parent sections + bounded child chunks
  → BGE embeddings + PostgreSQL/pgvector

Query
  → dense cosine retrieval + BM25 lexical retrieval
  → reciprocal-rank fusion
  → cross-encoder reranking
  → parent sliding-window or full-parent context
  → optional cited answer generation
```

### Retrieval design

- **Structure-aware ingestion:** PyMuPDF layout blocks, TOC hierarchy, repeated-header/footer filtering, printed-heading detection, page ranges, and Vulkan API-symbol metadata.
- **Parent-child indexing:** section parents are stored separately from child chunks. Retrieval ranks children precisely, then can return nearby siblings or the full parent section.
- **Hybrid retrieval:** normalized `BAAI/bge-base-en-v1.5` embeddings (768 dimensions) use pgvector cosine search alongside BM25 lexical retrieval; reciprocal-rank fusion combines both candidate lists.
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L6-v2` scores query/chunk pairs after fusion.
- **Grounded answers:** an optional local Ollama or hosted OpenAI-compatible endpoint receives only retrieved passages and is instructed to cite them.

## Project layout

```text
app/
  api/          FastAPI endpoints, health, and metrics
  cli/          Search and benchmark commands
  core/         Settings, PostgreSQL engine, and ORM models
  evaluations/  Retrieval metrics and report generation
  indexing/     Embedding-text construction
  ingestion/    PDF parsing and document ingestion
  retrieval/    Dense, BM25, fusion, and reranking stages
  services/     Grounded answer-generation service
  tests/        Unit tests
data/evaluation/  Versioned relevance judgments and reports
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
docker compose up -d postgres
python -m app.ingest data\vulkan_documentation.pdf --replace
```

Run hybrid retrieval and cross-encoder reranking:

```powershell
python -m app.search "What does vkCmdPipelineBarrier do?" --rerank --limit 5
```

Use adjacent child context (the default is one chunk on either side):

```powershell
python -m app.search "Explain the graphics pipeline" --rerank --context-window 2
```

Return a child's complete parent section when broad context is required:

```powershell
python -m app.search "Explain swap chain recreation" --rerank --parent-context --context-window 0
```

## API

Start the HTTP service locally:

```powershell
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

### Test from Postman, Talend API Tester, or Insomnia

No interactive `/docs` page is required. Create a request with these settings:

| Field | Value |
| --- | --- |
| Method | `POST` |
| URL | `http://127.0.0.1:8000/v1/search` |
| Header | `Content-Type: application/json` |
| Body | Raw JSON |

Use this JSON body to inspect retrieved passages:

```json
{
  "query": "What does vkCmdPipelineBarrier do?",
  "limit": 3,
  "rerank": true,
  "context_window": 1,
  "parent_context": false
}
```

The service also exposes `GET /health`, `GET /metrics`, `POST /v1/answers`, `GET /v1/documents/vulkan-documentation`, and `GET /v1/documents/project-readme`.

### Local Ollama answers

The answer endpoint uses an OpenAI-compatible chat-completions interface. With Ollama running locally:

```env
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:4b
LLM_API_KEY=ollama
LLM_TIMEOUT_SECONDS=120
LLM_MAX_TOKENS=600
```

Create another `POST` request in your API client:

| Field | Value |
| --- | --- |
| URL | `http://127.0.0.1:8000/v1/answers` |
| Header | `Content-Type: application/json` |
| Body | Raw JSON |

```json
{
  "query": "What is a swapchain?",
  "limit": 5,
  "rerank": true,
  "context_window": 1,
  "parent_context": false
}
```

The response contains an `answer` string and a `citations` array with the document name, section, and source page for every citation used.

For Docker deployment, run `docker compose up --build`. Use `host.docker.internal` instead of `localhost` in `LLM_BASE_URL` when Ollama runs on the Windows host.

## Evaluation

The versioned benchmark has 36 page-labeled Vulkan queries. It records Recall@k, MRR@k, nDCG@k, per-query retrieved pages, and P50/P95 latency.

```powershell
python -m app.evaluate --output data\evaluation\reports\hybrid-reranked.json
python -m app.evaluate --no-rerank --output data\evaluation\reports\hybrid-only.json
```

Treat these starter labels as a maintained benchmark: review labels when the source document or chunking strategy changes, add difficult/negative queries, and compare reports before claiming a retrieval improvement.

## Verification

```powershell
python -m unittest app.tests.test_chunker app.tests.test_evaluation
python -m compileall -q app
```

GitHub Actions runs these checks on pushes and pull requests.
