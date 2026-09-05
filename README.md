# vkassist

`vkassist` is a local retrieval-augmented generation (RAG) foundation for searching Vulkan documentation. It extracts text from a PDF, splits it into overlapping chunks, turns those chunks into embeddings, and stores them in PostgreSQL with pgvector for semantic search.

The current command-line interface performs retrieval only: it returns the most relevant source passages and their page numbers. A language-model response layer can be added on top later.

## How it works

```text
Vulkan PDF -> text chunks -> embeddings -> PostgreSQL + pgvector
User question -> query embedding -> cosine similarity search -> source passages
```

- **PDF extraction and chunking:** PyMuPDF extracts the document; chunks are roughly 1,200 characters with a 200-character overlap.
- **Embeddings:** `BAAI/bge-base-en-v1.5` creates normalized, 768-dimensional embeddings locally through Sentence Transformers.
- **Vector search:** PostgreSQL with the pgvector extension stores embeddings and uses an HNSW index for cosine-distance search.

## Requirements

- Python 3.11 or newer
- Docker Desktop (or a PostgreSQL instance with the pgvector extension)
- A Vulkan PDF to ingest

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Create your local configuration file:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Start PostgreSQL with pgvector:

   ```powershell
   docker compose up -d
   ```

4. Put a PDF in `data/`, then ingest it. The first run downloads the embedding model; later runs use the local Hugging Face cache.

   ```powershell
   python -m app.ingest data\vulkan_documentation.pdf --replace
   ```

5. Search the indexed document:

   ```powershell
   python -m app.search "What does vkCmdPipelineBarrier do?"
   ```

   Use `--limit` to change the number of results:

   ```powershell
   python -m app.search "What does a VkBuffer do?" --limit 3
   ```

## Configuration

`DATABASE_URL` is read from `.env`. The default value in `.env.example` matches the included Docker Compose service. Keep `.env` local; it is intentionally not committed.

## Project structure

```text
app/
  chunker.py       PDF parsing and structure-aware chunking
  embeddings.py    Embedding-model loading and vector generation
  ingest.py        PDF ingestion command
  search.py        Semantic-search command
  database.py      PostgreSQL and pgvector initialization
  models.py        SQLAlchemy document-chunk model
compose.yaml       Local PostgreSQL + pgvector service
```

## Notes

- Re-running ingestion without `--replace` adds another copy of the document's chunks. Use `--replace` when re-ingesting the same PDF.
- Each `python -m app.search` command starts a new process, so the embedding model is loaded into memory for that command. Model files remain cached locally after their initial download.
