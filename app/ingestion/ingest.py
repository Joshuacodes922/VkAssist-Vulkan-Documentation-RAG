import argparse
from pathlib import Path

from sqlalchemy import delete

from app.chunker import parse_document
from app.database import SessionLocal, initialize_database
from app.embeddings import embed_documents
from app.indexing.embedding_text import build_embedding_text
from app.models import DocumentChunk, DocumentSection


def collect_pdf_paths(paths: list[Path | str]) -> list[Path]:
    """Collect and deduplicate PDF files from paths, directories, and glob patterns."""
    pdf_files: list[Path] = []
    seen: set[Path] = set()

    for p in paths:
        target = Path(p).resolve()
        if target.is_dir():
            found = sorted(target.rglob("*.pdf"))
            for file_path in found:
                if file_path not in seen and file_path.is_file():
                    seen.add(file_path)
                    pdf_files.append(file_path)
        elif target.is_file() and target.suffix.lower() == ".pdf":
            if target not in seen:
                seen.add(target)
                pdf_files.append(target)
        else:
            # Check for glob pattern on parent
            parent = target.parent if target.parent.exists() else Path.cwd()
            pattern = target.name
            for file_path in sorted(parent.glob(pattern)):
                if file_path.suffix.lower() == ".pdf" and file_path not in seen and file_path.is_file():
                    seen.add(file_path)
                    pdf_files.append(file_path)

    return pdf_files


def ingest_pdf(pdf_path: Path, replace_existing: bool = False) -> int:
    """Ingest a single PDF file into PostgreSQL with dense embeddings and parent sections."""
    initialize_database()

    print(f"Parsing {pdf_path.name} ({pdf_path})...")
    parsed = parse_document(pdf_path)
    chunks = parsed.chunks

    if not chunks:
        print(f"Warning: No text chunks were extracted from {pdf_path.name}")
        return 0

    print(f"Extracted {len(chunks)} chunks across {len(parsed.sections)} sections")
    print(f"Generating embeddings for {pdf_path.name}...")

    embedding_texts: list[str] = []
    for chunk in chunks:
        embedding_text = build_embedding_text(
            document_name=chunk.document_name,
            section=chunk.section,
            page_number=chunk.page_number,
            content=chunk.content,
            api_symbols=chunk.api_symbols,
        )
        embedding_texts.append(embedding_text)

    embeddings = embed_documents(embedding_texts)

    records: list[DocumentChunk] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        record = DocumentChunk(
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            page_end=chunk.page_end,
            chunk_index=chunk.chunk_index,
            section=chunk.section,
            content=chunk.content,
            api_symbols=", ".join(chunk.api_symbols) or None,
            embedding=embedding,
        )
        records.append(record)

    with SessionLocal.begin() as session:
        if replace_existing:
            session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_name == pdf_path.name
                )
            )
            session.execute(
                delete(DocumentSection).where(
                    DocumentSection.document_name == pdf_path.name
                )
            )

        sections = [
            DocumentSection(
                document_name=section.document_name,
                section_key=section.key,
                section_path=section.path,
                start_page=section.start_page,
                end_page=section.end_page,
                content=section.content,
            )
            for section in parsed.sections
        ]
        session.add_all(sections)
        session.flush()
        section_ids = {section.section_key: section.id for section in sections}
        for record, chunk in zip(records, chunks, strict=True):
            record.section_id = section_ids.get(chunk.parent_key)

        session.add_all(records)

    print(f"Stored {len(records)} chunks for {pdf_path.name}\n")
    return len(records)


def ingest_documents(
    paths: list[Path | str],
    replace_existing: bool = False,
    replace_all: bool = False,
) -> dict[str, int]:
    """Ingest multiple PDFs or directories of PDFs into the knowledge base."""
    initialize_database()
    pdf_files = collect_pdf_paths(paths)

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {paths}")

    if replace_all:
        print("Clearing all existing documents from database...")
        with SessionLocal.begin() as session:
            session.execute(delete(DocumentChunk))
            session.execute(delete(DocumentSection))

    results: dict[str, int] = {}
    print(f"Found {len(pdf_files)} document(s) to ingest:\n" + "\n".join(f"  - {f.name}" for f in pdf_files) + "\n")

    for pdf_path in pdf_files:
        count = ingest_pdf(pdf_path, replace_existing=replace_existing and not replace_all)
        results[pdf_path.name] = count

    total_chunks = sum(results.values())
    print(f"Ingestion complete: {len(results)} document(s), {total_chunks} total chunks stored.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest one or more PDF documents into vkassist RAG.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more PDF file paths, directory paths, or glob patterns.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing chunks for each document being ingested.",
    )
    parser.add_argument(
        "--replace-all",
        action="store_true",
        help="Clear all existing documents from the database before ingesting.",
    )
    args = parser.parse_args()

    ingest_documents(
        paths=args.paths,
        replace_existing=args.replace,
        replace_all=args.replace_all,
    )


if __name__ == "__main__":
    main()
