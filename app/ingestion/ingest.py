import argparse
from pathlib import Path

from sqlalchemy import delete

from app.chunker import parse_document
from app.database import SessionLocal, initialize_database
from app.embeddings import embed_documents
from app.indexing.embedding_text import build_embedding_text
from app.models import DocumentChunk, DocumentSection


def ingest_pdf(pdf_path: Path, replace_existing: bool = False) -> None:
    initialize_database()

    print(f"Parsing {pdf_path}...")
    parsed = parse_document(pdf_path)
    chunks = parsed.chunks

    if not chunks:
        raise RuntimeError("No text chunks were extracted from the PDF")

    print(f"Extracted {len(chunks)} chunks")
    print("Generating embeddings...")

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
            session.execute(delete(DocumentSection).where(DocumentSection.document_name == pdf_path.name))

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
            record.section_id = section_ids[chunk.parent_key]

        session.add_all(records)

    print(f"Stored {len(records)} chunks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    ingest_pdf(
        pdf_path=args.pdf_path,
        replace_existing=args.replace,
    )


if __name__ == "__main__":
    main()
