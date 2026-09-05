import argparse
from pathlib import Path

from sqlalchemy import delete

from app.chunker import parse_pdf
from app.database import SessionLocal, initialize_database
from app.embeddings import embed_documents
from app.models import DocumentChunk


def ingest_pdf(pdf_path: Path, replace_existing: bool = False) -> None:
    initialize_database()

    print(f"Parsing {pdf_path}...")
    chunks = parse_pdf(pdf_path)

    if not chunks:
        raise RuntimeError("No text chunks were extracted from the PDF")

    print(f"Extracted {len(chunks)} chunks")
    print("Generating embeddings...")

    embeddings = embed_documents(
        [chunk.content for chunk in chunks]
    )

    records = [
        DocumentChunk(
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            api_symbols=", ".join(chunk.api_symbols) or None,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]

    with SessionLocal.begin() as session:
        if replace_existing:
            session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_name == pdf_path.name
                )
            )

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