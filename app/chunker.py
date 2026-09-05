import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class Chunk:
    document_name: str
    page_number: int
    chunk_index: int
    content: str
    api_symbols: list[str]


def normalize_text(text: str) -> str:
    # Replace repeated whitespace while keeping paragraph boundaries usable.
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_api_symbols(text: str) -> list[str]:
    patterns = [
        r"\bvk[A-Z][A-Za-z0-9_]*\b",
        r"\bVk[A-Z][A-Za-z0-9_]*\b",
        r"\bVK_[A-Z0-9_]+\b",
        r"\bVUID-[A-Za-z0-9_-]+\b",
    ]

    symbols: set[str] = set()

    for pattern in patterns:
        symbols.update(re.findall(pattern, text))

    return sorted(symbols)


def split_words_with_overlap(
    text: str,
    chunk_size: int = 350,
    overlap: int = 60,
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    words = text.split()

    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def parse_pdf(pdf_path: Path) -> list[Chunk]:
    document = pymupdf.open(pdf_path)
    chunks: list[Chunk] = []
    global_chunk_index = 0

    for page_index, page in enumerate(document):
        # Sorting usually improves reading order for multi-block pages.
        page_text = page.get_text("text", sort=True)
        page_text = normalize_text(page_text)

        for content in split_words_with_overlap(page_text):
            chunks.append(
                Chunk(
                    document_name=pdf_path.name,
                    page_number=page_index + 1,
                    chunk_index=global_chunk_index,
                    content=content,
                    api_symbols=extract_api_symbols(content),
                )
            )

            global_chunk_index += 1

    return chunks