import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class Chunk:
    document_name: str
    page_number: int
    page_end: int
    chunk_index: int
    content: str
    section: str | None
    parent_key: str
    api_symbols: list[str]


@dataclass
class Section:
    key: str
    document_name: str
    path: str | None
    start_page: int
    end_page: int
    content: str


@dataclass
class ParsedDocument:
    chunks: list[Chunk]
    sections: list[Section]


@dataclass
class TocEntry:
    level: int
    title: str
    page_number: int


@dataclass
class TextBlock:
    page_number: int
    text: str
    is_code: bool
    is_heading: bool = False


def normalize_text(text: str) -> str:
    """Remove PDF noise while preserving paragraph and code line boundaries."""
    text = text.replace("\u00ad", "")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_table_of_contents(document: pymupdf.Document) -> list[TocEntry]:
    return [
        TocEntry(level, normalize_text(title), page_number)
        for level, title, page_number in document.get_toc()
        if page_number > 0
    ]


def build_section_paths(toc_entries: list[TocEntry]) -> list[tuple[int, str]]:
    hierarchy: dict[int, str] = {}
    paths: list[tuple[int, str]] = []
    for entry in toc_entries:
        hierarchy[entry.level] = entry.title
        for level in [level for level in hierarchy if level > entry.level]:
            del hierarchy[level]
        paths.append((entry.page_number, " > ".join(hierarchy[level] for level in sorted(hierarchy))))
    return paths


def get_section_for_page(page_number: int, section_paths: list[tuple[int, str]]) -> str | None:
    section = None
    for section_page, path in section_paths:
        if section_page > page_number:
            break
        section = path
    return section


def section_start_for_page(
    page_number: int,
    section_paths: list[tuple[int, str]],
) -> tuple[int, str | None]:
    start_page, section = page_number, None
    for section_page, path in section_paths:
        if section_page > page_number:
            break
        start_page, section = section_page, path
    return start_page, section


def heading_fingerprint(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def marginal_fingerprint(text: str) -> str:
    return re.sub(r"\d+", "#", heading_fingerprint(text))


def repeated_marginal_blocks(document: pymupdf.Document) -> set[str]:
    """Find recurring running headers/footers before they enter the corpus."""
    occurrences: dict[str, set[int]] = {}
    for page_index, page in enumerate(document):
        page_height = page.rect.height
        for block in page.get_text("blocks", sort=True):
            text = normalize_text(block[4])
            is_margin = block[1] < page_height * 0.12 or block[3] > page_height * 0.88
            fingerprint = marginal_fingerprint(text)
            if is_margin and fingerprint and len(text.split()) <= 20:
                occurrences.setdefault(fingerprint, set()).add(page_index)
    return {fingerprint for fingerprint, pages in occurrences.items() if len(pages) >= 3}


def toc_heading_paths(toc_entries: list[TocEntry]) -> dict[str, list[tuple[int, str]]]:
    mapping: dict[str, list[tuple[int, str]]] = {}
    for entry, (_, path) in zip(toc_entries, build_section_paths(toc_entries), strict=True):
        mapping.setdefault(heading_fingerprint(entry.title), []).append((entry.page_number, path))
    return mapping


def matching_toc_heading(
    text: str,
    page_number: int,
    headings: dict[str, list[tuple[int, str]]],
) -> tuple[int, str] | None:
    if len(text.split()) > 18:
        return None
    for start_page, path in headings.get(heading_fingerprint(text), []):
        if abs(page_number - start_page) <= 2:
            return start_page, path
    return None


def extract_api_symbols(text: str) -> list[str]:
    patterns = [r"\bvk[A-Z][A-Za-z0-9_]*\b", r"\bVk[A-Z][A-Za-z0-9_]*\b", r"\bVK_[A-Z0-9_]+\b", r"\bVUID-[A-Za-z0-9_-]+\b"]
    return sorted({symbol for pattern in patterns for symbol in re.findall(pattern, text)})


def is_code_block(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    code_lines = sum(bool(re.search(r"[{};]|#include|\b(?:if|for|while|return|class|void)\b", line)) for line in lines)
    return code_lines / len(lines) >= 0.35


def extract_text_blocks(
    page: pymupdf.Page,
    page_number: int,
    ignored_margins: set[str] | None = None,
) -> list[TextBlock]:
    """Keep PDF layout blocks intact so paragraphs and code are not word-sliced."""
    blocks: list[TextBlock] = []
    for block in page.get_text("blocks", sort=True):
        text = normalize_text(block[4])
        page_height = page.rect.height
        is_margin = block[1] < page_height * 0.12 or block[3] > page_height * 0.88
        if not text or (is_margin and marginal_fingerprint(text) in (ignored_margins or set())):
            continue
        if is_margin and re.fullmatch(r"\d+", text):
            continue
        is_code = is_code_block(text)
        blocks.append(TextBlock(page_number, text, is_code, len(text.split()) <= 18 and not is_code))
    return blocks


def split_oversized_block(block: TextBlock, max_words: int) -> list[TextBlock]:
    if len(block.text.split()) <= max_words:
        return [block]
    units = [line.strip() for line in block.text.splitlines() if line.strip()] if block.is_code else re.split(r"(?<=[.!?])\s+", block.text)
    pieces: list[TextBlock] = []
    current: list[str] = []
    current_words = 0
    for unit in units:
        unit_words = len(unit.split())
        if current and current_words + unit_words > max_words:
            pieces.append(TextBlock(block.page_number, "\n".join(current) if block.is_code else " ".join(current), block.is_code))
            current, current_words = [], 0
        if unit_words > max_words:
            words = unit.split()
            for start in range(0, len(words), max_words):
                pieces.append(TextBlock(block.page_number, " ".join(words[start:start + max_words]), block.is_code))
        else:
            current.append(unit)
            current_words += unit_words
    if current:
        pieces.append(TextBlock(block.page_number, "\n".join(current) if block.is_code else " ".join(current), block.is_code))
    return pieces


def pack_blocks(blocks: list[TextBlock], chunk_size: int = 350, overlap_words: int = 60) -> list[tuple[str, int, int]]:
    """Pack whole paragraphs/code blocks and overlap only complete tail blocks."""
    if overlap_words >= chunk_size:
        raise ValueError("overlap_words must be smaller than chunk_size")
    units = [piece for block in blocks for piece in split_oversized_block(block, chunk_size)]
    packed: list[tuple[str, int, int]] = []
    current: list[TextBlock] = []
    current_words = 0
    for unit in units:
        unit_words = len(unit.text.split())
        if current and current_words + unit_words > chunk_size:
            packed.append(("\n\n".join(item.text for item in current), current[0].page_number, current[-1].page_number))
            overlap: list[TextBlock] = []
            overlap_count = 0
            for item in reversed(current):
                item_words = len(item.text.split())
                if overlap and overlap_count + item_words > overlap_words:
                    break
                overlap.insert(0, item)
                overlap_count += item_words
            current, current_words = overlap, overlap_count
            # A large next block cannot coexist with the overlap while still
            # respecting the chunk budget; prefer an intact block.
            if current_words + unit_words > chunk_size:
                current, current_words = [], 0
        current.append(unit)
        current_words += unit_words
    if current:
        packed.append(("\n\n".join(item.text for item in current), current[0].page_number, current[-1].page_number))
    return packed


def parse_document(pdf_path: Path, chunk_size: int = 350, overlap_words: int = 60) -> ParsedDocument:
    with pymupdf.open(pdf_path) as document:
        toc_entries = extract_table_of_contents(document)
        section_paths = build_section_paths(toc_entries)
        headings = toc_heading_paths(toc_entries)
        ignored_margins = repeated_marginal_blocks(document)
        grouped: dict[str, list[TextBlock]] = {}
        section_info: dict[str, tuple[str | None, int]] = {}
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            start_page, path = section_start_for_page(page_number, section_paths)
            for block in extract_text_blocks(page, page_number, ignored_margins):
                matched_heading = matching_toc_heading(block.text, page_number, headings)
                if matched_heading:
                    start_page, path = matched_heading
                    block.is_heading = True
                key = hashlib.sha256(
                    f"{pdf_path.name}:{path or 'front-matter'}:{start_page}".encode()
                ).hexdigest()[:24]
                grouped.setdefault(key, []).append(block)
                section_info.setdefault(key, (path, start_page))

    sections: list[Section] = []
    chunks: list[Chunk] = []
    for key, blocks in grouped.items():
        path, start_page = section_info[key]
        content = "\n\n".join(block.text for block in blocks)
        sections.append(Section(key, pdf_path.name, path, start_page, blocks[-1].page_number, content))
        for content, page_start, page_end in pack_blocks(blocks, chunk_size, overlap_words):
            chunks.append(Chunk(pdf_path.name, page_start, page_end, len(chunks), content, path, key, extract_api_symbols(content)))
    return ParsedDocument(chunks, sections)


def parse_pdf(pdf_path: Path, chunk_size: int = 350, overlap: int = 60) -> list[Chunk]:
    """Backward-compatible child-chunk entry point."""
    return parse_document(pdf_path, chunk_size, overlap).chunks
