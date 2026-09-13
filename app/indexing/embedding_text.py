def build_embedding_text(
    document_name: str,
    section: str | None,
    page_number: int,
    content: str,
    api_symbols: list[str],
) -> str:
    """
    Creates the text sent to the embedding model.

    The original clean content should still be stored separately
    and displayed to the user.
    """
    parts = [
        f"Document: {document_name}",
        f"Page: {page_number}",
    ]

    if section:
        parts.append(f"Section: {section}")

    if api_symbols:
        parts.append(f"Vulkan symbols: {', '.join(api_symbols)}")

    parts.append("")
    parts.append(content)

    return "\n".join(parts)