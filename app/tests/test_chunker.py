import unittest

from app.chunker import (
    TextBlock,
    extract_api_symbols,
    matching_toc_heading,
    normalize_text,
    pack_blocks,
)


class ChunkerTests(unittest.TestCase):
    def test_normalizes_pdf_noise_without_flattening_paragraphs(self) -> None:
        self.assertEqual(normalize_text("one\u00ad\n\n\n  two"), "one\n\ntwo")

    def test_extracts_vulkan_symbols(self) -> None:
        text = "Call vkQueueSubmit with VkSubmitInfo and VK_SUCCESS."
        self.assertEqual(extract_api_symbols(text), ["VK_SUCCESS", "VkSubmitInfo", "vkQueueSubmit"])

    def test_packing_respects_budget_and_keeps_block_boundaries(self) -> None:
        blocks = [TextBlock(1, "alpha " * 120, False), TextBlock(2, "beta " * 120, False)]
        chunks = pack_blocks(blocks, chunk_size=200, overlap_words=40)
        self.assertTrue(all(len(content.split()) <= 200 for content, _, _ in chunks))
        self.assertEqual(chunks[0][1:], (1, 1))
        self.assertEqual(chunks[1][1:], (2, 2))

    def test_detects_a_toc_heading_at_its_document_boundary(self) -> None:
        headings = {"graphicspipeline": [(69, "Drawing > Graphics pipeline")]}
        self.assertEqual(
            matching_toc_heading("Graphics pipeline", 69, headings),
            (69, "Drawing > Graphics pipeline"),
        )


if __name__ == "__main__":
    unittest.main()
