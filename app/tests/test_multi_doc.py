import tempfile
import unittest
from pathlib import Path

from app.ingestion.ingest import collect_pdf_paths


class MultiDocTests(unittest.TestCase):
    def test_collect_pdf_paths_from_files_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            doc1 = base / "doc1.pdf"
            doc2 = base / "doc2.pdf"
            nested = base / "subdir"
            nested.mkdir()
            doc3 = nested / "doc3.pdf"
            non_pdf = base / "notes.txt"

            doc1.write_bytes(b"%PDF-1.4 test")
            doc2.write_bytes(b"%PDF-1.4 test")
            doc3.write_bytes(b"%PDF-1.4 test")
            non_pdf.write_text("not a pdf", encoding="utf-8")

            # Test directory scanning
            found_dir = collect_pdf_paths([base])
            self.assertEqual(len(found_dir), 3)
            self.assertEqual({f.name for f in found_dir}, {"doc1.pdf", "doc2.pdf", "doc3.pdf"})

            # Test individual file scanning
            found_files = collect_pdf_paths([doc1, doc3])
            self.assertEqual(len(found_files), 2)
            self.assertEqual({f.name for f in found_files}, {"doc1.pdf", "doc3.pdf"})

            # Test deduplication
            found_dup = collect_pdf_paths([doc1, base])
            self.assertEqual(len(found_dup), 3)


if __name__ == "__main__":
    unittest.main()
