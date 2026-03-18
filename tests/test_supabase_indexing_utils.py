"""Tests for indexing utilities."""
import base64
import unittest

import numpy as np

from core.supabase_vector_db.indexing_utils import (
    build_metadata_rows,
    decode_embedding,
    infer_document_type,
)


class IndexingUtilsTests(unittest.TestCase):
    def test_decode_embedding_from_int8_base64(self):
        """Test decoding int8 base64 embeddings from Perplexity API."""
        original = np.array([1, -2, 3], dtype=np.int8)
        encoded = base64.b64encode(original.tobytes()).decode("utf-8")
        decoded = decode_embedding(encoded)
        self.assertEqual(decoded, [1.0, -2.0, 3.0])

    def test_decode_embedding_large_array(self):
        """Test decoding large int8 arrays."""
        original = np.array([i % 128 - 64 for i in range(1024)], dtype=np.int8)
        encoded = base64.b64encode(original.tobytes()).decode("utf-8")
        decoded = decode_embedding(encoded)
        self.assertEqual(len(decoded), 1024)
        self.assertEqual(decoded[0], float(original[0]))

    def test_infer_document_type_pdf(self):
        """Test document type inference for PDF files."""
        self.assertEqual(infer_document_type("archivo.PDF"), "pdf")
        self.assertEqual(infer_document_type("reglamento.pdf"), "pdf")
        self.assertEqual(infer_document_type("PLAN_ESTUDIOS.Pdf"), "pdf")

    def test_infer_document_type_markdown(self):
        """Test document type inference for Markdown files."""
        self.assertEqual(infer_document_type("readme.md"), "md")
        self.assertEqual(infer_document_type("document.MD"), "md")

    def test_infer_document_type_unknown(self):
        """Test document type inference for unknown file types."""
        self.assertEqual(infer_document_type("sin_extension"), "unknown")
        self.assertEqual(infer_document_type("file.xyz"), "xyz")

    def test_infer_document_type_with_path(self):
        """Test document type inference with full file path."""
        self.assertEqual(infer_document_type("docs/reglamentos/archivo.pdf"), "pdf")
        self.assertEqual(infer_document_type("/home/user/documento.md"), "md")

    def test_build_metadata_rows_default_values(self):
        """Test building metadata rows with default parameters."""
        rows = build_metadata_rows(["uno.pdf", "dos.md"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_file_name"], "uno.pdf")
        self.assertEqual(rows[0]["document_type"], "pdf")
        self.assertEqual(rows[0]["version_tag"], "current")
        self.assertEqual(rows[0]["category"], "general")

    def test_build_metadata_rows_custom_params(self):
        """Test building metadata rows with custom parameters."""
        rows = build_metadata_rows(
            ["uno.pdf", "dos.md"],
            version_tag="2024",
            category="reglamentos"
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_file_name"], "uno.pdf")
        self.assertEqual(rows[0]["document_type"], "pdf")
        self.assertEqual(rows[0]["version_tag"], "2024")
        self.assertEqual(rows[0]["category"], "reglamentos")
        self.assertEqual(rows[1]["source_file_name"], "dos.md")
        self.assertEqual(rows[1]["document_type"], "md")

    def test_build_metadata_rows_no_chunk_count(self):
        """Test that chunk_count is not included in metadata rows."""
        rows = build_metadata_rows(["archivo.pdf"], version_tag="v1", category="docs")
        self.assertNotIn("chunk_count", rows[0])
        # Verify expected fields are present
        expected_fields = {"source_file_name", "document_type", "version_tag", "category"}
        self.assertEqual(set(rows[0].keys()), expected_fields)

    def test_build_metadata_rows_empty_list(self):
        """Test building metadata rows from empty file list."""
        rows = build_metadata_rows([])
        self.assertEqual(len(rows), 0)

    def test_build_metadata_rows_single_file(self):
        """Test building metadata rows for single file."""
        rows = build_metadata_rows(["single.pdf"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_file_name"], "single.pdf")


if __name__ == "__main__":
    unittest.main()
