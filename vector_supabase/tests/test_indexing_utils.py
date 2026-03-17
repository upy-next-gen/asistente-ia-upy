import base64
import unittest

import numpy as np

from vector_supabase.indexing_utils import (
    EMBEDDING_DIM,
    build_metadata_rows,
    decode_embedding,
    infer_document_type,
)


class IndexingUtilsTests(unittest.TestCase):
    def test_decode_embedding_from_int8_base64(self):
        original = np.array([1, -2, 3], dtype=np.int8)
        encoded = base64.b64encode(original.tobytes()).decode("utf-8")
        decoded = decode_embedding(encoded)
        self.assertEqual(decoded, [1.0, -2.0, 3.0])

    def test_infer_document_type(self):
        self.assertEqual(infer_document_type("archivo.PDF"), "pdf")
        self.assertEqual(infer_document_type("sin_extension"), "unknown")

    def test_build_metadata_rows_shape(self):
        rows = build_metadata_rows(["uno.pdf", "dos.md"], version_tag="2024", category="reglamentos")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["source_file_name"], "uno.pdf")
        self.assertEqual(rows[0]["document_type"], "pdf")
        self.assertEqual(rows[0]["version_tag"], "2024")
        self.assertEqual(rows[0]["category"], "reglamentos")
        self.assertNotIn("chunk_count", rows[0])


if __name__ == "__main__":
    unittest.main()

