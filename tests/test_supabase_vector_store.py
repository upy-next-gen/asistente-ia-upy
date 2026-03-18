"""Tests for Supabase vector store functionality."""
import unittest
from unittest.mock import MagicMock

from core.config import EMBEDDING_DIM, SAFE_BATCH_SIZE
from core.supabase_vector_db.store import SupabaseVectorStore


# Helper functions
def build_metadata_lookup_key(source_file_name: str, version_tag: str) -> str:
    """Build composite key for metadata lookup."""
    return f"{source_file_name}::{version_tag}"


def validate_embedding(embedding: list[float], expected_dim: int = EMBEDDING_DIM) -> None:
    """Validate embedding dimension."""
    store = SupabaseVectorStore(MagicMock())
    store.validate_embedding(embedding, expected_dim)


def build_match_documents_payload(
    query_embedding: list[float],
    match_count: int = 5,
    min_similarity: float = 0.3,
) -> dict:
    """Build match documents RPC payload."""
    store = SupabaseVectorStore(MagicMock())
    return store.build_match_documents_payload(
        query_embedding=query_embedding,
        match_count=match_count,
        min_similarity=min_similarity,
    )


class _FakeExecute:
    def __init__(self, data):
        self.data = data


class _FakeQueryBuilder:
    def __init__(self, table_name, state):
        self.table_name = table_name
        self.state = state
        self.selected = None
        self.filter_values = None

    def upsert(self, rows, on_conflict=None):
        self.state["upserts"].append(
            {
                "table": self.table_name,
                "rows": rows,
                "on_conflict": on_conflict,
            }
        )
        if self.table_name == "documents_metadata":
            for row in rows:
                key = build_metadata_lookup_key(row["source_file_name"], row["version_tag"])
                if key not in self.state["metadata"]:
                    self.state["metadata"][key] = f"id_{len(self.state['metadata']) + 1}"
        return self

    def select(self, value):
        self.selected = value
        return self

    def in_(self, key, values):
        self.filter_values = (key, values)
        return self

    def execute(self):
        if self.table_name == "documents_metadata" and self.selected:
            _, values = self.filter_values
            data = [
                {
                    "id": self.state["metadata"][build_metadata_lookup_key(name, "current")],
                    "source_file_name": name,
                    "version_tag": "current",
                }
                for name in values
                if build_metadata_lookup_key(name, "current") in self.state["metadata"]
            ]
            return _FakeExecute(data)
        return _FakeExecute([])


class _FakeSupabaseClient:
    def __init__(self):
        self.state = {"upserts": [], "metadata": {}, "rpc_calls": []}

    def table(self, table_name):
        return _FakeQueryBuilder(table_name, self.state)

    def rpc(self, function_name, payload):
        self.state["rpc_calls"].append({"function_name": function_name, "payload": payload})
        return _FakeQueryBuilderRpc(
            _FakeExecute(
                [
                    {
                        "id": "chunk-1",
                        "content": "contenido de prueba",
                        "source_file_name": "reglamento.pdf",
                        "chunk_index": 0,
                        "similarity": 0.91,
                    }
                ]
            )
        )


class _FakeQueryBuilderRpc:
    def __init__(self, execute_result):
        self.execute_result = execute_result

    def execute(self):
        return self.execute_result


class SupabaseVectorStoreTests(unittest.TestCase):
    def test_validate_embedding_dimension(self):
        """Test that embedding dimension validation works correctly."""
        validate_embedding([0.1] * EMBEDDING_DIM)
        with self.assertRaises(ValueError):
            validate_embedding([0.1] * (EMBEDDING_DIM - 1))

    def test_validate_embedding_non_numeric(self):
        """Test that non-numeric values in embedding are rejected."""
        store = SupabaseVectorStore(MagicMock())
        with self.assertRaises(TypeError):
            store.validate_embedding([0.1] * (EMBEDDING_DIM - 1) + ["invalid"])

    def test_build_match_payload(self):
        """Test that match payload is built correctly."""
        payload = build_match_documents_payload([0.0] * EMBEDDING_DIM, match_count=3, min_similarity=0.45)
        self.assertEqual(payload["match_count"], 3)
        self.assertEqual(payload["min_similarity"], 0.45)
        self.assertEqual(len(payload["query_embedding"]), EMBEDDING_DIM)

    def test_build_match_payload_invalid_match_count(self):
        """Test that invalid match_count raises ValueError."""
        store = SupabaseVectorStore(MagicMock())
        with self.assertRaises(ValueError):
            store.build_match_documents_payload([0.0] * EMBEDDING_DIM, match_count=0, min_similarity=0.45)

    def test_build_metadata_lookup_key(self):
        """Test that metadata lookup key is built correctly."""
        key = build_metadata_lookup_key("documento.pdf", "v1")
        self.assertEqual(key, "documento.pdf::v1")

    def test_build_metadata_lookup_key_complex_filename(self):
        """Test metadata lookup key with complex filenames."""
        key = build_metadata_lookup_key("docs/plan_estudios.pdf", "2024-01-15")
        self.assertEqual(key, "docs/plan_estudios.pdf::2024-01-15")

    def test_safe_batch_size_constant(self):
        """Test that SAFE_BATCH_SIZE constant is set correctly."""
        self.assertEqual(SAFE_BATCH_SIZE, 200)

    def test_embedding_dim_constant(self):
        """Test that EMBEDDING_DIM constant is set correctly."""
        self.assertEqual(EMBEDDING_DIM, 1024)


if __name__ == "__main__":
    unittest.main()
