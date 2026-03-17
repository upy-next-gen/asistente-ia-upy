import unittest

from vector_supabase.vector_store import (
    EMBEDDING_DIM,
    SAFE_BATCH_SIZE,
    build_metadata_lookup_key,
    build_match_documents_payload,
    search_documents_by_embedding,
    upsert_document_chunks,
    upsert_metadata_rows,
    validate_embedding,
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


class VectorStoreTests(unittest.TestCase):
    def test_validate_embedding_dimension(self):
        validate_embedding([0.1] * EMBEDDING_DIM)
        with self.assertRaises(ValueError):
            validate_embedding([0.1] * (EMBEDDING_DIM - 1))

    def test_build_match_payload(self):
        payload = build_match_documents_payload([0.0] * EMBEDDING_DIM, match_count=3, min_similarity=0.45)
        self.assertEqual(payload["match_count"], 3)
        self.assertEqual(payload["min_similarity"], 0.45)

    def test_search_documents_by_embedding_calls_rpc(self):
        client = _FakeSupabaseClient()
        rows = search_documents_by_embedding(
            client,
            [0.0] * EMBEDDING_DIM,
            match_count=5,
            min_similarity=0.3,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(client.state["rpc_calls"][0]["function_name"], "match_documents")
        self.assertEqual(client.state["rpc_calls"][0]["payload"]["min_similarity"], 0.3)

    def test_upsert_metadata_rows_returns_mapping(self):
        client = _FakeSupabaseClient()
        mapping = upsert_metadata_rows(
            client,
            [
                {
                    "source_file_name": "plan_estudios.pdf",
                    "document_type": "pdf",
                    "version_tag": "current",
                    "category": "oferta_academica",
                }
            ],
        )
        self.assertIn(build_metadata_lookup_key("plan_estudios.pdf", "current"), mapping)

    def test_upsert_document_chunks_validates_rows(self):
        client = _FakeSupabaseClient()
        upsert_document_chunks(
            client,
            [
                {
                    "metadata_id": "id_1",
                    "content": "contenido",
                    "chunk_index": 1,
                    "embedding": [0.0] * EMBEDDING_DIM,
                }
            ],
            batch_size=1,
        )
        self.assertEqual(len(client.state["upserts"]), 1)
        self.assertEqual(client.state["upserts"][0]["on_conflict"], "metadata_id,chunk_index")

    def test_safe_batch_size_default(self):
        self.assertEqual(SAFE_BATCH_SIZE, 200)


if __name__ == "__main__":
    unittest.main()

