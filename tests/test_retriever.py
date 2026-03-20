from unittest.mock import patch, MagicMock


class TestDocumentRetriever:

    def _make_retriever(self, mock_settings, mock_backend):
        mock_settings.RETRIEVER_TOP_K = 5
        mock_settings.MIN_SIMILARITY = 0.3
        backend = MagicMock()
        mock_backend.return_value = backend
        from core.rag.retriever import DocumentRetriever
        retriever = DocumentRetriever()
        return retriever, backend

    @patch("core.rag.retriever.settings")
    @patch("core.rag.retriever.SupabaseVectorRetriever")
    def test_retrieve_returns_rows(self, mock_backend, mock_settings):
        retriever, backend = self._make_retriever(mock_settings, mock_backend)
        backend.search.return_value = [
            {"id": "1", "content": "Texto A", "source_file_name": "a.pdf", "similarity_0_1": 0.9},
            {"id": "2", "content": "Texto B", "source_file_name": "b.pdf", "similarity_0_1": 0.8},
        ]

        results = retriever.retrieve([0.1] * 1024)
        assert len(results) == 2
        assert results[0]["source_file_name"] == "a.pdf"

    @patch("core.rag.retriever.settings")
    @patch("core.rag.retriever.SupabaseVectorRetriever")
    def test_retrieve_empty_results(self, mock_backend, mock_settings):
        retriever, backend = self._make_retriever(mock_settings, mock_backend)
        backend.search.return_value = []

        results = retriever.retrieve([0.1] * 1024)
        assert results == []

    @patch("core.rag.retriever.settings")
    @patch("core.rag.retriever.SupabaseVectorRetriever")
    def test_retrieve_uses_default_top_k_from_settings(self, mock_backend, mock_settings):
        retriever, backend = self._make_retriever(mock_settings, mock_backend)
        backend.search.return_value = []

        retriever.retrieve([0.1] * 1024)
        kwargs = backend.search.call_args.kwargs
        assert kwargs["match_count"] == 5
        assert kwargs["min_similarity"] == 0.3

    @patch("core.rag.retriever.settings")
    @patch("core.rag.retriever.SupabaseVectorRetriever")
    def test_retrieve_overrides_top_k(self, mock_backend, mock_settings):
        retriever, backend = self._make_retriever(mock_settings, mock_backend)
        backend.search.return_value = []

        retriever.retrieve([0.1] * 1024, top_k=8)
        kwargs = backend.search.call_args.kwargs
        assert kwargs["match_count"] == 8
        