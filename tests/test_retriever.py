from unittest.mock import patch, MagicMock


class TestDocumentRetriever:

    def _make_retriever(self, mock_settings, mock_store_class, mock_get_client):
        mock_settings.RETRIEVER_TOP_K = 5
        mock_settings.MIN_SIMILARITY = 0.3
        
        mock_store = mock_store_class.return_value
        
        from core.rag.retriever import DocumentRetriever
        retriever = DocumentRetriever()
        return retriever, mock_store

    @patch("core.rag.retriever.get_supabase_client")
    @patch("core.rag.retriever.SupabaseVectorStore")
    @patch("core.rag.retriever.settings")
    def test_retrieve_returns_rows(self, mock_settings, mock_store_class, mock_get_client):
        retriever, mock_store = self._make_retriever(mock_settings, mock_store_class, mock_get_client)
        
        mock_payload = {"some": "payload"}
        mock_store.build_match_documents_payload.return_value = mock_payload
        
        mock_response = MagicMock()
        mock_response.data = [
            {"id": "1", "content": "Texto A", "source_file_name": "a.pdf"},
            {"id": "2", "content": "Texto B", "source_file_name": "b.pdf"},
        ]
        mock_store.client.rpc.return_value.execute.return_value = mock_response

        results = retriever.retrieve([0.1] * 1024)
        assert len(results) == 2
        assert results[0]["source_file_name"] == "a.pdf"
        
        mock_store.build_match_documents_payload.assert_called_once_with(
            query_embedding=[0.1] * 1024,
            match_count=5,
            min_similarity=0.3
        )
        mock_store.client.rpc.assert_called_once_with("match_documents", mock_payload)

    @patch("core.rag.retriever.get_supabase_client")
    @patch("core.rag.retriever.SupabaseVectorStore")
    @patch("core.rag.retriever.settings")
    def test_retrieve_empty_results(self, mock_settings, mock_store_class, mock_get_client):
        retriever, mock_store = self._make_retriever(mock_settings, mock_store_class, mock_get_client)
        
        mock_response = MagicMock()
        mock_response.data = []
        mock_store.client.rpc.return_value.execute.return_value = mock_response

        results = retriever.retrieve([0.1] * 1024)
        assert results == []

    @patch("core.rag.retriever.get_supabase_client")
    @patch("core.rag.retriever.SupabaseVectorStore")
    @patch("core.rag.retriever.settings")
    def test_retrieve_uses_default_top_k_from_settings(self, mock_settings, mock_store_class, mock_get_client):
        retriever, mock_store = self._make_retriever(mock_settings, mock_store_class, mock_get_client)
        
        mock_response = MagicMock()
        mock_response.data = []
        mock_store.client.rpc.return_value.execute.return_value = mock_response

        retriever.retrieve([0.1] * 1024)
        kwargs = mock_store.build_match_documents_payload.call_args.kwargs
        assert kwargs["match_count"] == 5
        assert kwargs["min_similarity"] == 0.3

    @patch("core.rag.retriever.get_supabase_client")
    @patch("core.rag.retriever.SupabaseVectorStore")
    @patch("core.rag.retriever.settings")
    def test_retrieve_overrides_top_k(self, mock_settings, mock_store_class, mock_get_client):
        retriever, mock_store = self._make_retriever(mock_settings, mock_store_class, mock_get_client)
        
        mock_response = MagicMock()
        mock_response.data = []
        mock_store.client.rpc.return_value.execute.return_value = mock_response

        retriever.retrieve([0.1] * 1024, top_k=8)
        kwargs = mock_store.build_match_documents_payload.call_args.kwargs
        assert kwargs["match_count"] == 8
        