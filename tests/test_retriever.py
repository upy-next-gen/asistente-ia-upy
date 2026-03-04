from unittest.mock import patch, MagicMock


class TestDocumentRetriever:
    @patch("core.rag.retriever.clients")
    @patch("core.rag.retriever.VectorStoreIndex")
    def test_retrieve_returns_nodes(self, mock_index_cls, mock_clients):
        mock_clients.init_embeddings = MagicMock()
        mock_clients.vector_store = MagicMock()
        mock_index = MagicMock()
        mock_index_cls.from_vector_store.return_value = mock_index
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_retriever.retrieve.return_value = [MagicMock(), MagicMock()]

        from core.rag.retriever import DocumentRetriever
        retriever = DocumentRetriever()
        results = retriever.retrieve("test query")
        assert len(results) == 2
        mock_retriever.retrieve.assert_called_once_with("test query")

    @patch("core.rag.retriever.clients")
    @patch("core.rag.retriever.VectorStoreIndex")
    def test_retrieve_empty(self, mock_index_cls, mock_clients):
        mock_clients.init_embeddings = MagicMock()
        mock_clients.vector_store = MagicMock()
        mock_index = MagicMock()
        mock_index_cls.from_vector_store.return_value = mock_index
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_retriever.retrieve.return_value = []

        from core.rag.retriever import DocumentRetriever
        retriever = DocumentRetriever()
        results = retriever.retrieve("query sin resultados")
        assert results == []