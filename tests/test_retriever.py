import pytest
from unittest.mock import patch, MagicMock, call


class TestDocumentRetriever:

    def _make_retriever(self, mock_settings, mock_chroma, mock_pplx):
        mock_settings.PERPLEXITY_API_KEY = "pplx-test"
        mock_settings.CHROMA_DIR = "/tmp/chroma"
        mock_settings.CHROMA_COLLECTION_NAME = "upy_docs_pplx"
        mock_settings.EMBEDDING_MODEL = "pplx-embed-context-v1-0.6b"
        mock_settings.RETRIEVER_TOP_K = 5

        mock_collection = MagicMock()
        mock_chroma.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection
        mock_pplx.return_value = MagicMock()

        from core.rag.retriever import DocumentRetriever
        retriever = DocumentRetriever()
        return retriever, mock_collection

    @patch("core.rag.retriever.settings")
    @patch("core.rag.retriever.chromadb")
    @patch("core.rag.retriever.Perplexity")
    def test_retrieve_returns_nodes(self, mock_pplx, mock_chroma, mock_settings):
        retriever, mock_collection = self._make_retriever(
            mock_settings, mock_chroma, mock_pplx
        )

        mock_embed_response = MagicMock()
        mock_embed_response.data[0].data[0].embedding = "AAAA"
        mock_pplx.return_value.contextualized_embeddings.create.return_value = mock_embed_response

        mock_collection.query.return_value = {
            "documents": [["Texto A", "Texto B"]],
            "metadatas": [[{"file_name": "a.pdf"}, {"file_name": "b.pdf"}]],
            "distances": [[0.1, 0.2]],
        }

        results = retriever.retrieve("test query")
        assert len(results) == 2

    @patch("core.rag.retriever.settings")
    @patch("core.rag.retriever.chromadb")
    @patch("core.rag.retriever.Perplexity")
    def test_retrieve_empty_results(self, mock_pplx, mock_chroma, mock_settings):
        retriever, mock_collection = self._make_retriever(
            mock_settings, mock_chroma, mock_pplx
        )

        mock_embed_response = MagicMock()
        mock_embed_response.data[0].data[0].embedding = "AAAA"
        mock_pplx.return_value.contextualized_embeddings.create.return_value = mock_embed_response

        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        results = retriever.retrieve("query sin resultados")
        assert results == []

    @patch("core.rag.retriever.settings")
    @patch("core.rag.retriever.chromadb")
    @patch("core.rag.retriever.Perplexity")
    def test_retrieve_none_results(self, mock_pplx, mock_chroma, mock_settings):
        retriever, mock_collection = self._make_retriever(
            mock_settings, mock_chroma, mock_pplx
        )

        mock_embed_response = MagicMock()
        mock_embed_response.data[0].data[0].embedding = "AAAA"
        mock_pplx.return_value.contextualized_embeddings.create.return_value = mock_embed_response

        mock_collection.query.return_value = None

        results = retriever.retrieve("query")
        assert results == []

    @patch("core.rag.retriever.settings")
    @patch("core.rag.retriever.chromadb")
    @patch("core.rag.retriever.Perplexity")
    def test_retrieve_scores_are_inverted_distance(self, mock_pplx, mock_chroma, mock_settings):
        retriever, mock_collection = self._make_retriever(
            mock_settings, mock_chroma, mock_pplx
        )

        mock_embed_response = MagicMock()
        mock_embed_response.data[0].data[0].embedding = "AAAA"
        mock_pplx.return_value.contextualized_embeddings.create.return_value = mock_embed_response

        mock_collection.query.return_value = {
            "documents": [["Texto A"]],
            "metadatas": [[{"file_name": "a.pdf"}]],
            "distances": [[0.3]],
        }

        results = retriever.retrieve("test")
        assert pytest.approx(results[0].score) == 0.7

    @patch("core.rag.retriever.settings")
    @patch("core.rag.retriever.chromadb")
    @patch("core.rag.retriever.Perplexity")
    def test_retrieve_node_metadata_preserved(self, mock_pplx, mock_chroma, mock_settings):
        retriever, mock_collection = self._make_retriever(
            mock_settings, mock_chroma, mock_pplx
        )

        mock_embed_response = MagicMock()
        mock_embed_response.data[0].data[0].embedding = "AAAA"
        mock_pplx.return_value.contextualized_embeddings.create.return_value = mock_embed_response

        mock_collection.query.return_value = {
            "documents": [["Contenido del nodo"]],
            "metadatas": [[{"file_name": "guia.pdf", "page": 3}]],
            "distances": [[0.1]],
        }

        results = retriever.retrieve("test")
        assert results[0].node.metadata["file_name"] == "guia.pdf"
        assert results[0].node.metadata["page"] == 3
        assert results[0].node.text == "Contenido del nodo"

    @patch("core.rag.retriever.settings")
    @patch("core.rag.retriever.chromadb")
    @patch("core.rag.retriever.Perplexity")
    def test_retrieve_queries_with_correct_top_k(self, mock_pplx, mock_chroma, mock_settings):
        retriever, mock_collection = self._make_retriever(
            mock_settings, mock_chroma, mock_pplx
        )

        mock_embed_response = MagicMock()
        mock_embed_response.data[0].data[0].embedding = "AAAA"
        mock_pplx.return_value.contextualized_embeddings.create.return_value = mock_embed_response

        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        retriever.retrieve("test")
        call_kwargs = mock_collection.query.call_args
        assert call_kwargs.kwargs["n_results"] == 5    
        