from unittest.mock import MagicMock, patch

from core.rag.embedder import QueryEmbedder


class TestQueryEmbedder:
    @patch("core.rag.embedder.decode_embedding")
    def test_embed_calls_client_with_model(self, mock_decode_embedding):
        client = MagicMock()
        response = MagicMock()
        response.data = [MagicMock(data=[MagicMock(embedding="base64-embedding")])]
        client.contextualized_embeddings.create.return_value = response
        mock_decode_embedding.return_value = [0.1, 0.2, 0.3]

        embedder = QueryEmbedder(client=client, model="pplx-embed-context-v1-0.6b")
        result = embedder.embed("hola")

        client.contextualized_embeddings.create.assert_called_once_with(
            input=[["hola"]],
            model="pplx-embed-context-v1-0.6b",
        )
        mock_decode_embedding.assert_called_once_with("base64-embedding")
        assert result == [0.1, 0.2, 0.3]
