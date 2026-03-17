from unittest.mock import MagicMock, patch
from core.rag.context import ContextBuilder


class TestContextBuilder:
    @patch("core.rag.context.observe", lambda **kwargs: lambda f: f)
    def test_build_empty_nodes(self):
        result = ContextBuilder.build([])
        assert result == ""

    def test_build_single_node(self):
        node = MagicMock()
        node.metadata = {"file_name": "guia.pdf"}
        node.text = "Texto del documento"
        result = ContextBuilder.build([node])
        assert "[Fuente: guia.pdf]" in result
        assert "Texto del documento" in result

    def test_build_multiple_nodes(self):
        node1 = MagicMock()
        node1.metadata = {"file_name": "a.pdf"}
        node1.text = "Texto A"
        node2 = MagicMock()
        node2.metadata = {"file_name": "b.pdf"}
        node2.text = "Texto B"
        result = ContextBuilder.build([node1, node2])
        assert "Texto A" in result
        assert "Texto B" in result
        assert "---" in result

    def test_build_missing_filename(self):
        node = MagicMock()
        node.metadata = {}
        node.text = "Sin fuente"
        result = ContextBuilder.build([node])
        assert "[Fuente: documento]" in result

        