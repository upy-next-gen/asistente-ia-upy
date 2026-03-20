from unittest.mock import patch
from core.rag.context import ContextBuilder


class TestContextBuilder:
    @patch("core.rag.context.observe", lambda **kwargs: lambda f: f)
    def test_build_empty_nodes(self):
        result = ContextBuilder.build([])
        assert result == ""

    def test_build_single_row(self):
        row = {"source_file_name": "guia.pdf", "content": "Texto del documento"}
        result = ContextBuilder.build([row])
        assert "[Fuente: guia.pdf]" in result
        assert "Texto del documento" in result

    def test_build_multiple_rows(self):
        row1 = {"source_file_name": "a.pdf", "content": "Texto A"}
        row2 = {"source_file_name": "b.pdf", "content": "Texto B"}
        result = ContextBuilder.build([row1, row2])
        assert "Texto A" in result
        assert "Texto B" in result
        assert "---" in result

    def test_build_missing_filename(self):
        row = {"content": "Sin fuente"}
        result = ContextBuilder.build([row])
        assert "[Fuente: documento]" in result 
        