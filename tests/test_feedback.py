import pytest
from unittest.mock import patch, MagicMock


class TestFeedbackService:
    @patch("core.feedback.suggestions.clients")
    def test_save_success(self, mock_clients):
        mock_table = MagicMock()
        mock_supabase = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_clients.supabase = mock_supabase

        from core.feedback.suggestions import FeedbackService
        service = FeedbackService()
        service.save("buena sugerencia")

        mock_supabase.table.assert_called_once_with("sugerencias")
        mock_table.insert.assert_called_once_with({"contenido": "buena sugerencia"})

    @patch("core.feedback.suggestions.clients")
    def test_save_raises_without_supabase(self, mock_clients):
        mock_clients.supabase = None

        from core.feedback.suggestions import FeedbackService
        service = FeedbackService()

        with pytest.raises(RuntimeError, match="Supabase no está configurado"):
            service.save("algo")