from unittest.mock import MagicMock, patch


class TestTracingManager:
    def _get_manager(self):
        from core.observability.tracing import TracingManager
        return TracingManager

    def test_shutdown_flushes(self):
        manager = self._get_manager()
        mock_client = MagicMock()
        with patch.object(manager, "_client", mock_client):
            manager.shutdown()
            mock_client.flush.assert_called_once()

    def test_update_trace(self):
        manager = self._get_manager()
        mock_client = MagicMock()
        with patch.object(manager, "_client", mock_client):
            manager.update_trace(
                user_id="user1",
                session_id="sess1",
                metadata={"key": "val"},
            )
            mock_client.update_current_trace.assert_called_once_with(
                user_id="user1",
                session_id="sess1",
                metadata={"key": "val"},
            )

    def test_score(self):
        manager = self._get_manager()
        mock_client = MagicMock()
        with patch.object(manager, "_client", mock_client):
            manager.score("retrieval_nodes", 5.0, "test")
            mock_client.score_current_trace.assert_called_once_with(
                name="retrieval_nodes",
                value=5.0,
                comment="test",
            )