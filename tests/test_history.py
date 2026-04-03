from core.llm.history import trim_history


class TestTrimHistory:
    def test_keeps_system_and_latest_messages(self):
        history = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]

        trimmed = trim_history(history, max_messages=2)
        assert trimmed == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]

    def test_returns_original_when_under_limit(self):
        history = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u1"},
        ]
        assert trim_history(history, max_messages=10) == history
