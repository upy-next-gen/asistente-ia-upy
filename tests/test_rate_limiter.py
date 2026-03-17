import time
import pytest
from unittest.mock import patch
from core.security.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_first_request_is_allowed(self):
        limiter = RateLimiter()
        assert limiter.is_allowed("session-1") is True

    def test_requests_within_limit_are_allowed(self):
        with patch("core.security.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_WINDOW = 60
            mock_settings.RATE_LIMIT_MAX_REQUESTS = 5
            limiter = RateLimiter()
            for _ in range(5):
                assert limiter.is_allowed("session-1") is True

    def test_request_exceeding_limit_is_blocked(self):
        with patch("core.security.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_WINDOW = 60
            mock_settings.RATE_LIMIT_MAX_REQUESTS = 3
            limiter = RateLimiter()
            limiter.is_allowed("session-1")
            limiter.is_allowed("session-1")
            limiter.is_allowed("session-1")
            assert limiter.is_allowed("session-1") is False

    def test_different_sessions_are_independent(self):
        with patch("core.security.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_WINDOW = 60
            mock_settings.RATE_LIMIT_MAX_REQUESTS = 1
            limiter = RateLimiter()
            limiter.is_allowed("session-1")
            assert limiter.is_allowed("session-1") is False
            assert limiter.is_allowed("session-2") is True

    def test_expired_timestamps_are_cleaned(self):
        with patch("core.security.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_WINDOW = 1
            mock_settings.RATE_LIMIT_MAX_REQUESTS = 2
            limiter = RateLimiter()
            limiter.is_allowed("session-1")
            limiter.is_allowed("session-1")
            assert limiter.is_allowed("session-1") is False
            time.sleep(1.1)
            assert limiter.is_allowed("session-1") is True

    def test_new_session_gets_empty_list(self):
        with patch("core.security.rate_limiter.settings") as mock_settings:
            mock_settings.RATE_LIMIT_WINDOW = 60
            mock_settings.RATE_LIMIT_MAX_REQUESTS = 10
            limiter = RateLimiter()
            assert "session-new" not in limiter._sessions
            limiter.is_allowed("session-new")
            assert "session-new" in limiter._sessions