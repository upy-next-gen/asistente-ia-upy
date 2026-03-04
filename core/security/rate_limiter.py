import time
from core.config import settings


class RateLimiter:
    def __init__(self):
        self._sessions: dict[str, list[float]] = {}

    def is_allowed(self, session_id: str) -> bool:
        now = time.time()
        window = settings.rate_limit_window
        max_requests = settings.rate_limit_max_requests

        if session_id not in self._sessions:
            self._sessions[session_id] = []

        timestamps = self._sessions[session_id]
        self._sessions[session_id] = [t for t in timestamps if now - t < window]

        if len(self._sessions[session_id]) >= max_requests:
            return False

        self._sessions[session_id].append(now)
        return True