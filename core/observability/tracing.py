import atexit
from langfuse import get_client


class TracingManager:
    _client = get_client()

    @classmethod
    def shutdown(cls) -> None:
        cls._client.flush()

    @classmethod
    def update_trace(
        cls,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        cls._client.update_current_trace(
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )

    @classmethod
    def score(cls, name: str, value: float, comment: str | None = None) -> None:
        cls._client.score_current_trace(
            name=name,
            value=value,
            comment=comment,
        )


atexit.register(TracingManager.shutdown)