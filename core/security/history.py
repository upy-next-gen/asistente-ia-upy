from core.config import settings


def trim_history(history: list[dict]) -> list[dict]:
    system = history[:1]
    conversation = history[1:]
    if len(conversation) > settings.MAX_HISTORY_MESSAGES:
        conversation = conversation[-settings.MAX_HISTORY_MESSAGES:]
    return system + conversation