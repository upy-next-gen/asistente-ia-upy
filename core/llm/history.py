"""Utilities for managing LLM message history windows."""


def trim_history(history: list[dict], max_messages: int) -> list[dict]:
    """Recorta el historial de conversación conservando siempre el system prompt.

    Args:
        history: Lista de mensajes con roles "system", "user", "assistant".
        max_messages: Número máximo de mensajes de conversación a conservar.

    Returns:
        Historial recortado con el system prompt intacto al inicio.
    """
    system = history[:1]
    conversation = history[1:]
    if len(conversation) > max_messages:
        conversation = conversation[-max_messages:]
    return system + conversation
