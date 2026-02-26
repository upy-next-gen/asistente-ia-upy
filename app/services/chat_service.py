from app.chain.conversation import conversation_chain
from app.core.logging import logger


def get_chat_response(user_message: str) -> str:
    """
    Invoke the LangChain conversation chain and return the model response.
    All LLM orchestration logic lives here, keeping the endpoint thin.
    """
    logger.info("Invoking conversation chain | message_length=%d", len(user_message))
    response: str = conversation_chain(user_message)
    logger.info("Chain response received | response_length=%d", len(response))
    return response
