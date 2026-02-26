from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.core.config import settings

SYSTEM_PROMPT = (
    "Eres el asistente virtual oficial de la Universidad Politécnica de Yucatán (UPY). "
    "Tu ÚNICA tarea es responder preguntas estrictamente relacionadas con la universidad, "
    "trámites, carreras, horarios, vida estudiantil y temas académicos de la institución. "
    "REGLA INQUEBRANTABLE: Si el usuario pregunta sobre cualquier otro tema fuera de la UPY, "
    "responde exactamente: 'Lo siento, como asistente de la UPY, mi conocimiento y funciones "
    "están limitados a temas de la universidad. ¿Te puedo ayudar con alguna duda sobre "
    "inscripciones o carreras?'. No proporciones ninguna otra información."
)


def build_llm() -> ChatOpenAI:
    """Instantiate the LangChain-compatible LLM pointed at DeepSeek."""
    return ChatOpenAI(
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.7,
    )


def build_chain():
    """
    Build a simple LangChain LCEL chain.
    Structure: prompt messages → LLM → string output parser.

    This module is the designated integration point for the RAG pipeline.
    When RAG is added, the retriever and context injection will be
    incorporated here without modifying any other module.
    """
    llm = build_llm()
    parser = StrOutputParser()

    def invoke(user_message: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        return (llm | parser).invoke(messages)

    return invoke


# Module-level singleton — instantiated once on startup
conversation_chain = build_chain()
