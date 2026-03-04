import chainlit as cl
from langfuse import observe

from core.config import settings
from core.prompts import PromptManager
from core.rag.retriever import DocumentRetriever
from core.rag.context import ContextBuilder
from core.llm.chat import ChatService
from core.feedback.suggestions import FeedbackService
from core.observability.tracing import TracingManager

retriever = DocumentRetriever()
chat_service = ChatService()
feedback_service = FeedbackService()


def _trim_history(history: list[dict]) -> list[dict]:
    system = history[:1]
    conversation = history[1:]
    if len(conversation) > settings.max_history_messages:
        conversation = conversation[-settings.max_history_messages:]
    return system + conversation


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="¿Qué puedes hacer?",
            message="¿Qué puedes hacer?",
            icon="/public/question.svg",
        ),
        cl.Starter(
            label="Carreras disponibles",
            message="¿Qué carreras ofrece la UPY?",
            icon="/public/idea.svg",
        ),
        cl.Starter(
            label="Proceso de inscripción",
            message="¿Cómo me inscribo en la UPY?",
            icon="/public/help.svg",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": PromptManager.SYSTEM}],
    )


@cl.action_callback("abrir_sugerencias")
async def handle_feedback(action: cl.Action):
    res = await cl.AskUserMessage(
        content=(
            "🟣 **Modo Sugerencias Activado**\n\n"
            "Ahora estás enviando una sugerencia. "
            "¿Qué te gustaría que el Asistente UPY pudiera hacer? "
            "Ayúdanos a mejorar contándonos tus ideas o problemas."
        ),
        timeout=120,
    ).send()

    if res:
        content = res["output"].strip()
        if content:
            try:
                feedback_service.save(content)
                await cl.Message(
                    content="Gracias! Tu sugerencia fue guardada correctamente.",
                ).send()
            except Exception:
                await cl.Message(
                    content="Hubo un error al guardar tu sugerencia. Inténtalo de nuevo más tarde.",
                ).send()


@cl.on_message
@observe(name="rag_pipeline")
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")

    TracingManager.update_trace(
        session_id=cl.context.session.id,
        metadata={"query": message.content},
    )

    nodes = retriever.retrieve(message.content)
    context = ContextBuilder.build(nodes)
    user_message = PromptManager.build_user_message(message.content, context)

    TracingManager.score("retrieval_nodes", float(len(nodes)))
    TracingManager.score("context_length", float(len(context)))

    message_history.append({"role": "user", "content": user_message})
    message_history = _trim_history(message_history)
    cl.user_session.set("message_history", message_history)

    msg = cl.Message(content="")
    await msg.send()

    try:
        stream = await chat_service.stream(message_history)
        full_response = ""
        async for part in stream:
            if token := part.choices[0].delta.content:
                full_response += token
                await msg.stream_token(token)

        await msg.update()
        message_history.append({"role": "assistant", "content": full_response})
        cl.user_session.set("message_history", message_history)

        actions = [
            cl.Action(
                name="abrir_sugerencias",
                label="📬 Buzón de Sugerencias",
                tooltip="¿Qué te gustaría que el Asistente UPY pudiera hacer?",
                payload={"action": "feedback"},
            )
        ]
        await cl.Message(
            content="¿Tienes alguna sugerencia para mejorar el asistente?",
            actions=actions,
        ).send()

    except Exception as e:
        msg.content = f"Error al conectar con el servidor: {e}"
        await msg.update()