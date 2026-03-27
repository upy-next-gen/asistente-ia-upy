import chainlit as cl
from chainlit.server import app
from core.utils.logger import get_logger
from langfuse import observe
from perplexity import Perplexity
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from starlette.routing import Route

from core.config import settings
from core.prompts import PromptManager
from core.rag.retriever import DocumentRetriever
from core.rag.context import ContextBuilder
from core.llm.chat import ChatService
from core.feedback.suggestions import FeedbackService
from core.observability.tracing import TracingManager
from core.security.rate_limiter import RateLimiter
from core.supabase_vector_db.indexing_utils import decode_embedding
from core.auth.entra_id import get_entra_client, EntraIDConfigError, EntraIDAuthError

logger = get_logger(__name__)

retriever = DocumentRetriever()
chat_service = ChatService()
feedback_service = FeedbackService()
rate_limiter = RateLimiter()
pplx_client = Perplexity(api_key=settings.PERPLEXITY_API_KEY)


# ────────────────────────────────────────────────────────────────
# Auth routes (Entra ID OAuth2)
# Routes must be inserted BEFORE Chainlit's SPA catch-all handler
# so they are matched first by Starlette's router.
# ────────────────────────────────────────────────────────────────


async def auth_login(request: Request) -> JSONResponse:
    """Build the Entra ID authorization URL and return it as JSON.

    Returns a JSON error if Entra ID is not configured.
    """
    try:
        client = get_entra_client()
        auth_url = client.build_auth_url()
        return JSONResponse({"auth_url": auth_url})
    except EntraIDConfigError as exc:
        logger.warning("Entra ID not configured: %s", exc)
        return JSONResponse(
            {"error": str(exc)},
            status_code=503,
        )


async def auth_callback(request: Request):
    """Handle the OAuth2 callback from Microsoft Entra ID.

    Exchanges the authorization code for tokens, extracts user
    identity, and returns a success page that closes itself.
    """
    code = request.query_params.get("code")
    error = request.query_params.get("error")

    if error:
        error_desc = request.query_params.get("error_description", error)
        logger.warning("Entra callback error: %s", error_desc)
        return HTMLResponse(
            _auth_result_page(success=False, message=error_desc),
            status_code=400,
        )

    if not code:
        return HTMLResponse(
            _auth_result_page(success=False, message="No se recibió código de autorización."),
            status_code=400,
        )

    try:
        client = get_entra_client()
        tokens = await client.exchange_code_for_tokens(code)
        id_token = tokens.get("id_token", "")

        if not id_token:
            return HTMLResponse(
                _auth_result_page(success=False, message="No se recibió id_token."),
                status_code=400,
            )

        user = client.extract_user_from_id_token(id_token)
        logger.info("User authenticated: %s (%s)", user.name, user.email)

        return HTMLResponse(
            _auth_result_page(
                success=True,
                message=f"¡Bienvenido, {user.name}! ({user.email})",
            )
        )

    except (EntraIDConfigError, EntraIDAuthError) as exc:
        logger.error("Auth callback failed: %s", exc)
        return HTMLResponse(
            _auth_result_page(success=False, message=str(exc)),
            status_code=500,
        )


def _auth_result_page(success: bool, message: str) -> str:
    """Generate a minimal HTML page shown inside the auth popup."""
    icon = "✅" if success else "❌"
    color = "#4ade80" if success else "#f87171"
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head><meta charset="UTF-8"><title>Autenticación</title>
    <style>
        body {{ font-family: 'Outfit', 'Segoe UI', sans-serif;
               display: flex; align-items: center; justify-content: center;
               min-height: 100vh; margin: 0;
               background: hsl(270, 47%, 7%); color: #f2f2f2; }}
        .card {{ text-align: center; padding: 2rem; }}
        .icon {{ font-size: 3rem; }}
        p {{ margin-top: 1rem; color: {color}; }}
        small {{ opacity: 0.5; }}
    </style></head>
    <body><div class="card">
        <div class="icon">{icon}</div>
        <p>{message}</p>
        <small>Esta ventana se cerrará automáticamente…</small>
    </div>
    <script>setTimeout(()=>window.close(), 3000);</script>
    </body></html>
    """


# Insert auth routes BEFORE Chainlit's SPA catch-all so they match first
app.routes.insert(0, Route("/auth/login", auth_login, methods=["GET"]))
app.routes.insert(1, Route("/auth/callback", auth_callback, methods=["GET"]))


def embed_query(query_text: str) -> list[float]:
    response = pplx_client.contextualized_embeddings.create(
        input=[[query_text]],
        model=settings.EMBEDDING_MODEL,
    )
    return decode_embedding(response.data[0].data[0].embedding)


def _trim_history(history: list[dict]) -> list[dict]:
    system = history[:1]
    conversation = history[1:]
    if len(conversation) > settings.MAX_HISTORY_MESSAGES:
        conversation = conversation[-settings.MAX_HISTORY_MESSAGES:]
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
        if not content:
            return
        if len(content) > settings.MAX_SUGGESTION_LENGTH:
            await cl.Message(
                content=f"Tu sugerencia excede el límite de {settings.MAX_SUGGESTION_LENGTH} caracteres. Por favor, resúmela.",
            ).send()
            return
        try:
            feedback_service.save(content)
            await cl.Message(
                content="Gracias! Tu sugerencia fue guardada correctamente.",
            ).send()
        except Exception:
            logger.exception("Error al guardar sugerencia")
            await cl.Message(
                content="Hubo un error al guardar tu sugerencia. Inténtalo de nuevo más tarde.",
            ).send()


@cl.on_message
@observe(name="rag_pipeline")
async def on_message(message: cl.Message):
    session_id = cl.context.session.id

    if not rate_limiter.is_allowed(session_id):
        await cl.Message(
            content="Has enviado demasiados mensajes. Espera un momento antes de intentar de nuevo.",
        ).send()
        return

    if len(message.content) > settings.MAX_MESSAGE_LENGTH:
        await cl.Message(
            content=f"Tu mensaje excede el límite de {settings.MAX_MESSAGE_LENGTH} caracteres. Por favor, acórtalo.",
        ).send()
        return

    message_history = cl.user_session.get("message_history")

    TracingManager.update_trace(
        session_id=session_id,
        metadata={"query": message.content},
    )

    query_embedding = embed_query(message.content)
    rows = retriever.retrieve(query_embedding, top_k=settings.RETRIEVER_TOP_K)
    context = ContextBuilder.build(rows)
    user_message = PromptManager.build_user_message(message.content, context)

    TracingManager.score("retrieval_nodes", float(len(rows)))
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

    except Exception:
        logger.exception("Error en el pipeline de on_message")
        msg.content = "Ocurrió un error temporal. Inténtalo de nuevo en unos minutos."
        await msg.update()
        