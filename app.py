import os
import base64

import numpy as np
import chromadb
from langfuse.openai import AsyncOpenAI
from supabase import create_client
from perplexity import Perplexity
import chainlit as cl
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

# Inicializamos el cliente de DeepSeek (async para streaming)
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# Inicializar Supabase
supabase = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", ""),
)

# Cliente de Perplexity para embeddings de queries
pplx_client = Perplexity(api_key=os.getenv("PERPLEXITY_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Cargar colección ChromaDB con embeddings de Perplexity
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection(
    name="upy_docs_pplx",
    metadata={"hnsw:space": "cosine"},
)


def decode_embedding(b64_string: str) -> list[float]:
    """Decodificar embedding base64 int8 a lista de floats."""
    raw = base64.b64decode(b64_string)
    return np.frombuffer(raw, dtype=np.int8).astype(np.float32).tolist()


def embed_query(query_text: str) -> list[float]:
    """Generar embedding para una consulta usando Perplexity API."""
    response = pplx_client.contextualized_embeddings.create(
        input=[[query_text]],
        model=EMBEDDING_MODEL,
    )
    return decode_embedding(response.data[0].data[0].embedding)

# System prompt con instrucciones para usar el contexto de documentos
SYSTEM_PROMPT = (
    "Eres el asistente virtual oficial de la Universidad Politécnica de Yucatán (UPY). "
    "Tu ÚNICA tarea es responder preguntas estrictamente relacionadas con la universidad, "
    "trámites, carreras, horarios, vida estudiantil y temas académicos de la institución. "
    "REGLA INQUEBRANTABLE: Si el usuario te pregunta sobre cualquier otro tema fuera de la UPY "
    "(recetas, política, código de programación general, chistes, etc.), debes negarte cortésmente "
    "diciendo exactamente esto: 'Lo siento, como asistente de la UPY, mi conocimiento y funciones "
    "están limitados a temas de la universidad. ¿Te puedo ayudar con alguna duda sobre inscripciones "
    "o carreras?'. No des ninguna otra información.\n\n"
    "IMPORTANTE: Se te proporcionará contexto de documentos oficiales de la UPY. "
    "Usa esta información para dar respuestas precisas y fundamentadas. "
    "Si la información del contexto no es suficiente para responder, indícalo honestamente."
)

@cl.set_starters
async def set_starters():
    """Chips de sugerencia iniciales para guiar al usuario."""
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
    """Inicializar la sesión con el historial de mensajes."""
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": SYSTEM_PROMPT}],
    )


@cl.action_callback("abrir_sugerencias")
async def handle_feedback_action(action: cl.Action):
    """Manejar el clic en el botón de sugerencias."""
    
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
        sugerencia = res["output"].strip()
        if sugerencia:
            try:
                supabase.table("sugerencias").insert(
                    {"contenido": sugerencia}
                ).execute()
                await cl.Message(
                    content="Gracias! Tu sugerencia fue guardada correctamente.",
                ).send()
            except Exception:
                await cl.Message(
                    content="Hubo un error al guardar tu sugerencia. Inténtalo de nuevo más tarde.",
                ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Procesar cada mensaje del usuario con RAG + DeepSeek (streaming)."""
    message_history = cl.user_session.get("message_history")

    # Generar embedding de la consulta y buscar en ChromaDB
    query_embedding = embed_query(message.content)
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas"],
    )

    context = ""
    if results and results["documents"] and results["documents"][0]:
        context_parts = []
        for doc_text, metadata in zip(results["documents"][0], results["metadatas"][0]):
            source = metadata.get("file_name", "documento")
            context_parts.append(f"[Fuente: {source}]\n{doc_text}")
        context = "\n\n---\n\n".join(context_parts)

    # Construir el mensaje del usuario con contexto
    if context:
        user_message = (
            f"Contexto de documentos oficiales de la UPY:\n\n{context}\n\n"
            f"---\n\nPregunta del usuario: {message.content}"
        )
    else:
        user_message = message.content

    message_history.append({"role": "user", "content": user_message})

    msg = cl.Message(content="")
    await msg.send()

    try:
        stream = await client.chat.completions.create(
            model="deepseek-chat",
            messages=message_history,
            stream=True,
        )

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
