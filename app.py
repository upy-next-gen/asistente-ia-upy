import os
import chromadb
from langfuse.openai import AsyncOpenAI
from supabase import create_client
import chainlit as cl
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

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

# Configurar embeddings locales (HuggingFace, gratis)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
)

# Cargar índice vectorial desde ChromaDB
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection("upy_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever(similarity_top_k=5)

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

    # Buscar contexto relevante en los documentos
    nodes = retriever.retrieve(message.content)
    context = ""
    if nodes:
        context_parts = []
        for node in nodes:
            source = node.metadata.get("file_name", "documento")
            context_parts.append(f"[Fuente: {source}]\n{node.text}")
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
