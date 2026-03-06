# Asistente IA UPY

Asistente virtual de la Universidad Politécnica de Yucatán basado en RAG (Retrieval-Augmented Generation) con Chainlit, DeepSeek y ChromaDB.

## Arquitectura

El proyecto sigue una arquitectura modular orientada a objetos. El punto de entrada (`app.py`) actúa como capa delgada que orquesta los módulos de `core/`.

```
asistente-ia-upy/
│   app.py                          # Punto de entrada Chainlit (hooks y orquestación)
│   Dockerfile
│   pyproject.toml
│   uv.lock
│
├── core/
│   ├── config.py                   # Configuración centralizada (dataclass + env vars)
│   ├── clients.py                  # Clientes externos (DeepSeek, Supabase, ChromaDB)
│   ├── prompts.py                  # System prompt y templates
│   │
│   ├── rag/
│   │   ├── retriever.py            # Búsqueda vectorial en ChromaDB
│   │   └── context.py              # Construcción del contexto RAG
│   │
│   ├── llm/
│   │   └── chat.py                 # Llamada al LLM con streaming y timeout
│   │
│   ├── feedback/
│   │   └── suggestions.py          # Buzón de sugerencias (Supabase)
│   │
│   ├── observability/
│   │   └── tracing.py              # Trazabilidad con Langfuse v3
│   │
│   └── security/
│       └── rate_limiter.py         # Rate limiting por sesión
│
├── scripts/
│   └── index_docs.py               # Indexación de documentos en ChromaDB
│
├── tests/                          # Unit tests (24/24 passing)
│   ├── test_config.py
│   ├── test_clients.py
│   ├── test_prompts.py
│   ├── test_context.py
│   ├── test_retriever.py
│   ├── test_chat.py
│   ├── test_feedback.py
│   └── test_tracing.py
│
├── docs/                           # PDFs para indexación RAG
├── public/                         # Assets de UI (logos, CSS, iconos)
└── .chainlit/
    └── config.toml                 # Configuración de Chainlit
```

## Pipeline RAG

El flujo de cada mensaje del usuario sigue estos pasos:

1. **Validación** — se verifica que el mensaje no exceda el límite de caracteres y que la sesión no haya superado el rate limit.
2. **Retrieval** (`core/rag/retriever.py`) — el mensaje se convierte en embedding con `BAAI/bge-small-en-v1.5` y se buscan los 5 fragmentos más similares en ChromaDB.
3. **Context Building** (`core/rag/context.py`) — los fragmentos recuperados se formatean con su fuente de origen.
4. **Prompt Assembly** (`core/prompts.py`) — el contexto RAG se inyecta junto con la pregunta del usuario en el prompt.
5. **LLM Generation** (`core/llm/chat.py`) — se envía el historial completo (truncado) a DeepSeek con streaming.
6. **Observabilidad** — cada etapa genera trazas y scores en Langfuse.

## Manejo de Múltiples Peticiones

El sistema implementa múltiples capas de protección contra peticiones masivas o abuso:

### Rate Limiting por Sesión

**Archivo:** `core/security/rate_limiter.py`

Cada sesión de Chainlit tiene un identificador único (`cl.context.session.id`) asignado automáticamente por WebSocket. El `RateLimiter` mantiene un diccionario en memoria con los timestamps de mensajes por sesión. Antes de procesar cualquier mensaje, se verifica:

- Se limpian los timestamps fuera de la ventana de tiempo (default: 60 segundos).
- Si la sesión ya alcanzó el máximo de requests en esa ventana (default: 10), el mensaje se rechaza sin procesarlo.
- Si está dentro del límite, se registra el timestamp y se continúa.

Esto protege los créditos de DeepSeek contra usuarios que bombardeen el chatbot, ya sea manualmente o con scripts automatizados. Los valores son configurables vía variables de entorno `RATE_LIMIT_WINDOW` y `RATE_LIMIT_MAX_REQUESTS`.

### Tope al Historial de Mensajes

**Archivo:** `app.py` (función `_trim_history`)

Cada mensaje enviado al LLM incluye el historial completo de la conversación. Sin límite, una conversación larga generaría llamadas progresivamente más caras y lentas. La función `_trim_history` preserva siempre el system prompt y recorta la conversación a los últimos N mensajes (default: 20, equivalente a 10 intercambios usuario-asistente). Configurable vía `MAX_HISTORY_MESSAGES`.

### Validación de Inputs

**Archivo:** `app.py`

Se valida la longitud de cada mensaje del usuario (default: 2000 caracteres) y de cada sugerencia del buzón (default: 1000 caracteres). Los mensajes que excedan el límite se rechazan con un aviso al usuario, sin consumir tokens del LLM ni espacio en Supabase. Configurable vía `MAX_MESSAGE_LENGTH` y `MAX_SUGGESTION_LENGTH`.

### Timeout en la Llamada al LLM

**Archivo:** `core/llm/chat.py`

La llamada a DeepSeek tiene un timeout de 30 segundos (configurable vía `LLM_TIMEOUT`). Si el servidor no responde en ese tiempo, `asyncio.wait_for` lanza un `TimeoutError` que se captura y muestra al usuario como error de conexión. Esto evita que conexiones colgadas consuman recursos del servidor indefinidamente.

### Identificación de Sesiones

Chainlit asigna un UUID único por conexión WebSocket (por pestaña del navegador). Esta identificación es anónima y se usa tanto para el rate limiting como para las trazas en Langfuse. No requiere autenticación. Si se necesita identificación por persona (no por pestaña), Chainlit soporta OAuth o login con contraseña como capa adicional.

### Limitaciones Actuales

- **Concurrencia global:** No hay semáforo que limite el número total de llamadas simultáneas a DeepSeek. En un escenario con muchos usuarios concurrentes, el rate limit de la API de DeepSeek sería el único tope.
- **Embedding model:** Corre en CPU single-threaded. Con alta concurrencia, las búsquedas vectoriales se encolan.
- **Rate limiter en memoria:** Al reiniciar el contenedor, los contadores se pierden. Para producción con múltiples réplicas se requeriría Redis.

## Observabilidad (Langfuse)

**Archivos:** `core/observability/tracing.py`, decoradores `@observe` en los módulos

El SDK de Langfuse v3 (basado en OpenTelemetry) traza el pipeline completo:

- `@observe(name="rag_pipeline")` en `on_message` crea la traza padre.
- `@observe(name="document_retrieval")` en `DocumentRetriever.retrieve()` traza la búsqueda vectorial.
- `@observe(name="context_building")` en `ContextBuilder.build()` traza la construcción del contexto.
- `@observe(name="llm_stream")` en `ChatService.stream()` traza la llamada al LLM.

Adicionalmente, `TracingManager` registra scores numéricos (`retrieval_nodes`, `context_length`) y metadata de sesión en cada traza.

## Variables de Entorno

| Variable | Descripción | Default |
|---|---|---|
| `DEEPSEEK_API_KEY` | API key de DeepSeek | — |
| `LLM_BASE_URL` | Base URL del LLM | `https://api.deepseek.com` |
| `EMBEDDING_MODEL` | Modelo de embeddings | `BAAI/bge-small-en-v1.5` |
| `SUPABASE_URL` | URL de Supabase (opcional) | — |
| `SUPABASE_KEY` | Key de Supabase (opcional) | — |
| `LANGFUSE_SECRET_KEY` | Secret key de Langfuse | — |
| `LANGFUSE_PUBLIC_KEY` | Public key de Langfuse | — |
| `LANGFUSE_BASE_URL` | Host de Langfuse | `https://cloud.langfuse.com` |
| `MAX_HISTORY_MESSAGES` | Máximo de mensajes en historial | `20` |
| `MAX_MESSAGE_LENGTH` | Máximo de caracteres por mensaje | `2000` |
| `MAX_SUGGESTION_LENGTH` | Máximo de caracteres por sugerencia | `1000` |
| `LLM_TIMEOUT` | Timeout de la llamada al LLM (segundos) | `30` |
| `RATE_LIMIT_WINDOW` | Ventana de rate limit (segundos) | `60` |
| `RATE_LIMIT_MAX_REQUESTS` | Máximo de requests por ventana | `10` |

## Ejecución

### Requisitos

- Docker
- Archivos PDF en `docs/` para indexación RAG

### Build y Run

```bash
docker build -t asistente-upy .
docker run -p 8000:8000 --env-file .env asistente-upy
```

La app estará disponible en `http://localhost:8000`.

### Tests

```bash
docker run --rm asistente-upy sh -c "uv pip install pytest pytest-asyncio && pytest tests/ -v"
```

## Stack Tecnológico

- **Frontend:** Chainlit
- **LLM:** DeepSeek (API compatible con OpenAI)
- **Embeddings:** BAAI/bge-small-en-v1.5 (HuggingFace, local)
- **Vector Store:** ChromaDB (persistente, local)
- **Observabilidad:** Langfuse v3 (OpenTelemetry)
- **Feedback:** Supabase (opcional)
- **Gestión de dependencias:** uv
- **Contenedorización:** Docker
