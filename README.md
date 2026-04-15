# Asistente IA UPY

Asistente virtual de la Universidad Politécnica de Yucatán basado en RAG (Retrieval-Augmented Generation) con:
- **Frontend:** Chainlit
- **LLM:** DeepSeek (Claude alternative)
- **Vector Store:** Supabase pgvector (cloud PostgreSQL con extensión vector)
- **Embeddings:** Perplexity API (contextualized embeddings 1024-dim)
- **Observabilidad:** Langfuse v3 (OpenTelemetry)

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
│   ├── clients.py                  # Clientes externos (DeepSeek, Supabase, Perplexity)
│   ├── prompts.py                  # System prompt y templates
│   │
│   ├── supabase_vector_db/         # Módulo vector store (reemplazo de ChromaDB)
│   │   ├── __init__.py             # Exports públicos
│   │   ├── store.py                # SupabaseVectorStore (validación + RPC payloads)
│   │   ├── retriever.py            # DocumentRetriever (búsqueda vectorial)
│   │   ├── indexing_utils.py       # Utilidades (decode_embedding, infer_type)
│   │   └── sql/
│   │       └── setup_pgvector.sql  # Schema PostgreSQL + RPC function
│   │
│   ├── rag/
│   │   ├── retriever.py            # Orquestador RAG (usa SupabaseVectorRetriever)
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
│   ├── index_docs_supabase.py      # Indexación de documentos (PDFs → Perplexity → Supabase)
│   └── indexing_utils.py           # Utilidades compartidas (moved from vector_supabase/)
│
├── tests/                          # Unit tests (pytest)
│   ├── test_config.py
│   ├── test_clients.py
│   ├── test_prompts.py
│   ├── test_context.py
│   ├── test_retriever.py
│   ├── test_chat.py
│   ├── test_feedback.py
│   ├── test_tracing.py
│   ├── test_supabase_vector_store.py    # Tests para Supabase store
│   └── test_supabase_indexing_utils.py  # Tests para utilidades de indexación
│
├── docs/                           # PDFs para indexación RAG
├── public/                         # Assets de UI (logos, CSS, iconos)
│
└── .chainlit/
    └── config.toml                 # Configuración de Chainlit
```

## Pipeline RAG

El flujo de cada mensaje del usuario sigue estos pasos:

1. **Validación** — se verifica que el mensaje no exceda el límite de caracteres y que la sesión no haya superado el rate limit.
2. **Embedding** (`app.py`) — el mensaje se convierte en embedding con **Perplexity API** (`pplx-embed-context-v1-0.6b`, 1024-dim, contextualized).
3. **Retrieval** (`core/rag/retriever.py` + `core/supabase_vector_db/`) — se buscan los 5 fragmentos más similares en **Supabase pgvector** usando función RPC `match_documents` con Top-K + `min_similarity` (≥ 0.3).
4. **Context Building** (`core/rag/context.py`) — los fragmentos recuperados se formatean con su fuente de origen.
5. **Prompt Assembly** (`core/prompts.py`) — el contexto RAG se inyecta junto con la pregunta del usuario en el prompt.
6. **LLM Generation** (`core/llm/chat.py`) — se envía el historial completo (truncado) a DeepSeek con streaming.
7. **Observabilidad** — cada etapa genera trazas y scores en Langfuse.

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

### Timeout en la Llamada al LLM

**Archivo:** `core/llm/chat.py`

La llamada a DeepSeek tiene un timeout de 30 segundos (configurable vía `LLM_TIMEOUT`). Si el servidor no responde en ese tiempo, `asyncio.wait_for` lanza un `TimeoutError` que se captura y muestra al usuario como error de conexión. Esto evita que conexiones colgadas consuman recursos del servidor indefinidamente.

### Identificación de Sesiones

Chainlit asigna un UUID único por conexión WebSocket (por pestaña del navegador). Esta identificación es anónima y se usa tanto para el rate limiting como para las trazas en Langfuse. No requiere autenticación. Si se necesita identificación por persona (no por pestaña), Chainlit soporta OAuth o login con contraseña como capa adicional.

### Limitaciones Actuales

- **Concurrencia global:** No hay semáforo que limite el número total de llamadas simultáneas a DeepSeek. En un escenario con muchos usuarios concurrentes, el rate limit de la API de DeepSeek sería el único tope.
- **Embedding model:** Perplexity API tiene límites de rate. El batch size (200 documentos) está optimizado para no exceder cuotas.
- **Rate limiter en memoria:** Al reiniciar el contenedor, los contadores se pierden. Para producción con múltiples réplicas se requeriría Redis.

## Migración: ChromaDB → Supabase pgvector

El proyecto fue migrado de ChromaDB (local, HuggingFace embeddings) a **Supabase pgvector** (cloud, Perplexity contextualized embeddings) por:

- **Escalabilidad:** Supabase es cloud-managed, soporta múltiples replicas y backups automáticos.
- **Calidad de embeddings:** Perplexity API proporciona embeddings contextualizados (1024-dim) vs. embeddings genéricos locales (384-dim).
- **HNSW Index:** Búsqueda vectorial rápida (~O(log n)) con índice aproximado en PostgreSQL.
- **Top-K + Quality Filter:** Nueva estrategia de búsqueda flexible (Top-5 con `min_similarity ≥ 0.3`) en lugar de thresholds rígidos.

### Retrocompatibilidad

La carpeta `vector_supabase/` mantiene archivos de referencia (SQL schema, documentación) pero el código runtime usa `core/supabase_vector_db/`. Los tests en `tests/test_supabase_*.py` validan la nueva arquitectura.

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
| `PERPLEXITY_API_KEY` | API key de Perplexity (embeddings) | — |
| `EMBEDDING_MODEL` | Modelo de embeddings Perplexity | `pplx-embed-context-v1-0.6b` |
| `SUPABASE_URL` | URL de Supabase PostgreSQL | — |
| `SUPABASE_KEY` | Key de runtime del chatbot (usar **Secret key**) | — |
| `EMBEDDING_DIM` | Dimensionalidad de embeddings | `1024` |
| `SAFE_BATCH_SIZE` | Tamaño de batch para Perplexity API | `200` |
| `MAX_CHUNKS_PER_REQUEST` | Máximo de chunks a recuperar | `20` |
| `MIN_SIMILARITY` | Umbral mínimo de similitud (0-1) | `0.3` |
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

- Docker (o Python 3.9+ con `uv`)
- Cuenta Supabase activa
- API keys: Perplexity, DeepSeek, Supabase
- PDFs en `docs/` para indexación RAG

### Setup Rápido

1. **Crear `.env` con las credenciales:**
   ```bash
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-secret-key
   PERPLEXITY_API_KEY=your-perplexity-key
   DEEPSEEK_API_KEY=your-deepseek-key
   ```

2. **Crear schema PostgreSQL** (en Supabase SQL Editor):
   ```bash
   # Ejecutar: context_docs/setup_pgvector.sql
   ```

   Este script aplica RLS en tablas vectoriales, crea el rol fijo `chatbot_asker`
   para el flujo anónimo del chatbot y limita el acceso de consulta a la función RPC
   `match_documents`.

3. **Indexar documentos (solo backend con Secret key):**
   ```bash
   SUPABASE_KEY=your-secret-key uv run python scripts/index_docs_supabase.py
   ```

4. **Ejecutar chatbot (con Publishable key):**
   ```bash
   uv run chainlit run app.py
   # Abre http://localhost:8000
   ```

5. **Validar seguridad en Supabase (SQL Editor):**
   ```sql
   -- RLS activo
   select tablename, rowsecurity
   from pg_tables
   where schemaname = 'public'
     and tablename in ('documents_metadata', 'documents_embeddings');

   -- Politicas activas
   select schemaname, tablename, policyname, roles, cmd
   from pg_policies
   where schemaname = 'public'
     and tablename in ('documents_metadata', 'documents_embeddings');

   -- Permiso RPC para rol fijo
   select has_function_privilege(
     'chatbot_asker',
     'public.match_documents(vector,integer,double precision)',
     'EXECUTE'
   ) as chatbot_asker_can_execute_match_documents;
   ```

### Build y Run con Docker

```bash
docker build -t asistente-upy .
docker run -p 8000:8000 --env-file .env asistente-upy
```

### Tests

```bash
# Con uv
uv run pytest tests/ -v

# Con Docker
docker run --rm asistente-upy sh -c "pytest tests/ -v"
```

## Stack Tecnológico

| Componente | Tecnología | Notas |
|---|---|---|
| **Frontend** | Chainlit | Chat UI interactiva |
| **LLM** | DeepSeek (API) | Compatible con OpenAI API |
| **Embeddings** | Perplexity API | Contextualized embeddings 1024-dim (`pplx-embed-context-v1-0.6b`) |
| **Vector Store** | Supabase pgvector | PostgreSQL con extensión vector, índice HNSW |
| **Observabilidad** | Langfuse v3 | OpenTelemetry trazas y evals |
| **Feedback** | Supabase (optional) | Buzón de sugerencias |
| **Gestor de deps** | uv | Python package manager rápido |
| **Contenedorización** | Docker | Deployment en producción |
