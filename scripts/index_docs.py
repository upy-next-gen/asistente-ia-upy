"""
Script para indexar documentos PDF del directorio docs/ en ChromaDB.

Uso:
    uv run python scripts/index_docs.py

Requiere OPENAI_API_KEY en .env (usa DeepSeek para embeddings).
"""

import os
import sys
import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# Rutas
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def main():
    # Verificar que existan documentos
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        print("No se encontraron documentos en docs/. Agrega PDFs y vuelve a ejecutar.")
        sys.exit(1)

    print(f"Cargando documentos de {DOCS_DIR}...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print(f"  {len(documents)} fragmentos cargados de {len(set(d.metadata.get('file_name', '') for d in documents))} archivos.")

    # Configurar embeddings locales (gratis, sin API key)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    )

    # Inicializar ChromaDB persistente
    print(f"Creando índice vectorial en {CHROMA_DIR}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("upy_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Crear índice
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"Indexación completada. {len(documents)} fragmentos indexados en ChromaDB.")
    print(f"Base vectorial guardada en: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
