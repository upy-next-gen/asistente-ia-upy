import os
import sys

import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


class DocumentIndexer:
    def __init__(self, docs_dir: str, chroma_dir: str):
        self._docs_dir = docs_dir
        self._chroma_dir = chroma_dir
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def _validate(self) -> None:
        if not os.path.exists(self._docs_dir) or not os.listdir(self._docs_dir):
            print("No se encontraron documentos en docs/.")
            sys.exit(1)

    def _load(self) -> list:
        documents = SimpleDirectoryReader(self._docs_dir).load_data()
        file_count = len(set(d.metadata.get("file_name", "") for d in documents))
        print(f"{len(documents)} fragmentos cargados de {file_count} archivos.")
        return documents

    def _index(self, documents: list) -> None:
        chroma_client = chromadb.PersistentClient(path=self._chroma_dir)
        collection = chroma_client.get_or_create_collection("upy_docs")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )
        print(f"{len(documents)} fragmentos indexados en {self._chroma_dir}.")

    def run(self) -> None:
        self._validate()
        documents = self._load()
        self._index(documents)


if __name__ == "__main__":
    DocumentIndexer(DOCS_DIR, CHROMA_DIR).run()
