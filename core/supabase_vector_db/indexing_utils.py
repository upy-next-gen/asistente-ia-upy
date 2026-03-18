import base64
import os

import numpy as np


def decode_embedding(b64_string: str) -> list[float]:
    raw = base64.b64decode(b64_string)
    return np.frombuffer(raw, dtype=np.int8).astype(np.float32).tolist()


def infer_document_type(file_name: str) -> str:
    _, extension = os.path.splitext(file_name)
    return extension.lower().lstrip(".") or "unknown"


def build_metadata_rows(
    file_names: list[str],
    version_tag: str = "current",
    category: str = "general",
) -> list[dict]:
    return [
        {
            "source_file_name": file_name,
            "document_type": infer_document_type(file_name),
            "version_tag": version_tag,
            "category": category,
        }
        for file_name in file_names
    ]
