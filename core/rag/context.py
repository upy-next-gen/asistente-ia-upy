from langfuse import observe
from llama_index.core.schema import NodeWithScore


class ContextBuilder:
    @staticmethod
    @observe(name="context_building")
    def build(nodes: list[NodeWithScore]) -> str:
        if not nodes:
            return ""
        parts = []
        for node in nodes:
            source = node.metadata.get("file_name", "documento")
            parts.append(f"[Fuente: {source}]\n{node.text}")
        return "\n\n---\n\n".join(parts)


