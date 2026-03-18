from langfuse import observe


class ContextBuilder:
    @staticmethod
    @observe(name="context_building")
    def build(rows: list[dict]) -> str:
        if not rows:
            return ""
        parts = []
        for row in rows:
            source = row.get("source_file_name", "documento")
            text = row.get("content", "")
            parts.append(f"[Fuente: {source}]\n{text}")
        return "\n\n---\n\n".join(parts)


