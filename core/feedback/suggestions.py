from core.clients import clients


class FeedbackService:
    def __init__(self):
        self._supabase = clients.supabase

    def save(self, content: str) -> None:
        if self._supabase is None:
            raise RuntimeError("Supabase no está configurado")
        self._supabase.table("sugerencias").insert(
            {"contenido": content},
        ).execute()   