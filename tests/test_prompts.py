from core.prompts import PromptManager


class TestPromptManager:
    def test_system_prompt_exists(self):
        assert len(PromptManager.SYSTEM) > 0
        assert "UPY" in PromptManager.SYSTEM

    def test_build_user_message_with_context(self):
        result = PromptManager.build_user_message("¿Qué carreras hay?", "Contexto aquí")
        assert "Contexto aquí" in result
        assert "¿Qué carreras hay?" in result

    def test_build_user_message_without_context(self):
        result = PromptManager.build_user_message("¿Qué carreras hay?", "")
        assert result == "¿Qué carreras hay?"