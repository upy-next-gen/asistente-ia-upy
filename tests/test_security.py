"""
Security regression tests for vulnerabilities #2, #3, and #4.

These tests verify that:
- The centralized logger works correctly in both environments.
- Raw exceptions are never shown to users.
- CORS is not configured with a wildcard origin.
- File uploads are disabled.
"""

import logging
import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, ".chainlit", "config.toml")


# ===================================================================
# Vuln #2 — Logger module
# ===================================================================

class TestLoggerModule:
    """Verify the centralized logging module from core/utils/logger.py."""

    def test_get_logger_returns_logger(self):
        from core.utils.logger import get_logger

        logger = get_logger("test.security")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.security"

    def test_get_logger_is_idempotent(self):
        from core.utils.logger import get_logger

        logger_a = get_logger("test.idempotent")
        handler_count = len(logger_a.handlers)
        logger_b = get_logger("test.idempotent")
        assert logger_a is logger_b
        assert len(logger_b.handlers) == handler_count

    def test_dev_logger_creates_log_directory(self):
        from core.utils.logger import get_logger, _LOG_DIR

        get_logger("test.logdir")
        assert os.path.isdir(_LOG_DIR)

    def test_dev_logger_writes_to_file(self):
        from core.utils.logger import get_logger, _LOG_FILE

        logger = get_logger("test.filewrite")
        logger.info("security test log entry")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        assert os.path.isfile(_LOG_FILE)
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            contents = f.read()
        assert "security test log entry" in contents

    def test_logger_level_is_debug(self):
        from core.utils.logger import get_logger

        logger = get_logger("test.level")
        assert logger.level == logging.DEBUG


# ===================================================================
# Vuln #2 — Error message sanitization
# ===================================================================

class TestErrorSanitization:
    """Verify that app.py does not expose raw exceptions."""

    def test_no_raw_exception_in_error_message(self):
        """The except block in on_message must use a generic message."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()

        # The old vulnerable pattern should NOT exist
        assert 'f"Error al conectar con el servidor: {e}"' not in source, (
            "app.py still contains the raw exception pattern"
        )

    def test_generic_error_message_is_present(self):
        """The generic user-facing error message must be present."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()

        assert "Ocurrió un error temporal" in source

    def test_logger_exception_call_is_present(self):
        """logger.exception() must be called in the except block."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            source = f.read()

        assert "logger.exception(" in source


# ===================================================================
# Vuln #3 — CORS
# ===================================================================

class TestCorsConfiguration:
    """Verify that config.toml does not allow wildcard CORS origins."""

    def test_cors_no_wildcard(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        assert 'allow_origins = ["*"]' not in content, (
            "CORS is still configured with a wildcard origin"
        )

    def test_cors_has_explicit_origins(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        assert "allow_origins" in content
        # At least one HTTPS origin should be present
        assert "https://" in content


# ===================================================================
# Vuln #4 — File upload
# ===================================================================

class TestFileUploadConfiguration:
    """Verify that file uploads are disabled in config.toml."""

    def test_upload_is_disabled(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # The enabled = false line must exist after [features.spontaneous_file_upload]
        assert "enabled = false" in content, (
            "File upload is not disabled in config.toml"
        )

    def test_no_wildcard_mime_type(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        assert '"*/*"' not in content, (
            "config.toml still allows all MIME types"
        )

    def test_max_size_is_reasonable(self):
        """max_size_mb should be ≤ 20 MB."""
        import re

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        match = re.search(r"max_size_mb\s*=\s*(\d+)", content)
        assert match is not None, "max_size_mb not found in config.toml"
        assert int(match.group(1)) <= 20
