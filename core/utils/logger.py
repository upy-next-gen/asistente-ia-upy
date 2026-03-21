"""
Centralized logging module for the Asistente UPY project.

Reads the ENV variable from the .env file to determine the environment:
  - "development" (default): Rich-colored console output + rotating file handler.
  - "production": Structured JSON to stdout/stderr for Google Cloud Run.

Usage:
    from core.utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
import sys
import json
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from core.config import settings

from dotenv import load_dotenv

load_dotenv()

ENV = settings.ENV.lower()

# ---------------------------------------------------------------------------
# Paths (only used in development)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "app.log")

_DEV_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)s:%(lineno)d │ %(message)s"


# ---------------------------------------------------------------------------
# JSON formatter for production (Cloud Run)
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    """Produces a single-line JSON object per log record.

    Google Cloud's operations suite parses the ``severity`` field
    automatically when the log line is written to stdout / stderr.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": self.formatMessage(record),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def _build_dev_handlers() -> list[logging.Handler]:
    """Rich console handler + rotating file handler."""
    handlers: list[logging.Handler] = []

    # --- Rich console ---
    try:
        from rich.logging import RichHandler

        rich_handler = RichHandler(
            level=logging.DEBUG,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=True,
            show_path=True,
        )
        handlers.append(rich_handler)
    except ImportError:
        # Fallback if rich is not installed
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter(_DEV_FORMAT))
        handlers.append(console)

    # --- Rotating file ---
    os.makedirs(_LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_DEV_FORMAT))
    handlers.append(file_handler)

    return handlers


def _build_prod_handlers() -> list[logging.Handler]:
    """JSON stdout (INFO+) and stderr (ERROR+) for Cloud Run."""
    json_fmt = _JsonFormatter()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda r: r.levelno < logging.ERROR)
    stdout_handler.setFormatter(json_fmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(json_fmt)

    return [stdout_handler, stderr_handler]


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    The logger is configured only once; subsequent calls with the
    same *name* return the already-configured instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    handlers = (
        _build_dev_handlers() if ENV != "production" else _build_prod_handlers()
    )
    for handler in handlers:
        logger.addHandler(handler)

    return logger
