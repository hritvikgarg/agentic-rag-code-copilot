"""Logging configuration using only the standard library.

Design points:
* All project loggers live under the ``copilot`` namespace (use ``logging.getLogger(__name__)``).
* ``setup_logging`` is idempotent: calling it again replaces our handler instead of duplicating it.
* The formatter masks known secret values (e.g. the Gemini key) in every log line, including
  exception text, as a last line of defence. The primary rule is still: never log secrets.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TextIO

from copilot.config.settings import Settings, get_settings

LOGGER_NAME = "copilot"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
REDACTED = "***REDACTED***"
_MIN_SECRET_LENGTH = 8  # avoid masking short, common substrings by accident


class RedactingFormatter(logging.Formatter):
    """A formatter that replaces known secret values with a fixed marker."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        secrets: Iterable[str] = (),
    ) -> None:
        super().__init__(fmt, datefmt)
        # Longest first so a secret that contains another is fully masked.
        self._secrets = tuple(
            sorted({s for s in secrets if len(s) >= _MIN_SECRET_LENGTH}, key=len, reverse=True)
        )

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        return text


class _CopilotHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Marker subclass so we only ever remove handlers that we installed ourselves."""


def setup_logging(settings: Settings | None = None, stream: TextIO | None = None) -> logging.Logger:
    """Configure and return the ``copilot`` logger.

    Args:
        settings: Settings to use; defaults to ``get_settings()``.
        stream: Output stream; defaults to ``sys.stderr``.
    """
    settings = settings or get_settings()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(settings.log_level)
    logger.propagate = False  # we own the output format; avoid duplicate root-handler lines

    for handler in list(logger.handlers):
        if isinstance(handler, _CopilotHandler):
            logger.removeHandler(handler)

    handler = _CopilotHandler(stream)
    handler.setFormatter(RedactingFormatter(LOG_FORMAT, secrets=settings.secret_values()))
    logger.addHandler(handler)
    return logger
