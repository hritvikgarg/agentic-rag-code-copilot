"""Logging setup: level, idempotency, and secret redaction."""

import io
import logging

from copilot.config import Settings, setup_logging
from copilot.config.logging_setup import _CopilotHandler


def test_setup_sets_level_and_writes_to_stream():
    stream = io.StringIO()
    logger = setup_logging(Settings(_env_file=None, log_level="WARNING"), stream=stream)
    child = logging.getLogger("copilot.tests.child")
    child.info("hidden")
    child.warning("visible")
    output = stream.getvalue()
    assert "visible" in output
    assert "hidden" not in output
    assert logger.level == logging.WARNING


def test_setup_is_idempotent():
    for _ in range(3):
        logger = setup_logging(Settings(_env_file=None), stream=io.StringIO())
    ours = [h for h in logger.handlers if isinstance(h, _CopilotHandler)]
    assert len(ours) == 1


def test_secrets_are_redacted_from_messages_and_exceptions():
    secret = "unit-test-not-a-real-key-123456"
    stream = io.StringIO()
    setup_logging(Settings(_env_file=None, gemini_api_key=secret), stream=stream)
    log = logging.getLogger("copilot.tests.redaction")
    log.error("request failed with key=%s", secret)
    try:
        raise RuntimeError(f"boom {secret}")
    except RuntimeError:
        log.exception("unexpected")
    output = stream.getvalue()
    assert secret not in output
    assert "***REDACTED***" in output
