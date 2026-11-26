"""Shared pytest fixtures."""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

import pytest

from copilot.config.settings import get_settings
from tests.fixtures.synthetic_repo import build_synthetic_repo


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch):
    """Make every test independent of the developer's real environment and .env file."""
    for name in list(os.environ):
        if name.upper().startswith("COPILOT_") or name.upper() == "GEMINI_API_KEY":
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _restore_copilot_logging():
    """Undo ``setup_logging`` side effects after every test.

    ``setup_logging`` installs a handler on the global ``copilot`` logger bound to whatever stream
    was ``sys.stderr`` at that moment (pytest's capture stream). Left in place, it makes later tests
    print "--- Logging error ---" once that stream is closed, so the outcome of a test depended on
    which tests had run before it (seen with ``pytest -k`` selections of ``test_rag_cli.py``).
    """
    logger = logging.getLogger("copilot")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


_LOCAL_HOSTS = {None, "", "localhost", "127.0.0.1", "::1", "0.0.0.0"}  # noqa: S104


def _is_local(host: object) -> bool:
    return host in _LOCAL_HOSTS or (isinstance(host, str) and host.startswith("127."))


@pytest.fixture(autouse=True)
def _no_outbound_network(request, monkeypatch: pytest.MonkeyPatch):
    """Fail any test that tries to reach a non-local host (DNS lookup or TCP connect).

    This is a provider-boundary safeguard: whatever mock a test forgot, a hosted-LLM call or a
    model download cannot happen silently. It is checked at teardown, so a library that swallows
    the connection error (the Gemini client turns it into ``LLMConnectionError``) still fails the
    test. Opt-in live tests (markers ``live`` and ``llm_live``) are exempt.
    """
    attempts: list[str] = []  # a test may request this fixture, inspect it and clear it
    if "live" in request.keywords or "llm_live" in request.keywords:
        yield attempts
        return
    real_getaddrinfo, real_connect = socket.getaddrinfo, socket.socket.connect

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            attempts.append(f"DNS lookup of {host!r}")
            raise OSError("outbound network access is blocked in tests")
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else None
        if isinstance(address, tuple) and not _is_local(host):
            attempts.append(f"connect to {host!r}")
            raise OSError("outbound network access is blocked in tests")
        return real_connect(self, address)

    # A system proxy (HTTPS_PROXY=http://localhost:3128, common in CI and sandboxes) makes the
    # socket layer see only a loopback connection, so the HTTP client is guarded as well.
    import httpx

    real_send, real_async_send = httpx.Client.send, httpx.AsyncClient.send

    def guarded_send(self, request, *args, **kwargs):
        if not _is_local(request.url.host):
            attempts.append(f"HTTP request to {request.url.host!r}")
            raise httpx.ConnectError("outbound network access is blocked in tests")
        return real_send(self, request, *args, **kwargs)

    async def guarded_async_send(self, request, *args, **kwargs):
        if not _is_local(request.url.host):
            attempts.append(f"HTTP request to {request.url.host!r}")
            raise httpx.ConnectError("outbound network access is blocked in tests")
        return await real_async_send(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", guarded_send)
    monkeypatch.setattr(httpx.AsyncClient, "send", guarded_async_send)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield attempts
    assert not attempts, (
        f"test attempted outbound network access (hosted API or download): {attempts}"
    )


@pytest.fixture
def synthetic_repo(tmp_path: Path) -> Path:
    """A fresh synthetic repository (see tests/fixtures/synthetic_repo.py) in a temp directory."""
    return build_synthetic_repo(tmp_path / "synthetic-repo")


def pytest_collection_modifyitems(config, items):
    """Skip opt-in tests unless explicitly enabled.

    ``live`` (real embedding model): ``COPILOT_RUN_LIVE=1`` (the first run downloads ~0.64 GB).
    ``llm_live`` (hosted LLM call): ``COPILOT_RUN_LLM_LIVE=1``; the test itself also needs
    ``COPILOT_LLM_MODEL`` and ``GEMINI_API_KEY`` and skips with a clear reason without them.

    It also refuses oversized generated test ids (``check_test_id_lengths``).
    """
    check_test_id_lengths(items)
    skip_live = pytest.mark.skip(
        reason="live test: set COPILOT_RUN_LIVE=1 to run (downloads the model)"
    )
    skip_llm = pytest.mark.skip(
        reason="hosted-LLM test: set COPILOT_RUN_LLM_LIVE=1 (plus COPILOT_LLM_MODEL and "
        "GEMINI_API_KEY) to run"
    )
    for item in items:
        if "live" in item.keywords and os.environ.get("COPILOT_RUN_LIVE") != "1":
            item.add_marker(skip_live)
        if "llm_live" in item.keywords and os.environ.get("COPILOT_RUN_LLM_LIVE") != "1":
            item.add_marker(skip_llm)


# pytest stores the running test's id in the PYTEST_CURRENT_TEST environment variable, and Windows
# refuses values of 32767 characters or more (``ValueError`` from ``putenv``, which then shows up as
# setup/teardown errors). Parametrized string arguments become part of the id, so a large input must
# be given a short explicit ``id=``. Fail collection long before the OS limit is reached.
MAX_TEST_ID_CHARS = 8_000


def check_test_id_lengths(items: list[pytest.Item]) -> None:
    """Refuse oversized generated test ids (see ``MAX_TEST_ID_CHARS``)."""
    for item in items:
        if len(item.nodeid) > MAX_TEST_ID_CHARS:
            raise pytest.UsageError(
                f"test id of {len(item.nodeid)} characters (limit {MAX_TEST_ID_CHARS}) for "
                f"{item.nodeid[:100]!r}...: give the large parametrized argument a short id="
            )
