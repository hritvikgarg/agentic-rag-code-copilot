"""Guards against oversized pytest ids (Windows limits one environment variable to 32,767 chars)."""

from __future__ import annotations

import types

import pytest

from tests.conftest import MAX_TEST_ID_CHARS, check_test_id_lengths

PARSER_TESTS = "tests/unit/test_traceback_parser"


def _item(nodeid: str) -> pytest.Item:
    return types.SimpleNamespace(nodeid=nodeid)  # type: ignore[return-value]


def test_the_limit_is_far_below_the_windows_environment_variable_limit():
    assert MAX_TEST_ID_CHARS < 32_767 // 2


def test_the_collection_hook_accepts_ids_up_to_the_limit_and_refuses_longer_ones():
    check_test_id_lengths([_item("t.py::a" + "x" * (MAX_TEST_ID_CHARS - len("t.py::a")))])
    with pytest.raises(pytest.UsageError, match="short id="):
        check_test_id_lengths([_item("t.py::a" + "x" * MAX_TEST_ID_CHARS)])


def test_the_refusal_message_does_not_echo_the_whole_id():
    huge = "t.py::a[" + "y" * 50_000 + "]"
    with pytest.raises(pytest.UsageError) as raised:
        check_test_id_lengths([_item(huge)])
    assert len(str(raised.value)) < 400


def test_traceback_parser_test_ids_stay_short(request: pytest.FixtureRequest):
    """Every collected parser test (when they are in this run) has a short, readable id."""
    ids = [i.nodeid for i in request.session.items if PARSER_TESTS in i.nodeid.replace("\\", "/")]
    assert all(len(nodeid) <= 200 for nodeid in ids), [n[:80] for n in ids if len(n) > 200]


def test_the_original_opt_in_hook_still_runs_the_guard_and_still_skips_live_tests(monkeypatch):
    """The id check lives inside the existing hook; the live-test skipping must be unchanged."""
    from tests import conftest

    monkeypatch.delenv("COPILOT_RUN_LIVE", raising=False)
    live = types.SimpleNamespace(nodeid="t.py::live", keywords={"live": True}, markers=[])
    live.add_marker = live.markers.append  # type: ignore[attr-defined]
    ok = types.SimpleNamespace(nodeid="t.py::ok", keywords={}, markers=[])
    ok.add_marker = ok.markers.append  # type: ignore[attr-defined]
    conftest.pytest_collection_modifyitems(config=None, items=[live, ok])
    assert len(live.markers) == 1 and ok.markers == []
    too_long = types.SimpleNamespace(nodeid="x" * (MAX_TEST_ID_CHARS + 1), keywords={})
    with pytest.raises(pytest.UsageError):
        conftest.pytest_collection_modifyitems(config=None, items=[too_long])
