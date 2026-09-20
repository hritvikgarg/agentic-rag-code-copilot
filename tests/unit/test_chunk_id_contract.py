"""The chunk-id contract (docs/chunking.md): what does and does not change a chunk's identity."""

import hashlib
import json

import pytest

from copilot.chunking import LineChunker
from copilot.models.chunk import CHUNK_ID_SCHEMA, make_chunk_id
from tests.chunking_helpers import make_file

SOURCE = "".join(f"line {i}\n" for i in range(1, 31))  # 30 lines


def chunker(size=10, overlap=3, max_tokens=512):
    return LineChunker(size_lines=size, overlap_lines=overlap, max_tokens=max_tokens)


def ids(file, c=None):
    return [ch.chunk_id for ch in (c or chunker()).chunk_file(file)]


BASE = {
    "repository_name": "repo",
    "file_path": "src/a.py",
    "source_sha256": "a" * 64,
    "strategy": "line",
    "strategy_version": 1,
    "params": {"size_lines": 10, "overlap_lines": 3, "max_tokens": 512},
    "start_line": 1,
    "end_line": 10,
    "fragment_index": None,
}


def test_formula_matches_an_independent_implementation_of_the_documented_serialisation():
    payload = [
        "chunk-id/1", "repo", "src/a.py", "a" * 64, "line", 1,
        {"max_tokens": 512, "overlap_lines": 3, "size_lines": 10},
        ["w", 1, 10],
    ]  # fmt: skip
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    assert make_chunk_id(**BASE) == hashlib.sha256(text.encode()).hexdigest()[:16]
    assert CHUNK_ID_SCHEMA == "chunk-id/1"


def test_fragment_unit_serialisation():
    payload = [
        "chunk-id/1", "repo", "src/a.py", "a" * 64, "line", 1,
        {"max_tokens": 512, "overlap_lines": 3, "size_lines": 10},
        ["f", 7, 2],
    ]  # fmt: skip
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(text.encode()).hexdigest()[:16]
    got = make_chunk_id(**{**BASE, "start_line": 7, "end_line": 7, "fragment_index": 2})
    assert got == expected


def test_golden_id_locks_the_contract():
    """If this changes, the id contract changed: bump CHUNK_ID_SCHEMA and update the docs."""
    assert make_chunk_id(**BASE) == "9345bfe13c523d59"


def test_param_order_does_not_matter():
    reordered = {**BASE, "params": dict(reversed(list(BASE["params"].items())))}
    assert make_chunk_id(**reordered) == make_chunk_id(**BASE)


@pytest.mark.parametrize(
    ("field", "other"),
    [
        ("repository_name", "repo2"),
        ("file_path", "src/b.py"),
        ("source_sha256", "b" * 64),
        ("strategy", "ast"),
        ("strategy_version", 2),
        ("params", {"size_lines": 10, "overlap_lines": 3, "max_tokens": 511}),
        ("params", {"size_lines": 11, "overlap_lines": 3, "max_tokens": 512}),
        ("params", {"size_lines": 10, "overlap_lines": 4, "max_tokens": 512}),
        ("start_line", 2),
        ("end_line", 11),
        ("fragment_index", 0),
    ],
)
def test_every_identity_input_changes_the_id(field, other):
    assert make_chunk_id(**{**BASE, field: other}) != make_chunk_id(**BASE)


def test_serialisation_is_unambiguous_across_field_boundaries():
    a = make_chunk_id(**{**BASE, "repository_name": "a", "file_path": "b/c.py"})
    b = make_chunk_id(**{**BASE, "repository_name": "a\x00b", "file_path": "c.py"})
    c = make_chunk_id(**{**BASE, "repository_name": "a/b", "file_path": "c.py"})
    assert len({a, b, c}) == 3


# ---- the six required properties, exercised through the real chunker --------------------------
def test_A_same_source_strategy_and_config_gives_same_ids():
    assert ids(make_file(SOURCE)) == ids(make_file(SOURCE))


def test_B_different_repositories_with_identical_path_and_content_do_not_collide():
    one = make_file(SOURCE)
    other = one.model_copy(update={"repository_name": "another-repo"})
    assert set(ids(one)).isdisjoint(ids(other))


def test_C_changed_source_never_reuses_ids():
    old = make_file(SOURCE)
    new = make_file(SOURCE.replace("line 30", "line thirty"))  # edit only the very last line
    assert old.sha256 != new.sha256
    assert set(ids(old)).isdisjoint(ids(new))  # even chunks whose text is unchanged get new ids
    # chunk content hashes still show which chunks' text is unchanged
    old_hashes = {c.content_sha256 for c in chunker().chunk_file(old)}
    new_hashes = {c.content_sha256 for c in chunker().chunk_file(new)}
    assert old_hashes & new_hashes


def test_D_different_strategies_cannot_collide():
    class Other(LineChunker):
        @property
        def name(self):
            return "other"

    file = make_file(SOURCE)
    assert set(ids(file)).isdisjoint(
        ids(file, Other(size_lines=10, overlap_lines=3, max_tokens=512))
    )


@pytest.mark.parametrize(
    "other",
    [
        chunker(size=11),
        chunker(overlap=2),
        chunker(max_tokens=100),
    ],
)
def test_E_different_configuration_never_shares_ids(other):
    file = make_file(SOURCE)
    assert set(ids(file)).isdisjoint(ids(file, other))


def test_E_algorithm_version_bump_changes_ids(monkeypatch):
    file = make_file(SOURCE)
    before = ids(file)
    monkeypatch.setattr("copilot.chunking.line_chunker.STRATEGY_VERSION", 2)
    assert set(before).isdisjoint(ids(file))


def test_F_fragments_of_a_long_line_have_unique_deterministic_ids():
    line = "value = [" + ", ".join(["0"] * 300) + "]"  # 300 identical tokens => identical pieces
    file = make_file(f"a = 1\n{line}\nb = 2\n")
    c = chunker(size=10, overlap=2, max_tokens=40)
    first = c.chunk_file(file)
    fragments = [x for x in first if x.is_fragment]
    assert len(fragments) > 3
    assert len({x.chunk_id for x in fragments}) == len(fragments)
    assert len({x.content for x in fragments}) < len(fragments)  # some pieces have equal text
    assert [x.chunk_id for x in first] == [x.chunk_id for x in c.chunk_file(file)]
    assert [(x.fragment_index, x.fragment_count) for x in fragments] == [
        (i, len(fragments)) for i in range(len(fragments))
    ]


def test_ids_are_unique_within_a_file():
    got = ids(make_file(SOURCE))
    assert len(set(got)) == len(got)
