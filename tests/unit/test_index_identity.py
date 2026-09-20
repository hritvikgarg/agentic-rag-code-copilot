"""The deterministic index id: a hash of compatibility-relevant state and nothing else."""

import hashlib
import json

import pytest

from copilot.embeddings import HashEmbedder
from copilot.vectorstore import compute_index_id
from copilot.vectorstore.manifest import INDEX_ID_SCHEMA
from tests.vectorstore_helpers import build_test_index, make_chunker, make_provenance, spec_of

# Locked: id of the default test index (32-d hash embedder, prefixed, size 4 / overlap 1 / cap 512).
GOLDEN_ID = "19cc6462b92f5808"


def test_golden_id_is_locked():
    index, _, _ = build_test_index()
    assert index.index_id == GOLDEN_ID


def test_id_matches_the_documented_formula_written_out_by_hand():
    index, _, _ = build_test_index()
    spec = index.spec
    payload = {
        "schema": "index-id/1",
        "spec": {
            "repository_name": "repo",
            "repository_fingerprint": spec.repository_fingerprint,
            "chunking_strategy": "line",
            "chunking_strategy_version": 1,
            "chunk_params": {"size_lines": 4, "overlap_lines": 1, "max_tokens": 512},
            "chunk_id_schema": "chunk-id/1",
            "embedding_model_id": "fake/hash-embedder",
            "embedding_dimension": 32,
            "embedding_normalized": True,
            "text_style": "prefixed",
            "representation_version": 1,
            "index_type": "IndexFlatIP",
            "metric": "inner_product",
        },
    }
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    assert INDEX_ID_SCHEMA == "index-id/1"
    assert hashlib.sha256(text.encode()).hexdigest()[:16] == index.index_id


def test_same_inputs_same_identity_across_independent_builds():
    a, _, _ = build_test_index()
    b, _, _ = build_test_index()
    assert a.index_id == b.index_id
    assert a.spec == b.spec


def test_provenance_does_not_influence_the_id():
    index, _, _ = build_test_index()
    # Provenance (timestamp, runtime, faiss version) is informational: same spec, same id.
    other = make_provenance(created_at="2030-12-31T23:59:59Z", faiss_version="9.9.9")
    assert other != index.manifest.provenance
    assert compute_index_id(index.spec) == index.index_id


def test_id_is_16_lowercase_hex_characters():
    index, _, _ = build_test_index()
    assert len(index.index_id) == 16 and set(index.index_id) <= set("0123456789abcdef")


def test_source_change_changes_identity():
    base, _, _ = build_test_index()
    changed, _, _ = build_test_index({"src/a.py": "def a():\n    return 999\n"})
    assert changed.spec.repository_fingerprint != base.spec.repository_fingerprint
    assert changed.index_id != base.index_id


def test_chunk_configuration_change_changes_identity():
    base, _, _ = build_test_index()
    ids = {base.index_id}
    for chunker in (
        make_chunker(size=5),
        make_chunker(overlap=2),
        make_chunker(max_tokens=256),
    ):
        index, _, _ = build_test_index(chunker=chunker)
        assert index.spec.chunk_params != base.spec.chunk_params
        ids.add(index.index_id)
    assert len(ids) == 4  # all different


def test_embedding_model_change_changes_identity():
    base, _, _ = build_test_index()
    other, _, _ = build_test_index(embedder=HashEmbedder(dimension=64))  # different dimension
    assert other.spec.embedding_dimension != base.spec.embedding_dimension
    assert other.index_id != base.index_id
    assert compute_index_id(base.spec.model_copy(update={"embedding_model_id": "other/model"})) != (
        base.index_id
    )


def test_representation_change_changes_identity():
    prefixed, _, _ = build_test_index(style="prefixed")
    raw, _, _ = build_test_index(style="raw")
    assert prefixed.index_id != raw.index_id
    bumped = prefixed.spec.model_copy(update={"representation_version": 2})
    assert compute_index_id(bumped) != prefixed.index_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository_name", "other-repo"),
        ("repository_fingerprint", "0" * 64),
        ("chunking_strategy", "ast"),
        ("chunking_strategy_version", 2),
        ("chunk_params", {"size_lines": 4, "overlap_lines": 1, "max_tokens": 768}),
        ("chunk_id_schema", "chunk-id/2"),
        ("embedding_model_id", "other/model"),
        ("embedding_dimension", 33),
        ("embedding_normalized", False),
        ("text_style", "raw"),
        ("representation_version", 2),
        ("index_type", "IndexHNSWFlat"),
        ("metric", "l2"),
    ],
)
def test_every_spec_field_is_part_of_the_identity(field, value):
    index, _, _ = build_test_index()
    changed = index.spec.model_copy(update={field: value})
    assert compute_index_id(changed) != index.index_id, field


def test_chunk_param_key_order_does_not_matter():
    index, _, _ = build_test_index()
    reordered = spec_of(chunk_params=dict(reversed(list(index.spec.chunk_params.items()))))
    assert compute_index_id(reordered) == index.index_id
