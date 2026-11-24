"""Size matching, size profiles, artifact safety and frozen-config handling (Milestone 10)."""

from __future__ import annotations

import json

import pytest

from copilot.chunking.base import chunk_repository
from copilot.evaluation.chunking_experiment import matching
from copilot.evaluation.chunking_experiment.artifacts import (
    ArtifactError,
    safe_artifact_path,
    write_text_atomic,
)
from copilot.evaluation.chunking_experiment.config import (
    ArmConfig,
    ExperimentConfigError,
    arms_from_config,
    canonical_json,
    file_sha256,
    load_frozen_config,
)
from copilot.evaluation.chunking_experiment.matching import Candidate
from copilot.evaluation.chunking_experiment.profile import profile_by_scope, profile_chunks
from copilot.models.ingestion import IngestionResult
from tests.ast_helpers import ast_chunker, line_chunker
from tests.chunking_helpers import make_file


def _profile_of(median: float, mean: float):
    base = profile_chunks([], [], scope="python")
    from dataclasses import replace

    return replace(base, tokens_median=median, tokens_mean=mean)


def _candidate(parameter: int, median_error: float, mean_error: float) -> Candidate:
    return Candidate(parameter, _profile_of(1, 1), median_error, mean_error)


# --- selection policy -------------------------------------------------------------------------


def test_selection_is_lexicographic_median_then_mean_then_smaller_parameter():
    best = matching.select([_candidate(10, 0.05, 0.0), _candidate(12, -0.02, 0.5)])
    assert best.parameter == 12  # the median error decides first
    tie = matching.select([_candidate(20, 0.02, 0.30), _candidate(18, -0.02, 0.10)])
    assert tie.parameter == 18  # equal |median error| -> smaller |mean error|
    full_tie = matching.select([_candidate(20, 0.02, 0.1), _candidate(18, -0.02, -0.1)])
    assert full_tie.parameter == 18  # all equal -> smaller parameter


def test_grids_are_bounded_and_respect_the_cap():
    assert 0 < len(matching.C_SIZE_GRID) <= 100 and 60 in matching.C_SIZE_GRID
    assert max(matching.D_MERGE_GRID) <= 512 and min(matching.D_MERGE_GRID) >= 1
    assert 0 < matching.TOLERANCE < 1


def _ingestion(files) -> IngestionResult:
    class _Fake:
        repository_name = "repo"

    fake = _Fake()
    fake.files = tuple(files)
    return fake  # type: ignore[return-value]


def _repo():
    body = "\n".join(
        f"def f{i}(a):\n" + "".join(f"    v{k} = a + {k}\n" for k in range(6)) + "    return a\n"
        for i in range(40)
    )
    return [make_file(body, "pkg/a.py"), make_file("print(1)\n" * 70, "pkg/b.py")]


def test_match_arm_c_targets_the_python_profile_of_its_reference():
    ing = _ingestion(_repo())
    target = profile_by_scope(chunk_repository(ing, ast_chunker()).chunks, ing.files)["python"]
    result = matching.match_arm_c(ing, target, base_size=60, base_overlap=10, max_tokens=512)
    assert result.arm == "C" and result.target_arm == "B"
    assert len(result.candidates) == len(matching.C_SIZE_GRID)
    assert result.selected.parameter in matching.C_SIZE_GRID
    assert abs(result.selected.median_error) == min(abs(c.median_error) for c in result.candidates)
    assert result.adequate == (abs(result.selected.median_error) <= matching.TOLERANCE)


def test_match_arm_d_reports_inadequate_matches_instead_of_pretending():
    ing = _ingestion(_repo())
    target = profile_by_scope(chunk_repository(ing, line_chunker()).chunks, ing.files)["python"]
    impossible = _profile_of(target.tokens_median * 10, target.tokens_mean * 10)
    result = matching.match_arm_d(ing, impossible, base_size=60, base_overlap=10, max_tokens=512)
    # closest feasible: the lowest error available, ties (saturated merging) -> smaller budget
    assert abs(result.selected.median_error) == min(abs(c.median_error) for c in result.candidates)
    assert result.selected.parameter <= max(result.grid)
    assert result.adequate is False
    payload = result.to_json()
    assert payload["adequate"] is False and len(payload["candidates"]) == len(result.grid)


# --- profiles ---------------------------------------------------------------------------------


def test_profile_counts_redundancy_and_coverage_by_hand():
    file = make_file("".join(f"x{i} = {i}\n" for i in range(10)), "a.py")
    # windows of 6 lines with 2 overlap -> (1-6), (5-10): 12 slots over 10 distinct lines
    chunks = line_chunker(512, size=6, overlap=2).chunk_file(file)
    p = profile_chunks(chunks, [file], scope="python")
    assert p.chunk_count == 2 and p.redundancy == pytest.approx(12 / 10)
    assert p.coverage == pytest.approx(1.0) and p.non_blank_coverage == pytest.approx(1.0)
    assert p.source_lines == 10 and p.represented_lines == 10


def test_profile_reports_uncovered_blank_lines_separately_from_code():
    file = make_file("def a():\n    return 1\n\n\ndef b():\n    return 2\n", "a.py")
    chunks = ast_chunker().chunk_file(file)
    p = profile_chunks(chunks, [file], scope="python")
    assert p.coverage < 1.0 and p.non_blank_coverage == pytest.approx(1.0)


def test_profile_by_scope_separates_python_from_everything():
    files = [make_file("a = 1\n", "a.py"), make_file("let a = 1;\n", "a.js", language="javascript")]
    ing = _ingestion(files)
    scopes = profile_by_scope(chunk_repository(ing, line_chunker()).chunks, files)
    assert scopes["python"].chunk_count == 1 and scopes["all"].chunk_count == 2


# --- artifacts ---------------------------------------------------------------------------------


def test_artifacts_reject_traversal_and_overwrite_and_leave_no_temp_files(tmp_path):
    with pytest.raises(ArtifactError):
        safe_artifact_path(tmp_path, "../escape.json")
    with pytest.raises(ArtifactError):
        safe_artifact_path(tmp_path, "/etc/passwd")
    inside = safe_artifact_path(tmp_path, "a/b.json")
    write_text_atomic(inside, "one\n")
    with pytest.raises(ArtifactError):
        write_text_atomic(inside, "two\n")
    assert inside.read_text() == "one\n"
    write_text_atomic(inside, "two\n", overwrite=True)
    assert inside.read_text() == "two\n"
    assert [p.name for p in inside.parent.iterdir()] == ["b.json"]


def test_artifact_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    (base / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactError):
        safe_artifact_path(base, "link/x.json")


# --- frozen config -----------------------------------------------------------------------------


def _config():
    arms = [
        ArmConfig(
            "A", "baseline", "line", 1, {"size_lines": 60, "overlap_lines": 10, "max_tokens": 512}
        ),
        ArmConfig("B", "ast", "ast", 1, ast_chunker().params()),
        ArmConfig(
            "C",
            "sized",
            "line_sized",
            1,
            {
                "python_size_lines": 15,
                "python_overlap_lines": 0,
                "size_lines": 60,
                "overlap_lines": 10,
                "max_tokens": 512,
            },
        ),
        ArmConfig(
            "D",
            "merged",
            "ast_merged",
            1,
            {**ast_chunker().params(), "merge_max_tokens": 512, "base": "ast@1"},
        ),
    ]
    return {"schema": "chunking-experiment-config/1", "arms": [a.to_json() for a in arms]}


def test_frozen_config_round_trips_and_every_arm_rebuilds_its_chunker(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(canonical_json(_config()))
    loaded = load_frozen_config(path, expected_sha256=file_sha256(path))
    arms = arms_from_config(loaded)
    assert [a.arm_id for a in arms] == ["A", "B", "C", "D"]
    assert [a.chunker().name for a in arms] == ["line", "ast", "line_sized", "ast_merged"]


def test_frozen_config_detects_modification_and_malformed_input(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(canonical_json(_config()))
    digest = file_sha256(path)
    path.write_text(canonical_json({**_config(), "extra": 1}))
    with pytest.raises(ExperimentConfigError):
        load_frozen_config(path, expected_sha256=digest)
    path.write_text("{not json")
    with pytest.raises(ExperimentConfigError):
        load_frozen_config(path)
    path.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(ExperimentConfigError):
        load_frozen_config(path)
    with pytest.raises(ExperimentConfigError):
        arms_from_config({"arms": [{"arm_id": "A"}]})
    with pytest.raises(ExperimentConfigError):
        load_frozen_config(tmp_path / "missing.json")
