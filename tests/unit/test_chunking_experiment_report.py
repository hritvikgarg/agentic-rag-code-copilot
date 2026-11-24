"""Error analysis, SVG charts and the report generator (no retrieval or model is involved)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from copilot.evaluation.chunking_experiment import error_analysis, report, svg
from copilot.evaluation.chunking_experiment.artifacts import ArtifactError

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "chunking-comparison" / "results"
DESIGN = REPO / "experiments" / "chunking-comparison" / "design"


def _chunk(path, start, end, relevant=False, name=None, kind="function", tokens=100):
    return {
        "file_path": path,
        "start_line": start,
        "end_line": end,
        "relevant": relevant,
        "qualified_name": name,
        "chunk_type": kind,
        "token_estimate": tokens,
    }


def _row(qid, rank, retrieved):
    lines = next((c["end_line"] - c["start_line"] + 1 for c in retrieved if c["relevant"]), None)
    return {
        "question_id": qid,
        "first_hit_rank": rank,
        "first_hit_lines": lines,
        "retrieved": retrieved,
    }


def _four(arms):
    """Arms C and D mirror A so the fixtures only need to describe A and B."""
    return {**arms, "C": arms["A"], "D": arms["A"]}


def _per_question():
    docs = [_chunk("docs/a.md", 1, 40, kind="line_window") for _ in range(3)]
    miss = docs + [_chunk("x.py", 1, 5), _chunk("y.py", 1, 5)]
    hit = [_chunk("x.py", 1, 5, relevant=True, name="f")] + docs
    return _four(
        {
            "A": [_row("q1", None, miss), _row("q2", 1, hit), _row("q3", None, miss)],
            "B": [_row("q1", 1, hit), _row("q2", None, miss), _row("q3", None, miss)],
        }
    )


def test_groups_follow_hit_at_5_and_the_all_miss_group_uses_the_top_10_window():
    out = error_analysis.analyse(_per_question(), {"q1": ["x.py:1-5"]})
    groups = out["groups"]
    assert groups["b_improved_over_a"] == ["q1"]
    assert groups["b_regressed_vs_a"] == ["q2"]
    assert groups["all_arms_miss_top10"] == ["q3"]
    assert out["questions"]["q1"]["expected"] == ["x.py:1-5"]
    assert out["questions"]["q2"]["expected"] == []


def test_rank_beyond_five_is_not_a_top_5_hit():
    rows = _four(
        {
            "A": [_row("q", 7, [_chunk("x.py", 1, 5)])],
            "B": [_row("q", 5, [_chunk("x.py", 1, 5)])],
        }
    )
    groups = error_analysis.analyse(rows, {})["groups"]
    assert groups["b_improved_over_a"] == ["q"]
    assert groups["all_arms_miss_top10"] == []


def test_analysis_requires_all_four_arms():
    with pytest.raises(ValueError, match="missing"):
        error_analysis.analyse({"A": [], "B": []}, {})


def test_profile_counts_docs_repeated_symbols_and_redundancy():
    retrieved = [
        _chunk("docs/a.md", 1, 10, kind="line_window"),
        _chunk("x.py", 1, 10, name="f"),
        _chunk("x.py", 6, 15, name="f"),  # overlaps lines 6-10 of the previous chunk
    ]
    p = error_analysis.question_profile(_row("q", 2, retrieved))
    assert p["docs_in_top5"] == 1
    assert p["repeated_symbol_chunks_top10"] == 2
    assert p["retrieved_redundancy_top10"] == pytest.approx(30 / 25)
    assert p["top5"][0]["location"] == "docs/a.md:1-10"


def test_arm_summary_counts_doc_dominated_misses():
    summary = error_analysis.analyse(_per_question(), {})["arm_summary"]
    assert summary["A"]["misses_with_3plus_docs_in_top5"] == 2
    assert summary["B"]["misses_with_3plus_docs_in_top5"] == 2
    assert summary["A"]["mean_docs_in_top5"] == pytest.approx(3.0)


def test_svg_charts_are_well_formed_xml_and_escape_text():
    charts = [
        svg.grouped_bars(
            'Title <&> "q"',
            ["Hit@1", "Hit@5"],
            {"A": [10.0, 50.0], "B": [20.0, 90.0]},
            y_label="%",
            y_max=100,
        ),
        svg.forest("forest", [("B-A", 11.8, 0.0, 26.5), ("C-A", -5.0, -12.0, 2.0)], x_label="pp"),
        svg.range_plot("sizes", [("A", 16, 331, 469, 504, 512)], x_label="tokens"),
    ]
    for chart in charts:
        root = ET.fromstring(chart)
        assert root.tag.endswith("svg")
    assert "&lt;&amp;&gt;" in charts[0]


@pytest.mark.skipif(not (RESULTS / "aggregates.json").exists(), reason="saved results not present")
def test_report_from_saved_results_is_deterministic_and_never_overwrites(tmp_path):
    first = report.generate_report(RESULTS, DESIGN, tmp_path / "one")
    second = report.generate_report(RESULTS, DESIGN, tmp_path / "two")
    assert [p.relative_to(tmp_path / "one") for p in first] == [
        p.relative_to(tmp_path / "two") for p in second
    ]
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes()
    text = (tmp_path / "one" / "report.md").read_text(encoding="utf-8")
    assert "Primary comparison: B-A/hit@5" in text
    with pytest.raises(ArtifactError):
        report.generate_report(RESULTS, DESIGN, tmp_path / "one")
