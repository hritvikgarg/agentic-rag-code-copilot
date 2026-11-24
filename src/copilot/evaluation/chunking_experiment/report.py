"""Tables, plots and a human-readable report generated from the saved result files.

Nothing here measures anything: it only reads ``results/*.json``, the frozen design and the
benchmark, so every number in the report traces to a machine-readable file. The generated text is
descriptive and deliberately avoids words such as "better" or "significant"; interpretation lives
in the hand-written results document.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from copilot.evaluation.benchmark import load_benchmark
from copilot.evaluation.chunking_experiment import error_analysis, svg
from copilot.evaluation.chunking_experiment.artifacts import write_text_atomic
from copilot.evaluation.chunking_experiment.config import canonical_json

ARMS = ("A", "B", "C", "D")
LABELS = {
    "A": "A baseline (line)",
    "B": "B ast",
    "C": "C line, size-matched",
    "D": "D ast, merged",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _md(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(count: int, n: int) -> str:
    return f"{count}/{n} ({100 * count / n:.1f}%)"


def build_tables(
    results_dir: Path, design_dir: Path
) -> tuple[dict[str, tuple[list[str], list[list[object]]]], dict[str, Any]]:
    """Every table as ``(header, rows)`` plus the loaded inputs."""
    agg = _read(results_dir / "aggregates.json")
    paired = _read(results_dir / "paired-statistics.json")
    latency = _read(results_dir / "latency.json")
    check = _read(design_dir / "design-check.json")
    builds = {a: _read(results_dir / "builds" / f"{a}.json") for a in ARMS}
    config = _read(design_dir / "frozen-config.json")
    budgets = config["budget"]["budgets"]
    arms = agg["arms"]
    n = arms["A"]["n_questions"]
    t: dict[str, tuple[list[str], list[list[object]]]] = {}

    rows = []
    for scope in ("python", "all"):
        for arm in ARMS:
            p = check["arms"][arm]["profiles"][scope]
            rows.append(
                [scope, arm, p["chunk_count"], p["tokens_min"], p["tokens_p25"],
                 p["tokens_median"], round(p["tokens_mean"], 1), p["tokens_p75"],
                 p["tokens_max"], round(p["redundancy"], 3), round(p["non_blank_coverage"], 3)]
            )  # fmt: skip
    t["chunk-sizes"] = (
        ["scope", "arm", "chunks", "min", "p25", "median", "mean", "p75", "max", "redundancy",
         "non_blank_coverage"],
        rows,
    )  # fmt: skip

    rows = []
    for arm in ARMS:
        a = arms[arm]
        b, c = a["binary"], a["continuous"]
        rows.append(
            [arm]
            + [_pct(b[f"hit@{k}"]["count"], n) for k in (1, 3, 5, 10)]
            + [f"{c['mrr@10']:.3f}", f"{c['coverage@5']:.3f}", f"{c['coverage@10']:.3f}",
               _pct(b["coverage_hit@5"]["count"], n), _pct(b["coverage_hit@10"]["count"], n),
               a["first_rank"]["median"], f"{a['size']['mean_retrieved_lines']:.1f}",
               f"{a['size']['mean_retrieved_tokens_top10']:.0f}",
               f"{a['size']['mean_distinct_files_top10']:.2f}"]
        )  # fmt: skip
    t["fixed-k"] = (
        ["arm", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10", "coverage@5", "coverage@10",
         "coverage-hit@5", "coverage-hit@10", "median first rank", "mean retrieved lines",
         "mean retrieved tokens (top10)", "mean distinct files (top10)"],
        rows,
    )  # fmt: skip

    rows = []
    for arm in ARMS:
        for budget in budgets:
            a = arms[arm]
            s = a["budgets"][str(budget)]
            rows.append(
                [arm, budget, _pct(a["binary"][f"budget_hit@{budget}"]["count"], n),
                 f"{a['continuous'][f'budget_coverage@{budget}']:.3f}",
                 f"{s['mean_chunks']:.2f}", f"{s['mean_distinct_files']:.2f}",
                 f"{s['mean_tokens']:.0f}", f"{s['mean_utilization']:.3f}"]
            )  # fmt: skip
    t["fixed-budget"] = (
        ["arm", "budget (est. tokens)", "evidence hit", "region coverage", "mean chunks",
         "mean distinct files", "mean tokens used", "mean utilisation"],
        rows,
    )  # fmt: skip

    rows = []
    for key, comp in paired["comparisons"].items():
        for name, r in comp["binary"].items():
            tab = r["table"]
            rows.append(
                [key, name, r["count_x"], r["count_y"], f"{r['difference_pp']:+.1f}",
                 f"[{r['ci_pp'][0]:+.1f}, {r['ci_pp'][1]:+.1f}]", r["improved"], r["regressed"],
                 tab["both"], tab["neither"], f"{r['mcnemar_p']:.3f}",
                 "-" if r["holm_p"] is None else f"{r['holm_p']:.3f}",
                 "primary" if r["is_primary"] else "exploratory"]
            )  # fmt: skip
    t["paired-binary"] = (
        ["comparison (X-Y)", "endpoint", "X count", "Y count", "diff pp", "95% CI pp",
         "X only", "Y only", "both", "neither", "exact McNemar p", "Holm p", "status"],
        rows,
    )  # fmt: skip

    rows = []
    for key, comp in paired["comparisons"].items():
        for name, r in comp["continuous"].items():
            rows.append(
                [key, name, f"{r['mean_x']:.3f}", f"{r['mean_y']:.3f}",
                 f"{r['mean_difference']:+.3f}", f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"]
            )  # fmt: skip
    t["paired-continuous"] = (
        ["comparison (X-Y)", "endpoint", "mean X", "mean Y", "mean difference", "95% CI"],
        rows,
    )  # fmt: skip

    rows = []
    for arm in ARMS:
        b, lat = builds[arm], latency["summary"][arm]
        rows.append(
            [arm, b["chunk_count"], b["bytes"]["index_total"], f"{b['seconds']['chunk']:.2f}",
             f"{b['seconds']['embed']:.1f}", f"{b['seconds']['total']:.1f}",
             f"{lat['median_ms']:.1f}", f"{lat['p95_ms']:.1f}", lat["samples"]]
        )  # fmt: skip
    t["efficiency"] = (
        ["arm", "chunks", "index bytes", "chunking s", "embedding s", "build total s",
         "latency median ms", "latency p95 ms", "latency samples"],
        rows,
    )  # fmt: skip
    inputs = {
        "agg": agg, "paired": paired, "latency": latency, "check": check, "builds": builds,
        "config": config,
    }  # fmt: skip
    return t, inputs


def build_plots(inputs: dict[str, Any]) -> dict[str, str]:
    """SVG charts keyed by file name."""
    arms = inputs["agg"]["arms"]
    n = arms["A"]["n_questions"]
    plots: dict[str, str] = {}
    ks = (1, 3, 5, 10)
    plots["hit-at-k.svg"] = svg.grouped_bars(
        "Hit@k by arm (primary any-overlap rule)",
        [f"Hit@{k}" for k in ks],
        {a: [100 * arms[a]["binary"][f"hit@{k}"]["count"] / n for k in ks] for a in ARMS},
        y_label="% of questions",
        y_max=100,
    )
    budgets = inputs["config"]["budget"]["budgets"]
    plots["budget-hit.svg"] = svg.grouped_bars(
        "Evidence hit within a context budget (retrieval only)",
        [f"{b} tokens" for b in budgets],
        {
            a: [100 * arms[a]["binary"][f"budget_hit@{b}"]["count"] / n for b in budgets]
            for a in ARMS
        },
        y_label="% of questions",
        y_max=100,
    )
    rows = []
    for key in ("B-A", "B-C", "C-A", "D-A", "D-B"):
        r = inputs["paired"]["comparisons"][key]["binary"]["hit@5"]
        rows.append((f"{key} Hit@5", r["difference_pp"], r["ci_pp"][0], r["ci_pp"][1]))
    plots["paired-hit5-differences.svg"] = svg.forest(
        "Paired Hit@5 difference, X minus Y (percentage points, bootstrap 95% CI)",
        rows,
        x_label="percentage points (1 question = 2.94)",
    )
    profile_rows = []
    for arm in ARMS:
        p = inputs["check"]["arms"][arm]["profiles"]["python"]
        profile_rows.append(
            (f"{arm} Python", p["tokens_min"], p["tokens_p25"], p["tokens_median"], p["tokens_p75"],
             p["tokens_max"])
        )  # fmt: skip
    plots["python-chunk-sizes.svg"] = svg.range_plot(
        "Python chunk size (estimated tokens): min, quartiles, max",
        profile_rows,
        x_label="estimated tokens",
    )
    return plots


def _regions(benchmark_path: str) -> dict[str, list[str]]:
    bench = load_benchmark(benchmark_path)
    return {
        q.id: [f"{r.file_path}:{r.start_line}-{r.end_line}" for r in q.relevant]
        for q in bench.questions
    }


def generate_report(
    results_dir: Path, design_dir: Path, out_dir: Path, *, overwrite: bool = False
) -> list[Path]:
    """Write tables (CSV), plots (SVG), the error-analysis JSON and ``report.md``."""
    tables, inputs = build_tables(results_dir, design_dir)
    plots = build_plots(inputs)
    per_question = _read(results_dir / "per-question.json")["arms"]
    analysis = error_analysis.analyse(per_question, _regions(inputs["config"]["benchmark"]["path"]))
    written: list[Path] = []
    for name, (header, rows) in tables.items():
        written.append(
            write_text_atomic(
                out_dir / "tables" / f"{name}.csv", _csv(header, rows), overwrite=overwrite
            )
        )
    for name, text in plots.items():
        written.append(
            write_text_atomic(out_dir / "plots" / name, text + "\n", overwrite=overwrite)
        )
    written.append(
        write_text_atomic(
            out_dir / "error-analysis.json", canonical_json(analysis), overwrite=overwrite
        )
    )
    written.append(
        write_text_atomic(
            out_dir / "report.md", render_markdown(tables, inputs, analysis), overwrite=overwrite
        )
    )
    return written


def render_markdown(
    tables: dict[str, tuple[list[str], list[list[object]]]],
    inputs: dict[str, Any],
    analysis: dict[str, Any],
) -> str:
    """The generated report (descriptive; no interpretation)."""
    paired = inputs["paired"]
    cfg = inputs["config"]
    out = [
        "# Chunking comparison: generated report",
        "",
        "Generated from `results/*.json`; descriptive only. Interpretation is in "
        "`docs/experiments/chunking-comparison-results.md`.",
        "",
        f"* Questions: {inputs['agg']['arms']['A']['n_questions']} "
        f"(benchmark SHA-256 `{cfg['benchmark']['sha256'][:12]}…`). One question = 2.94 pp.",
        f"* Bootstrap: seed {paired['bootstrap']['seed']}, {paired['bootstrap']['resamples']} "
        f"paired resamples, {int(100 * paired['bootstrap']['confidence'])}% percentile intervals.",
        f"* Primary comparison: {paired['primary_key']} (exact McNemar, α = "
        f"{paired['primary']['alpha']}); all other tests exploratory, Holm family size "
        f"{paired['holm_family_size']}.",
        "* Index ids: " + ", ".join(f"{a} `{b['index_id']}`" for a, b in inputs["builds"].items()),
        "",
    ]
    titles = {
        "chunk-sizes": "Chunk sizes (estimated tokens)",
        "fixed-k": "Fixed-k retrieval (top 10)",
        "fixed-budget": "Fixed context budget (retrieval-only simulation)",
        "paired-binary": "Paired comparison: binary endpoints",
        "paired-continuous": "Paired comparison: continuous endpoints",
        "efficiency": "Efficiency",
    }
    for name, (header, rows) in tables.items():
        out += [f"## {titles[name]}", "", _md(header, rows), ""]
    out += ["## Question groups (primary rule, Hit@5)", ""]
    for group, ids in analysis["groups"].items():
        out.append(f"* **{group}** ({len(ids)}): " + (", ".join(ids) if ids else "none"))
    out += ["", "## Per-arm retrieval diagnostics", ""]
    summary = analysis["arm_summary"]
    out.append(
        _md(
            [
                "arm",
                "mean docs in top5",
                "misses with ≥3 docs in top5",
                "mean repeated-symbol chunks (top10)",
                "mean retrieved redundancy (top10)",
            ],
            [
                [
                    a,
                    f"{s['mean_docs_in_top5']:.2f}",
                    s["misses_with_3plus_docs_in_top5"],
                    f"{s['mean_repeated_symbol_chunks_top10']:.2f}",
                    f"{s['mean_retrieved_redundancy_top10']:.3f}",
                ]
                for a, s in summary.items()
            ],
        )  # fmt: skip
    )
    out += [
        "",
        f"Latency ranking mismatches vs the scoring pass: "
        f"{inputs['latency']['ranking_mismatches_vs_scoring_pass']}.",
        "",
    ]
    return "\n".join(out)
