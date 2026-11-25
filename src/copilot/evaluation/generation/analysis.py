"""Statistical analysis and reporting for a plain-vs-RAG generation experiment.

Everything reported is computed from *rows* (one per question and arm) that point back at the raw
run and judgement files, so every number can be traced and recomputed (``recompute_primary``).

**Primary endpoint** (fixed in the protocol before generation): the paired difference in mean
key-fact correctness, RAG minus plain, over answerable questions both arms answered and that were
scored. The estimate is the mean of the per-question differences with a percentile bootstrap CI
(shared resample indices, fixed seed; ``copilot.evaluation.stats``). It is given per repository and
combined. Pairs with a missing, failed or unscored side are **listed and counted, never silently
dropped and never imputed**; end-to-end reliability is reported separately from the
successful-answer analysis.

**What this module does not do:** it declares no winner. It states the estimate, the interval and
whether the interval excludes zero, and lists the caveats that apply (test artifact, synthetic
repository, same-family judge, unconfirmed model, incomplete pairs, reused historical answers).
Output produced from ``FakeLLMClient`` runs is labelled ``TEST ARTIFACT`` and is not a result.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from copilot.evaluation.generation.citations import RetrievedRegion, check_citations
from copilot.evaluation.generation.execution import GenerationRun, SideRecord
from copilot.evaluation.generation.inventory import RepoInventory
from copilot.evaluation.generation.judging import JudgedItem, JudgementFile
from copilot.evaluation.generation.protocol import PrimaryProtocol, protocol_hash
from copilot.evaluation.generation.references import find_references
from copilot.evaluation.generation.refusal import abstention_metrics, classify_response
from copilot.evaluation.generation.rubric import EvaluationError, score_assessment
from copilot.evaluation.generation.schema import GenerationQuestion
from copilot.evaluation.stats import bootstrap_ci, bootstrap_indices

REPORT_SCHEMA = "generation-report/1"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class RepoInputs:
    """Everything the analysis needs for one repository (loaded by the caller)."""

    run: GenerationRun
    questions: Mapping[str, GenerationQuestion]
    inventory: RepoInventory
    judgements: JudgementFile | None
    run_file: str
    judgement_file: str | None = None
    independent: bool = False
    kind: str = ""


def _score_of(question: GenerationQuestion, item: JudgedItem | None) -> tuple[str, dict]:
    """``(score_state, fields)``: scored / judge_failed / not_judged / side_unavailable."""
    if item is None:
        return "not_judged", {}
    if item.state == "skipped":
        return "side_unavailable", {}
    if item.state == "failed" or item.assessment is None:
        return "judge_failed", {}
    try:
        result = score_assessment(question, item.assessment)
    except EvaluationError:
        return "judge_failed", {}
    return "scored", {
        "score": result.score,
        "supported": result.supported,
        "total_facts": result.total_facts,
        "completeness": result.completeness,
        "groundedness": item.assessment.groundedness,
        "judged_abstained": item.assessment.abstained,
        "unsupported_claims": result.material_unsupported,
        "evaluator_kind": item.assessment.evaluator_kind,
    }


def _text_metrics(
    question: GenerationQuestion, side: SideRecord, inventory: RepoInventory
) -> dict[str, object]:
    out: dict[str, object] = {}
    out["abstention_label"] = classify_response(
        side.answer, state=side.state, system_insufficient=side.system_insufficient
    ).label
    if not side.has_text:
        return out
    sources = (
        {
            s.source_number: RetrievedRegion(
                source_number=s.source_number,
                file_path=s.file_path,
                start_line=s.start_line,
                end_line=s.end_line,
            )  # fmt: skip
            for s in side.sources
        }
        if side.arm == "rag"
        else None
    )
    report = check_citations(
        side.answer, arm=side.arm, inventory=inventory, answered=side.status == "answered",
        sources=sources, expected=question.relevant_regions or None,
    )  # fmt: skip
    out.update(
        citation_applicable=report.applicable,
        citation_missing=report.missing,
        citation_valid_fraction=report.valid_fraction,
        citations_invented=report.invented,
        citations_total=len(report.citations),
    )
    refs = find_references(side.answer, inventory)
    out.update(
        refs_not_found=refs.not_found,
        refs_ambiguous=refs.count("ambiguous"),
        refs_verified=refs.count("verified"),
    )
    return out


def build_rows(inputs: RepoInputs) -> list[dict[str, object]]:
    """One row per (question, arm): scores, programmatic metrics, usage and provenance."""
    run = inputs.run
    by_key = inputs.judgements.by_key() if inputs.judgements else {}
    rows: list[dict[str, object]] = []
    for entry in run.entries:
        question = inputs.questions[entry.question_id]
        for arm in ("plain", "rag"):
            side: SideRecord = getattr(entry, arm)
            item = by_key.get((question.id, arm))
            score_state, fields = _score_of(question, item)
            row: dict[str, object] = {
                "repository_id": run.repository_id,
                "question_id": question.id,
                "arm": arm,
                "answerable": question.is_answerable,
                "side_state": side.state,
                "origin": side.origin,
                "score_state": score_state,
                "run_file": inputs.run_file,
                "judgement_file": inputs.judgement_file,
                "blind_id": item.blind_id if item else None,
                "test_artifact": run.test_artifact,
                **fields,
            }
            row.update(_text_metrics(question, side, inputs.inventory))
            if arm == "rag" and question.is_answerable:
                row["retrieval_hit"] = (
                    side.first_relevant_source is not None if side.state == "successful" else None
                )
            if side.origin == "generated" and side.latency_seconds is not None:
                row["latency_seconds"] = side.latency_seconds
            for name in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
                value = getattr(side, name)
                if value is not None and side.origin == "generated":
                    row[name] = value
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- statistics


def paired_summary(
    pairs: Sequence[tuple[float, float]], *, seed: int, resamples: int, confidence: float
) -> dict[str, object]:
    """Mean of (rag - plain) over ``pairs`` of ``(plain, rag)`` with a bootstrap percentile CI."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "mean_plain": None, "mean_rag": None, "mean_difference": None,
                "ci_low": None, "ci_high": None, "ci_excludes_zero": None}  # fmt: skip
    plain = [p for p, _ in pairs]
    rag = [r for _, r in pairs]
    diffs = [r - p for p, r in pairs]
    low, high = bootstrap_ci(diffs, bootstrap_indices(n, resamples, seed), confidence)
    return {
        "n": n,
        "mean_plain": statistics.fmean(plain),
        "mean_rag": statistics.fmean(rag),
        "mean_difference": statistics.fmean(diffs),
        "ci_low": low,
        "ci_high": high,
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "confidence": confidence,
    }


def _pairs(
    rows: Sequence[Mapping[str, object]], field: str, *, answerable_only: bool = False
) -> tuple[list[tuple[float, float]], list[dict[str, object]]]:
    """Complete ``(plain, rag)`` pairs on ``field`` and the excluded questions with the reason."""
    grouped: dict[tuple[object, object], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        if answerable_only and not row["answerable"]:
            continue
        grouped.setdefault((row["repository_id"], row["question_id"]), {})[str(row["arm"])] = row
    pairs: list[tuple[float, float]] = []
    excluded: list[dict[str, object]] = []
    for (repo, qid), sides in grouped.items():
        plain, rag = sides.get("plain"), sides.get("rag")
        if plain is not None and rag is not None and plain.get(field) is not None \
                and rag.get(field) is not None:  # fmt: skip
            pairs.append((float(plain[field]), float(rag[field])))  # type: ignore[arg-type]
            continue
        excluded.append(
            {
                "repository_id": repo,
                "question_id": qid,
                "plain": (plain or {}).get("score_state", "missing") if field == "score" else None,
                "rag": (rag or {}).get("score_state", "missing") if field == "score" else None,
                "reason": "an arm has no value for this metric",
            }
        )
    return pairs, excluded


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def recompute_primary(
    rows: Sequence[Mapping[str, object]], repository_id: str | None = None
) -> float | None:
    """The primary estimate straight from rows (used to prove a report's number is traceable)."""
    subset = [r for r in rows if repository_id is None or r["repository_id"] == repository_id]
    pairs, _ = _pairs(subset, "score", answerable_only=True)
    return _mean([r - p for p, r in pairs])


def _arm_rows(rows: Sequence[Mapping[str, object]], arm: str) -> list[Mapping[str, object]]:
    return [r for r in rows if r["arm"] == arm]


def _reliability(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for arm in ("plain", "rag"):
        subset = _arm_rows(rows, arm)
        counts = {s: sum(1 for r in subset if r["side_state"] == s)
                  for s in ("successful", "failed", "not_attempted")}  # fmt: skip
        out[arm] = {
            **counts,
            "requested": len(subset),
            "success_rate": _rate(counts["successful"], len(subset)),
            "judge_failed": sum(1 for r in subset if r["score_state"] == "judge_failed"),
            "reused_historical": sum(1 for r in subset if r["origin"] == "historical"),
            "served_from_cache": sum(1 for r in subset if r["origin"] == "cache"),
        }
    return out


def _arm_secondary(rows: Sequence[Mapping[str, object]], arm: str) -> dict[str, object]:
    subset = _arm_rows(rows, arm)
    cite = [r for r in subset if r.get("citation_applicable")]
    valid = [float(r["citation_valid_fraction"]) for r in cite
             if r.get("citation_valid_fraction") is not None]  # fmt: skip
    abst = abstention_metrics(
        (bool(r["answerable"]), r["abstention_label"])  # type: ignore[misc]
        for r in subset
        if "abstention_label" in r
    )
    hits = [r["retrieval_hit"] for r in subset if r.get("retrieval_hit") is not None]
    latencies = [float(r["latency_seconds"]) for r in subset if "latency_seconds" in r]
    out: dict[str, object] = {
        "citations": {
            "applicable_answers": len(cite),
            "answers_missing_citations": sum(1 for r in cite if r.get("citation_missing")),
            "mean_valid_fraction": _mean(valid),
            "invented_citations": sum(int(r.get("citations_invented", 0)) for r in subset),
        },
        "hallucinated_references": {
            "answers_checked": sum(1 for r in subset if "refs_not_found" in r),
            "not_found_total": sum(int(r.get("refs_not_found", 0)) for r in subset),
            "ambiguous_total": sum(int(r.get("refs_ambiguous", 0)) for r in subset),
            "answers_with_not_found": sum(1 for r in subset if int(r.get("refs_not_found", 0)) > 0),
        },
        "abstention": abst.to_json(),
        "latency_seconds": {
            "n": len(latencies),
            "mean": _mean(latencies),
            "median": statistics.median(latencies) if latencies else None,
            "excluded_cache_or_historical": sum(
                1 for r in subset if r["side_state"] == "successful" and "latency_seconds" not in r
            ),
        },
        "tokens": {
            name: {
                "n": len(vals),
                "mean": _mean(vals),
                "total": sum(vals) if vals else None,
            }
            for name in ("prompt_tokens", "completion_tokens", "reasoning_tokens")
            for vals in [[float(r[name]) for r in subset if name in r]]
        },
    }
    if arm == "rag":
        out["retrieval"] = {
            "answerable_with_outcome": len(hits),
            "hit_at_k": _rate(sum(1 for h in hits if h), len(hits)),
        }
    return out


def analyse(rows: Sequence[Mapping[str, object]], protocol: PrimaryProtocol) -> dict[str, object]:
    """Endpoints from rows: primary (per repository and combined), secondaries, reliability."""
    seed, resamples, conf = (
        protocol.bootstrap_seed,
        protocol.bootstrap_resamples,
        protocol.confidence,
    )
    repos = sorted({str(r["repository_id"]) for r in rows})

    def block(subset: Sequence[Mapping[str, object]]) -> dict[str, object]:
        pairs, excluded = _pairs(subset, "score", answerable_only=True)
        g_pairs, _ = _pairs(subset, "groundedness")
        c_pairs, _ = _pairs(subset, "completeness", answerable_only=True)
        h_pairs, _ = _pairs([r for r in subset if "refs_not_found" in r], "refs_not_found")
        return {
            "primary": {
                **paired_summary(pairs, seed=seed, resamples=resamples, confidence=conf),
                "excluded_pairs": excluded,
                "answerable_questions": len(
                    {(r["repository_id"], r["question_id"]) for r in subset if r["answerable"]}
                ),
            },
            "secondary_paired": {
                "groundedness": paired_summary(
                    g_pairs, seed=seed, resamples=resamples, confidence=conf
                ),
                "completeness": paired_summary(
                    c_pairs, seed=seed, resamples=resamples, confidence=conf
                ),
                "hallucinated_reference_not_found": paired_summary(
                    h_pairs, seed=seed, resamples=resamples, confidence=conf
                ),
            },
            "per_arm": {arm: _arm_secondary(subset, arm) for arm in ("plain", "rag")},
            "reliability": _reliability(subset),
        }

    return {
        "combined": block(rows),
        "per_repository": {
            repo: block([r for r in rows if r["repository_id"] == repo]) for repo in repos
        },  # fmt: skip
    }


def caveats(
    inputs: Sequence[RepoInputs], protocol: PrimaryProtocol, analysis: Mapping[str, object]
) -> list[str]:
    out: list[str] = []
    if any(i.run.test_artifact for i in inputs) or any(
        i.judgements and i.judgements.test_artifact for i in inputs
    ):
        out.append(
            "TEST ARTIFACT: produced with a fake provider. It exercises the pipeline and "
            "contains no experimental result."
        )
    if not protocol.model_confirmed:
        out.append("The generation model was not confirmed in the protocol.")
    for i in inputs:
        if not i.independent:
            out.append(
                f"{i.run.repository_id}: authored by the project team (not independent software); "
                f"kind={i.kind or 'unspecified'}."
            )
        if i.judgements and i.judgements.same_model_as_generator:
            out.append(
                f"{i.run.repository_id}: the judge is the same model as the generator "
                "(self-preference risk)."
            )
        if i.judgements is None:
            out.append(f"{i.run.repository_id}: no judgements - no correctness scores exist.")
    combined = analysis["combined"]  # type: ignore[index]
    excluded = combined["primary"]["excluded_pairs"]  # type: ignore[index]
    if excluded:
        out.append(f"{len(excluded)} answerable question(s) lack a complete scored pair.")
    reliability = combined["reliability"]  # type: ignore[index]
    reused = sum(reliability[a]["reused_historical"] for a in ("plain", "rag"))  # type: ignore[index]
    if reused:
        out.append(f"{reused} side(s) were reused from earlier verified historical responses.")
    out.extend(protocol.limitations)
    return out


def build_report(
    inputs: Sequence[RepoInputs], protocol: PrimaryProtocol, *, input_files: Mapping[str, str]
) -> dict[str, object]:
    """The machine-readable report. ``input_files`` maps a label to its path for hashing."""
    rows: list[dict[str, object]] = []
    for item in inputs:
        rows.extend(build_rows(item))
    result = analyse(rows, protocol)
    return {
        "schema_version": REPORT_SCHEMA,
        "test_artifact": any(
            i.run.test_artifact or (i.judgements is not None and i.judgements.test_artifact)
            for i in inputs
        ),
        "protocol_hash": protocol_hash(protocol),
        "primary_endpoint": protocol.primary_endpoint.model_dump(),
        "input_sha256": {label: file_sha256(path) for label, path in sorted(input_files.items())},
        "analysis": result,
        "caveats": caveats(inputs, protocol, result),
        "rows": rows,
    }


# --------------------------------------------------------------------------- markdown


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: Mapping[str, object]) -> str:
    """A human-readable summary of the report. It states estimates; it declares no winner."""
    lines = ["# Plain LLM vs repository-aware RAG - answer-quality report", ""]
    if report["test_artifact"]:
        lines += [
            "> **TEST ARTIFACT - not an experimental result.** Produced with a fake provider to "
            "exercise the pipeline.",
            "",
        ]
    lines += [f"Protocol hash: `{report['protocol_hash']}`", ""]
    endpoint = report["primary_endpoint"]
    lines += [f"**Primary endpoint:** {endpoint['name']}", "", endpoint["definition"], ""]  # type: ignore[index]
    analysis = report["analysis"]
    lines += ["## Primary endpoint", "",
              "| Scope | n | plain | RAG | RAG - plain | 95% CI | excludes 0 |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip

    def row(label: str, block: Mapping[str, object]) -> str:
        p = block["primary"]  # type: ignore[index]
        ci = "n/a" if p["ci_low"] is None else f"[{_fmt(p['ci_low'])}, {_fmt(p['ci_high'])}]"
        return (
            f"| {label} | {p['n']} | {_fmt(p['mean_plain'])} | {_fmt(p['mean_rag'])} | "
            f"{_fmt(p['mean_difference'])} | {ci} | {_fmt(p['ci_excludes_zero'])} |"
        )

    lines.append(row("combined", analysis["combined"]))  # type: ignore[index]
    for repo, block in analysis["per_repository"].items():  # type: ignore[union-attr]
        lines.append(row(repo, block))
    lines += ["", "## Reliability (end to end)", "",
              "| Scope | arm | requested | successful | failed | not attempted | judge failed |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    scopes = [("combined", analysis["combined"])]  # type: ignore[index]
    scopes += list(analysis["per_repository"].items())  # type: ignore[union-attr]
    for label, block in scopes:
        for arm in ("plain", "rag"):
            r = block["reliability"][arm]
            lines.append(
                f"| {label} | {arm} | {r['requested']} | {r['successful']} | {r['failed']} | "
                f"{r['not_attempted']} | {r['judge_failed']} |"
            )
    lines += ["", "## Secondary endpoints (combined)", ""]
    combined = analysis["combined"]  # type: ignore[index]
    for name, summary in combined["secondary_paired"].items():
        lines.append(
            f"- {name}: n={summary['n']}, mean difference (RAG - plain)="
            f"{_fmt(summary['mean_difference'])}"
        )
    for arm in ("plain", "rag"):
        arm_block = combined["per_arm"][arm]
        abst = arm_block["abstention"]
        lines.append(
            f"- {arm}: appropriate abstention rate={_fmt(abst['appropriate_abstention_rate'])}, "
            f"false refusal rate={_fmt(abst['false_refusal_rate'])}, "
            f"unsupported answer rate={_fmt(abst['unsupported_answer_rate'])}, "
            f"invented citations={arm_block['citations']['invented_citations']}, "
            f"references not found={arm_block['hallucinated_references']['not_found_total']}"
        )
    if "retrieval" in combined["per_arm"]["rag"]:
        lines.append(f"- RAG retrieval hit@k (answerable): "
                     f"{_fmt(combined['per_arm']['rag']['retrieval']['hit_at_k'])}")  # fmt: skip
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in report["caveats"]]  # type: ignore[union-attr]
    lines += ["", "Every figure above is recomputable from the `rows` in the JSON report, which "
              "reference the raw run and judgement files by path and SHA-256.", ""]  # fmt: skip
    return "\n".join(lines)
