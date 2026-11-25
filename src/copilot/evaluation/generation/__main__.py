"""Commands for the plain-LLM vs RAG answer experiment (Milestones 11A / 11B).

``python -m copilot.evaluation.generation COMMAND``:

* ``protocol-init``  write the frozen primary protocol (maintainers; once, before generation).
* ``plan``           dry run: repositories, question counts, generation and judge calls, artifact
                     paths and verified historical reuse. Constructs no provider; offline.
* ``run``            generate answers (hosted model; requires ``--confirm-model PROVIDER/MODEL``).
* ``judge``          judge a run's answers (hosted judge model).
* ``report``         analysis from run and judgement files (offline).
* ``audit-worksheet`` / ``audit-agree``  the human audit (offline).

Only ``run`` and ``judge`` can contact a provider; both refuse to start unless the model has been
confirmed on the command line, and both go through the secret gate. Output contains repository
relative paths, numbers and model answers only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from copilot.config import Settings, get_settings, setup_logging
from copilot.evaluation.benchmark import BenchmarkFormatError
from copilot.evaluation.comparison import load_comparison
from copilot.evaluation.generation.analysis import RepoInputs, build_report, render_markdown
from copilot.evaluation.generation.audit import (
    AuditAnswer,
    AuditUnit,
    agreement,
    build_worksheet,
    parse_human_row,
    sample_audit,
)
from copilot.evaluation.generation.compat import check_historical_run, frozen_target
from copilot.evaluation.generation.execution import (
    RepositoryContext,
    ResponseCache,
    RunMismatchError,
    historical_sides,
    load_run,
    plan_run,
    run_path,
    run_repositories,
    settings_for,
)
from copilot.evaluation.generation.inventory import build_inventory
from copilot.evaluation.generation.judge import JudgeCache, LLMJudge
from copilot.evaluation.generation.judging import (
    JudgementMismatchError,
    judge_run,
    judgement_path,
    load_judgements,
)
from copilot.evaluation.generation.protocol import (
    PROTOCOL_PATH,
    PrimaryProtocol,
    RepositorySpec,
    default_protocol,
    dump_protocol,
    load_protocol,
    protocol_hash,
)
from copilot.evaluation.generation.reference import build_reference
from copilot.evaluation.generation.rubric import EvaluationError, score_assessment
from copilot.evaluation.generation.schema import (
    benchmark_digest,
    load_generation_benchmark,
    verify_generation_regions,
)
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.llm import LLMClient, LLMError, create_llm_client
from copilot.llm.retry import RetryPolicy
from copilot.security import SecurityError
from copilot.utils.atomic_file import write_text_atomically

DEFAULT_OUTPUT_DIR = "data/generation-eval/primary"
CopilotBench = "generation-eval/benchmarks/copilot_generation_055a8d5.jsonl"
SyntheticBench = "generation-eval/benchmarks/inventory_orders_generation.jsonl"

#: Replaced in tests; the only place a hosted client is constructed.
ClientFactory = Callable[[Settings], LLMClient]
create_client: ClientFactory = create_llm_client


def open_retriever(spec: RepositorySpec, root: Path, settings: Settings):
    """Open the repository's index (loads the local embedding model). Replaced in tests."""
    from copilot.embeddings.factory import create_embedder
    from copilot.retrieval import Retriever

    return Retriever.open(
        root / str(spec.index_dir), root / spec.source_root,
        embedder=create_embedder(settings), settings=settings,
        ignore_directories=spec.ignore_directories,
    )  # fmt: skip


# --------------------------------------------------------------------------- loading


def _specs(root: Path) -> tuple[RepositorySpec, ...]:
    out: list[RepositorySpec] = []
    for bench_path, source_root, index_dir, index_id, independent in (
        (CopilotBench, "data/snapshots/a282ec471ebf1387cb4feafcfe732ad24a431c7a/tree",
         "data/indexes/4aaeab9c650ab76c", "4aaeab9c650ab76c", False),
        (SyntheticBench, "generation-eval/repos/inventory-orders", None, None, False),
    ):  # fmt: skip
        bench = load_generation_benchmark(root / bench_path)
        meta = bench.meta
        assert meta is not None, f"{bench_path} has no meta file"
        out.append(
            RepositorySpec(
                repository_id=meta.repository_id, kind=meta.source_kind,
                repository_name=meta.repository_name, source_root=source_root,
                benchmark_path=bench_path, benchmark_digest=benchmark_digest(bench),
                repository_fingerprint=meta.repository_fingerprint, index_dir=index_dir,
                index_id=index_id, ignore_directories=meta.ignore_directories,
                independent=independent,
            )
        )  # fmt: skip
    return tuple(out)


def _files_for(spec: RepositorySpec, root: Path, settings: Settings):
    policy = IngestionPolicy.from_settings(settings).with_extra_ignored_directories(
        *spec.ignore_directories
    )
    return ingest_repository(
        root / spec.source_root, policy, repository_name=spec.repository_name
    ).files


def _check_benchmark(spec: RepositorySpec, root: Path, settings: Settings):
    bench = load_generation_benchmark(root / spec.benchmark_path)
    if benchmark_digest(bench) != spec.benchmark_digest:
        raise BenchmarkFormatError(
            f"{spec.repository_id}: the benchmark differs from the one frozen in the protocol"
        )
    files = _files_for(spec, root, settings)
    problems = verify_generation_regions(bench, files)
    if problems:
        raise BenchmarkFormatError(
            f"{spec.repository_id}: {len(problems)} region problem(s): {problems[0]}"
        )
    return bench, files


def _selected(protocol: PrimaryProtocol, names: Sequence[str]) -> tuple[RepositorySpec, ...]:
    if not names:
        return protocol.repositories
    chosen = tuple(s for s in protocol.repositories if s.repository_id in names)
    unknown = set(names) - {s.repository_id for s in chosen}
    if unknown:
        raise ValueError(f"unknown repository: {', '.join(sorted(unknown))}")
    return chosen


def _contexts(
    protocol: PrimaryProtocol, root: Path, settings: Settings, names: Sequence[str],
    *, with_retriever: bool,
) -> list[RepositoryContext]:  # fmt: skip
    out: list[RepositoryContext] = []
    for spec in _selected(protocol, names):
        bench, _ = _check_benchmark(spec, root, settings)
        retriever = None
        if with_retriever:
            if spec.index_dir is None:
                raise RunMismatchError(
                    f"{spec.repository_id}: no index is configured; build one first "
                    "(see docs/generation-evaluation.md)"
                )
            retriever = open_retriever(spec, root, settings)
        out.append(
            RepositoryContext(
                spec, bench.questions, spec.benchmark_digest, retriever, spec.index_id
            )
        )
    return out


def _resolved(protocol: PrimaryProtocol, confirm: str | None) -> PrimaryProtocol:
    if not confirm:
        return protocol
    provider, _, model = confirm.partition("/")
    return protocol.with_model(provider, model)


def _historical(
    protocol: PrimaryProtocol, contexts: Sequence[RepositoryContext], files: Sequence[str],
    root: Path,
):  # fmt: skip
    """``(sides by repository, human-readable report lines)`` for verified historical reuse."""
    sides: dict[str, dict] = {}
    lines: list[str] = []
    for name in files:
        path = root / name
        run = load_comparison(path)
        ctx = next((c for c in contexts if c.spec.repository_name == run.repository_name), None)
        if ctx is None or ctx.spec.index_dir is None:
            lines.append(f"{path.name}: not a historical file of a configured repository")
            continue
        # Reuse is evaluated for the model the protocol confirms; if none is confirmed yet, for
        # the model of the file itself (to report what *would* be reusable).
        provider = protocol.llm_provider or run.llm_provider
        model = protocol.llm_model or run.llm_model
        target = frozen_target(
            questions={q.id: q.question for q in ctx.questions},
            repository_name=run.repository_name or "",
            index_id=ctx.index_id or "", llm_provider=provider, llm_model=model,
            max_output_tokens=protocol.max_output_tokens, temperature=protocol.temperature,
            top_k=protocol.top_k, plain_prompt_version=protocol.plain_prompt_version,
            rag_prompt_version=protocol.rag_prompt_version,
        )  # fmt: skip
        report = check_historical_run(
            run, target, source_name=path.name, index_dir=root / ctx.spec.index_dir
        )
        usable = {arm: report.reusable(arm) for arm in ("plain", "rag")}
        basis = "confirmed model" if protocol.model_confirmed else "the file's own model"
        lines.append(
            f"{path.name}: successful sides {dict(report.successful_sides)}; reusable under "
            f"{basis}: plain {len(usable['plain'])}, rag {len(usable['rag'])}"
            + ("" if protocol.model_confirmed else " (model not confirmed: nothing is reused)")
        )
        if protocol.model_confirmed:
            sides.setdefault(ctx.spec.repository_id, {}).update(
                historical_sides(run.entries, usable, path.name)
            )
    return sides, lines


# --------------------------------------------------------------------------- commands


def _cmd_protocol_init(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    target = root / args.protocol
    if target.exists() and not args.force:
        print(
            f"{target.name} already exists; the frozen protocol is not overwritten", file=sys.stderr
        )
        return 2
    protocol = default_protocol(_specs(root))
    write_text_atomically(target, dump_protocol(protocol))
    print(f"wrote {args.protocol}\nprotocol hash (model unconfirmed): {protocol_hash(protocol)}")
    return 0


def _cmd_plan(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    protocol = _resolved(load_protocol(root / args.protocol), args.confirm_model)
    contexts = _contexts(protocol, root, settings, args.repository, with_retriever=False)
    historical, notes = _historical(protocol, contexts, args.historical, root)
    out_dir = root / args.output_dir
    plan = plan_run(contexts, protocol, output_dir=out_dir, historical=historical)
    print("DRY RUN - no provider is constructed or called")
    print(f"protocol: {protocol.name}  hash: {protocol_hash(protocol)}")
    model = (
        f"{protocol.llm_provider}/{protocol.llm_model}"
        if protocol.model_confirmed
        else "NOT CONFIRMED"
    )
    print(f"generation model: {model}  max_output_tokens: {protocol.max_output_tokens}  "
          f"temperature: {protocol.temperature}  top_k: {protocol.top_k}")  # fmt: skip
    for line in plan.lines:
        print(
            f"- {line.repository_id}: {line.questions} questions ({line.answerable} answerable, "
            f"{line.unanswerable} unanswerable); generation calls <= {line.generation_calls} "
            f"({line.reused_sides} sides already available); judge calls <= {line.judge_calls}; "
            f"run file {line.artifact}"
        )
    print(f"total generation calls (upper bound): {plan.generation_calls}")
    print(f"total judge calls (upper bound): {plan.judge_calls}")
    for note in notes:
        print(f"historical: {note}")
    print("judge artifacts:", ", ".join(
        judgement_path(out_dir, c.spec.repository_id).as_posix() for c in contexts))  # fmt: skip
    return 0


def _retry(args: argparse.Namespace) -> RetryPolicy:
    return RetryPolicy(
        initial_delay_seconds=args.retry_initial_delay,
        max_delay_seconds=args.retry_max_delay,
        max_attempts=args.retry_max_attempts,
    )


def _cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    if not args.confirm_model:
        print("refusing to run: pass --confirm-model PROVIDER/MODEL to confirm the model that "
              "generates both arms", file=sys.stderr)  # fmt: skip
        return 2
    protocol = _resolved(load_protocol(root / args.protocol), args.confirm_model)
    llm = create_client(settings)
    if (llm.provider, llm.model) != (protocol.llm_provider, protocol.llm_model):
        print("refusing to run: the configured client is not the confirmed model", file=sys.stderr)
        return 2
    contexts = _contexts(protocol, root, settings, args.repository, with_retriever=True)
    historical, notes = _historical(protocol, contexts, args.historical, root)
    for note in notes:
        print(f"historical: {note}", file=sys.stderr)
    runs = run_repositories(
        contexts, protocol, llm, settings_for(protocol, settings),
        output_dir=root / args.output_dir,
        historical=historical, cache=ResponseCache(root / args.output_dir / "cache-generation"),
        retry_policy=_retry(args),
        max_consecutive_rate_limit_failures=args.max_consecutive_rate_limit_failures or None,
        progress=lambda m: print(m, file=sys.stderr), delay_seconds=args.delay_seconds,
    )  # fmt: skip
    for repo, run in runs.items():
        print(
            f"{repo}: {run.side_counts()}"
            + (f"  STOPPED: {run.stop_reason}" if run.stop_reason else "")
        )
    return 3 if any(r.stop_reason for r in runs.values()) else 0


def _cmd_judge(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    protocol = _resolved(load_protocol(root / args.protocol), args.confirm_model)
    if not args.judge_model:
        print("refusing to judge: pass --judge-model (a different model family from the "
              "generator is recommended)", file=sys.stderr)  # fmt: skip
        return 2
    llm = create_client(settings.model_copy(update={"llm_model": args.judge_model}))
    judge = LLMJudge(
        llm, max_output_tokens=protocol.judge.max_output_tokens, retry_policy=_retry(args),
        cache=JudgeCache(root / args.output_dir / "cache-judge"),
    )  # fmt: skip
    out_dir = root / args.output_dir
    for spec in _selected(protocol, args.repository):
        bench, files = _check_benchmark(spec, root, settings)
        run = load_run(run_path(out_dir, spec.repository_id))
        questions = {q.id: q for q in bench.questions}
        refs = {q.id: build_reference(q, files) for q in bench.questions}
        path = judgement_path(out_dir, spec.repository_id)
        prior = load_judgements(path) if path.is_file() else None
        done = judge_run(
            run, questions, refs, judge, checkpoint_path=path, resume_from=prior,
            seed=protocol.audit_seed, progress=lambda m: print(m, file=sys.stderr),
        )  # fmt: skip
        print(f"{spec.repository_id}: {done.counts()}")
    return 0


def _repo_inputs(protocol: PrimaryProtocol, root: Path, out_dir: Path, settings: Settings,
                 names: Sequence[str]) -> tuple[list[RepoInputs], dict[str, str]]:  # fmt: skip
    inputs: list[RepoInputs] = []
    files_by_label: dict[str, str] = {}
    for spec in _selected(protocol, names):
        bench, files = _check_benchmark(spec, root, settings)
        run_file = run_path(out_dir, spec.repository_id)
        run = load_run(run_file)
        if run.protocol_hash != protocol_hash(protocol):
            raise RunMismatchError(f"{spec.repository_id}: the run belongs to another protocol")
        judge_file = judgement_path(out_dir, spec.repository_id)
        judgements = load_judgements(judge_file) if judge_file.is_file() else None
        inputs.append(
            RepoInputs(
                run=run,
                questions={q.id: q for q in bench.questions},
                inventory=build_inventory(files, spec.repository_id),
                judgements=judgements,
                run_file=run_file.name,
                judgement_file=judge_file.name if judgements else None,
                independent=spec.independent,
                kind=spec.kind,
            )  # fmt: skip
        )
        files_by_label[f"run:{spec.repository_id}"] = str(run_file)
        if judgements is not None:
            files_by_label[f"judgements:{spec.repository_id}"] = str(judge_file)
        files_by_label[f"benchmark:{spec.repository_id}"] = str(root / spec.benchmark_path)
    return inputs, files_by_label


def _cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    out_dir = root / args.output_dir
    stored = load_protocol(root / args.protocol)
    # the run's protocol hash includes the confirmed model; recover it from the run files
    first = load_run(run_path(out_dir, _selected(stored, args.repository)[0].repository_id))
    protocol = stored.with_model(first.llm_provider, first.llm_model)
    inputs, labels = _repo_inputs(protocol, root, out_dir, settings, args.repository)
    report = build_report(inputs, protocol, input_files=labels)
    suffix = "-TEST-ARTIFACT" if report["test_artifact"] else ""
    json_path = out_dir / f"report{suffix}.json"
    md_path = out_dir / f"report{suffix}.md"
    write_text_atomically(json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_text_atomically(md_path, render_markdown(report))
    print(f"wrote {json_path} and {md_path}")
    return 0


def _paired_answers(inputs: Sequence[RepoInputs]):
    """Questions both arms answered with text (a system abstention would reveal the arm)."""
    for item in inputs:
        for entry in item.run.entries:
            if entry.plain.has_text and entry.rag.has_text:
                yield item, entry


def _cmd_audit_worksheet(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    out_dir = root / args.output_dir
    stored = load_protocol(root / args.protocol)
    first = load_run(run_path(out_dir, _selected(stored, args.repository)[0].repository_id))
    protocol = stored.with_model(first.llm_provider, first.llm_model)
    inputs, _ = _repo_inputs(protocol, root, out_dir, settings, args.repository)
    paired = list(_paired_answers(inputs))
    units = [AuditUnit(i.run.repository_id, e.question_id, e.answerable) for i, e in paired]
    sample = sample_audit(units, fraction=protocol.audit_fraction, seed=protocol.audit_seed)
    chosen = {(u.repository_id, u.question_id) for u in sample}
    answers, questions, references = [], {}, {}
    for item, entry in paired:
        key = (item.run.repository_id, entry.question_id)
        if key not in chosen:
            continue
        spec = protocol.repository(item.run.repository_id)
        files = _files_for(spec, root, settings)
        question = item.questions[entry.question_id]
        questions[key] = question.question
        references[key] = build_reference(question, files)
        for arm in ("plain", "rag"):
            answers.append(AuditAnswer(key[0], key[1], arm, getattr(entry, arm).answer or ""))
    sheet = build_worksheet(answers, questions, references, seed=protocol.audit_seed)
    suffix = "-TEST-ARTIFACT" if any(i.run.test_artifact for i in inputs) else ""
    write_text_atomically(out_dir / f"audit-worksheet{suffix}.csv", sheet.to_csv())
    write_text_atomically(out_dir / f"audit-key{suffix}.json", sheet.key_json())
    print(f"sampled {len(chosen)} of {len(units)} paired questions ({len(answers)} answers); "
          f"the key is audit-key{suffix}.json - do not open it while rating")  # fmt: skip
    return 0


def _cmd_audit_agree(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    out_dir = root / args.output_dir
    stored = load_protocol(root / args.protocol)
    first = load_run(run_path(out_dir, _selected(stored, args.repository)[0].repository_id))
    protocol = stored.with_model(first.llm_provider, first.llm_model)
    key = {e["item_id"]: e for e in json.loads(Path(args.key).read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(io.StringIO(Path(args.worksheet).read_text(encoding="utf-8"))))
    human_scores, judge_scores, human_abs, judge_abs, human_facts, judge_facts = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    rated = 0
    for spec in _selected(protocol, args.repository):
        bench, _ = _check_benchmark(spec, root, settings)
        questions = {q.id: q for q in bench.questions}
        judged = load_judgements(judgement_path(out_dir, spec.repository_id)).by_key()
        for row in rows:
            entry = key.get(row["item_id"])
            if entry is None or entry["repository_id"] != spec.repository_id:
                continue
            question = questions[entry["question_id"]]
            human = parse_human_row(row, [f.id for f in question.key_facts], args.rater)
            machine = judged.get((question.id, entry["arm"]))
            if human is None or machine is None or machine.assessment is None:
                continue
            rated += 1
            human_scores.append(score_assessment(question, human).score)
            judge_scores.append(score_assessment(question, machine.assessment).score)
            human_abs.append(human.abstained)
            judge_abs.append(machine.assessment.abstained)
            for fid, verdict in human.fact_verdicts.items():
                human_facts.append(verdict.value == "supported")
                judge_facts.append(machine.assessment.fact_verdicts[fid].value == "supported")
    if rated == 0:
        print("no rated rows were found in the worksheet", file=sys.stderr)
        return 2
    result = {
        "rated_answers": rated,
        "correctness_score_0_2": agreement(human_scores, judge_scores, ordinal=True).to_json(),
        "abstained": agreement(human_abs, judge_abs).to_json(),
        "key_fact_supported": agreement(human_facts, judge_facts).to_json(),
        "note": "kappa is null when a rater never varies (undefined); raw agreement is given",
    }
    path = out_dir / "audit-agreement.json"
    write_text_atomically(path, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


# --------------------------------------------------------------------------- entry point


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m copilot.evaluation.generation",
                                     description=__doc__)  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", default=".", help="project root (default: current directory)")
        p.add_argument("--protocol", default=PROTOCOL_PATH)
        p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
        p.add_argument("--repository", action="append", default=[], help="limit to a repository id")

    def retry_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--retry-initial-delay", type=float, default=5.0)
        p.add_argument("--retry-max-delay", type=float, default=60.0)
        p.add_argument("--retry-max-attempts", type=int, default=5)

    init = sub.add_parser("protocol-init", help="write the frozen primary protocol")
    common(init)
    init.add_argument("--force", action="store_true")

    plan = sub.add_parser("plan", help="dry run (no provider)")
    common(plan)
    plan.add_argument("--confirm-model", help="PROVIDER/MODEL (optional for a plan)")
    plan.add_argument("--historical", action="append", default=[], help="a comparison file")

    run = sub.add_parser("run", help="generate answers (hosted)")
    common(run)
    retry_flags(run)
    run.add_argument("--confirm-model", help="PROVIDER/MODEL that generates both arms")
    run.add_argument("--historical", action="append", default=[])
    run.add_argument("--delay-seconds", type=float, default=0.0)
    run.add_argument("--max-consecutive-rate-limit-failures", type=int, default=3)

    judge = sub.add_parser("judge", help="judge answers (hosted)")
    common(judge)
    retry_flags(judge)
    judge.add_argument("--confirm-model", help="the generation model (to resolve the protocol)")
    judge.add_argument("--judge-model", help="the judge model id")

    report = sub.add_parser("report", help="analysis report (offline)")
    common(report)

    worksheet = sub.add_parser("audit-worksheet", help="sample and write the blinded worksheet")
    common(worksheet)

    agree = sub.add_parser("audit-agree", help="agreement between a filled worksheet and the judge")
    common(agree)
    agree.add_argument("--worksheet", required=True)
    agree.add_argument("--key", required=True)
    agree.add_argument("--rater", default="rater-1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    setup_logging()
    settings = get_settings()
    commands = {
        "protocol-init": _cmd_protocol_init,
        "plan": _cmd_plan,
        "run": _cmd_run,
        "judge": _cmd_judge,
        "report": _cmd_report,
        "audit-worksheet": _cmd_audit_worksheet,
        "audit-agree": _cmd_audit_agree,
    }
    try:
        return commands[args.command](args, settings)
    except KeyboardInterrupt:
        print("\ninterrupted: finished sides are checkpointed; re-run to resume", file=sys.stderr)
        return 130
    except (
        SecurityError, LLMError, ValueError, BenchmarkFormatError, RunMismatchError,
        JudgementMismatchError, EvaluationError, FileNotFoundError,
    ) as exc:  # fmt: skip
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
