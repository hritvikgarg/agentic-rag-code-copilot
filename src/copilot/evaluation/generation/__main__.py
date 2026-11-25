"""Commands for the plain-LLM vs RAG answer experiment (Milestones 11A / 11B).

``python -m copilot.evaluation.generation COMMAND``:

* ``protocol-init``  write the frozen primary protocol (maintainers; once, before generation).
* ``plan``           dry run: repositories, question counts, generation and judge calls, artifact
                     paths and verified historical reuse. Constructs no provider; offline.
* ``run``            generate answers (hosted model; requires ``--confirm-model PROVIDER/MODEL``).
* ``judge``          judge a run's answers (hosted Gemini or local Ollama judge; the judge is chosen
                     with ``--judge-provider``/``--judge-model``, never by the generation model).
* ``judge-preflight`` check a local judge runtime (metadata only; never pulls a model).
* ``judge-controls`` run the synthetic judge controls (separate namespace; local or hosted judge).
* ``judge-validate`` evaluate the artifacts against the predeclared validation specification
                     (offline; reads existing artifacts only).
* ``judge-compare``  per-answer comparison of two judges on the same generated answers (offline).
* ``report``         analysis from run and judgement files (offline).
* ``audit-worksheet`` / ``audit-agree`` / ``audit-rule-check``  the human audit (offline).

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
    build_worksheet,
    sample_audit,
)
from copilot.evaluation.generation.audit_agree import evaluate_agreement
from copilot.evaluation.generation.audit_population import (
    classify_questions,
    describe_empty_sample,
    finalization_problems,
    pilot_problems,
    population_report,
)
from copilot.evaluation.generation.audit_rules import (
    build_rule_worksheet,
    evaluate_rule_check,
    rule_items,
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
from copilot.evaluation.generation.judge_compare import compare_judgements
from copilot.evaluation.generation.judge_compare import render_text as render_comparison
from copilot.evaluation.generation.judge_controls import (
    CONTROLS_PATH,
    ControlsError,
    check_against_benchmarks,
    controls_sha256,
    load_controls,
    load_results,
    results_path,
    run_controls,
)
from copilot.evaluation.generation.judge_preflight import run_preflight
from copilot.evaluation.generation.judge_selection import (
    JudgeSelection,
    check_judgements_dir,
    resolve_judge,
)
from copilot.evaluation.generation.judge_validation import (
    VALIDATION_SPEC_PATH,
    ValidationInputs,
    ValidationSpecError,
    evaluate_primary_audit,
    evaluate_validation,
    load_validation_spec,
    spec_sha256,
)
from copilot.evaluation.generation.judging import (
    JudgementMismatchError,
    judge_run,
    judgement_path,
    load_judgements,
)
from copilot.evaluation.generation.pilot import (
    DEFAULT_PILOT_OUTPUT_DIR,
    PILOT_SPEC_PATH,
    PilotMismatchError,
    PilotSpec,
    build_pilot_spec,
    check_namespaces,
    dump_pilot_spec,
    load_pilot_spec,
    restrict_context,
    verify_pilot_spec,
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
from copilot.evaluation.generation.rubric import EvaluationError
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
ClientFactory = Callable[..., LLMClient]
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
        (SyntheticBench, "generation-eval/repos/inventory-orders",
         "data/indexes/d115dc06e63659a2", "d115dc06e63659a2", False),
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
    *, with_retriever: bool, pilot: PilotSpec | None = None,
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
            actual = getattr(getattr(retriever, "index", None), "index_id", None)
            if actual is not None and spec.index_id is not None and actual != spec.index_id:
                raise RunMismatchError(
                    f"{spec.repository_id}: the opened index is {actual}, the protocol expects "
                    f"{spec.index_id}; the protocol must be re-frozen before generating"
                )
        ctx = RepositoryContext(
            spec, bench.questions, spec.benchmark_digest, retriever, spec.index_id
        )
        out.append(restrict_context(ctx, pilot) if pilot is not None else ctx)
    return out


def _pilot(args: argparse.Namespace, root: Path, stored: PrimaryProtocol) -> PilotSpec | None:
    """The verified pilot specification for ``--pilot`` (else ``None``); checks the namespace."""
    if not getattr(args, "pilot", False):
        return None
    check_namespaces(root / args.output_dir, root / DEFAULT_OUTPUT_DIR)
    spec = load_pilot_spec(root / args.pilot_spec)
    verify_pilot_spec(
        spec,
        protocol_hash(stored),
        {r.repository_id: r.benchmark_digest for r in stored.repositories},
    )
    return spec


def _expect_scope(run, pilot: PilotSpec | None, name: str) -> None:
    expected = "pilot" if pilot is not None else "primary"
    if run.scope != expected:
        raise RunMismatchError(
            f"{name}: this is a {run.scope!r} run but a {expected!r} run was requested; pilot and "
            "primary artifacts are never mixed"
        )


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


def _cmd_pilot_init(args: argparse.Namespace, settings: Settings) -> int:
    """Record the pilot questions (offline; reads only ids and answerability)."""
    root = Path(args.root)
    stored = load_protocol(root / args.protocol)
    benchmarks, digests = {}, {}
    for spec in stored.repositories:
        bench = load_generation_benchmark(root / spec.benchmark_path)
        benchmarks[spec.repository_id] = bench.questions
        digests[spec.repository_id] = benchmark_digest(bench)
        if digests[spec.repository_id] != spec.benchmark_digest:
            raise BenchmarkFormatError(
                f"{spec.repository_id}: the benchmark differs from the protocol"
            )
    spec = build_pilot_spec(stored.name, protocol_hash(stored), benchmarks, digests)
    target = root / args.pilot_spec
    text = dump_pilot_spec(spec)
    if target.exists():
        if target.read_text(encoding="utf-8") == text:
            print(f"{args.pilot_spec} already records exactly this selection (unchanged)")
            return 0
        print(f"{target.name} already exists and differs; a recorded pilot is never overwritten",
              file=sys.stderr)  # fmt: skip
        return 2
    write_text_atomically(target, text)
    for repo, ids in spec.questions.items():
        print(f"{repo}: {', '.join(ids)}")
    print(f"wrote {args.pilot_spec}; bounds {spec.bounds}")
    return 0


def _cmd_plan(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    stored = load_protocol(root / args.protocol)
    pilot = _pilot(args, root, stored)
    if pilot is not None and args.historical:
        raise ValueError("--historical is not allowed with --pilot (a pilot generates afresh)")
    protocol = _resolved(stored, args.confirm_model)
    contexts = _contexts(
        protocol, root, settings, args.repository, with_retriever=False, pilot=pilot
    )
    historical, notes = _historical(protocol, contexts, args.historical, root)
    out_dir = root / args.output_dir
    plan = plan_run(contexts, protocol, output_dir=out_dir, historical=historical)
    print("DRY RUN - no provider is constructed or called")
    if pilot is not None:
        print(f"PILOT {pilot.name}: separate namespace {args.output_dir}; not part of the primary")
        for repo, ids in pilot.questions.items():
            print(f"  pilot questions {repo}: {', '.join(ids)}")
        print(f"  bounds: {pilot.bounds}")
    print(f"protocol: {protocol.name}  hash: {protocol_hash(protocol)}")
    model = (
        f"{protocol.llm_provider}/{protocol.llm_model}"
        if protocol.model_confirmed
        else "NOT CONFIRMED"
    )
    print(f"generation model: {model}  max_output_tokens: {protocol.max_output_tokens}  "
          f"temperature: {protocol.temperature}  top_k: {protocol.top_k}")  # fmt: skip
    for ctx in contexts:
        index_dir = ctx.spec.index_dir
        present = index_dir is not None and (root / index_dir / "manifest.json").is_file()
        print(
            f"index {ctx.spec.repository_id}: {index_dir} ({'present' if present else 'MISSING'})"
        )
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
    stored = load_protocol(root / args.protocol)
    pilot = _pilot(args, root, stored)
    if pilot is not None and args.historical:
        print("refusing to run: --historical is not allowed with --pilot", file=sys.stderr)
        return 2
    protocol = _resolved(stored, args.confirm_model)
    llm = create_client(settings)
    if (llm.provider, llm.model) != (protocol.llm_provider, protocol.llm_model):
        print("refusing to run: the configured client is not the confirmed model", file=sys.stderr)
        return 2
    contexts = _contexts(
        protocol, root, settings, args.repository, with_retriever=True, pilot=pilot
    )
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


def _judge_selection(
    args: argparse.Namespace,
    settings: Settings,
    protocol: PrimaryProtocol,
    root: Path,
    pilot: PilotSpec | None,
    *,
    need_directory: bool = True,
) -> JudgeSelection:
    return resolve_judge(
        args,
        settings,
        protocol_max_output_tokens=protocol.judge.max_output_tokens,
        root=root,
        out_dir=root / args.output_dir,
        pilot=pilot is not None,
        primary_dir=root / DEFAULT_OUTPUT_DIR,
        pilot_dir=root / DEFAULT_PILOT_OUTPUT_DIR,
        need_directory=need_directory,
    )


def _judge_client(selection: JudgeSelection) -> LLMClient:
    """The judge client: local judges get JSON-mode output; hosted behaviour is unchanged."""
    if selection.local:
        return create_client(selection.settings, structured_output="json")
    return create_client(selection.settings)


def _cmd_judge(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    stored = load_protocol(root / args.protocol)
    pilot = _pilot(args, root, stored)
    protocol = _resolved(stored, args.confirm_model)
    selection = _judge_selection(args, settings, protocol, root, pilot)
    llm = _judge_client(selection)
    judge = LLMJudge(
        llm, max_output_tokens=selection.max_output_tokens, retry_policy=_retry(args),
        cache=JudgeCache(selection.cache_dir) if selection.cache_dir is not None else None,
    )  # fmt: skip
    out_dir = root / args.output_dir
    for spec in _selected(protocol, args.repository):
        bench, files = _check_benchmark(spec, root, settings)
        run = load_run(run_path(out_dir, spec.repository_id))
        _expect_scope(run, pilot, spec.repository_id)
        questions = {q.id: q for q in bench.questions}
        refs = {q.id: build_reference(q, files) for q in bench.questions}
        path = judgement_path(selection.judgements_dir, spec.repository_id)
        prior = load_judgements(path) if path.is_file() else None
        done = judge_run(
            run, questions, refs, judge, checkpoint_path=path, resume_from=prior,
            seed=protocol.audit_seed, progress=lambda m: print(m, file=sys.stderr),
        )  # fmt: skip
        line = f"{spec.repository_id}: {done.counts()}"
        if done.context_failures():
            line += (
                f"  context-limit failures: {done.context_failures()} (recorded as failed and "
                "excluded from the valid judgements)"
            )
        print(line)
    return 0


def _cmd_judge_preflight(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    protocol = load_protocol(root / args.protocol)
    selection = _judge_selection(args, settings, protocol, root, None, need_directory=False)
    if not selection.local:
        print("error: judge-preflight applies to a local judge (--judge-provider ollama)",
              file=sys.stderr)  # fmt: skip
        return 2
    llm = _judge_client(selection)
    result = run_preflight(
        llm, temperature=0.0, max_output_tokens=selection.max_output_tokens,
        expect_quantization=args.expect_quantization, require_gpu=args.require_gpu,
        warm_up=args.warm_up,
    )  # fmt: skip
    print(json.dumps(result, indent=2))
    for warning in result["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0 if result["ok"] else 3


def _benchmarks(protocol: PrimaryProtocol, root: Path, settings: Settings, names: Sequence[str]):
    """``(questions, references)`` per repository id, from the verified frozen benchmarks."""
    questions: dict[str, dict] = {}
    references: dict[str, dict] = {}
    for spec in _selected(protocol, names):
        bench, files = _check_benchmark(spec, root, settings)
        questions[spec.repository_id] = {q.id: q for q in bench.questions}
        references[spec.repository_id] = {q.id: build_reference(q, files) for q in bench.questions}
    return questions, references


def _cmd_judge_controls(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    protocol = load_protocol(root / args.protocol)
    if args.repeat:  # an independent second pass: no cache, model unloaded between requests
        args.no_cache = True
        if args.judge_keep_alive is None:
            args.judge_keep_alive = "0"
    selection = _judge_selection(args, settings, protocol, root, None)
    primary = (root / DEFAULT_OUTPUT_DIR).resolve()
    target = selection.judgements_dir.resolve()
    if target == primary or target.is_relative_to(primary):
        raise ValueError("the control directory must not be inside the primary directory")
    controls = load_controls(root / CONTROLS_PATH)
    validation_hash = spec_sha256(root / VALIDATION_SPEC_PATH)
    questions, references = _benchmarks(protocol, root, settings, [])
    pilot_path = root / PILOT_SPEC_PATH
    excluded = load_pilot_spec(pilot_path).questions if pilot_path.is_file() else {}
    check_against_benchmarks(controls, questions, excluded)
    llm = _judge_client(selection)
    judge = LLMJudge(
        llm, max_output_tokens=selection.max_output_tokens, retry_policy=_retry(args),
        cache=JudgeCache(selection.cache_dir) if selection.cache_dir is not None else None,
    )  # fmt: skip
    label = "repeat" if args.repeat else "pass1"
    path = results_path(selection.judgements_dir, label)
    prior = load_results(path) if path.is_file() else None
    run = run_controls(
        controls, judge, questions, references, pass_label=label,
        spec_sha256=controls_sha256(root / CONTROLS_PATH), validation_spec_sha256=validation_hash,
        checkpoint_path=path, prior=prior, progress=lambda m: print(m, file=sys.stderr),
    )  # fmt: skip
    print(json.dumps(run.summary(), indent=2))
    print(f"wrote {path}")
    return 0


def _cmd_judge_validate(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    protocol = load_protocol(root / args.protocol)
    spec = load_validation_spec(root / VALIDATION_SPEC_PATH)
    spec_hash = spec_sha256(root / VALIDATION_SPEC_PATH)
    inputs = ValidationInputs(
        controls_dir=args.controls_dir, controls_repeat_dir=args.controls_repeat_dir,
        pilot_dir=args.pilot_dir, pilot_repeat_dir=args.pilot_repeat_dir,
    )  # fmt: skip

    def controls(directory: str | None, label: str):
        if not directory:
            return None
        run = load_results(results_path(root / directory, label))
        if run.validation_spec_sha256 != spec_hash:
            raise ControlsError(
                f"{label}: the controls were judged under a different validation specification"
            )
        return run

    inputs.controls = controls(args.controls_dir, "pass1")
    inputs.controls_repeat = controls(args.controls_repeat_dir, "repeat")
    inputs.questions, _ = _benchmarks(protocol, root, settings, args.repository)

    def pilot_files(directory: str | None):
        found = {}
        for repo in inputs.questions:
            path = judgement_path(root / directory, repo) if directory else None
            if path is not None and path.is_file():
                found[repo] = load_judgements(path)
        return found

    inputs.pilot = pilot_files(args.pilot_dir)
    inputs.pilot_repeat = pilot_files(args.pilot_repeat_dir)
    agreement_path = (
        Path(args.agreement) if args.agreement
        else (root / args.pilot_dir / "audit-agreement-PILOT.json" if args.pilot_dir else None)
    )  # fmt: skip
    if agreement_path is not None and agreement_path.is_file():
        inputs.agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    report = evaluate_validation(spec, inputs, spec_hash)
    if args.primary_agreement:
        primary = json.loads(Path(args.primary_agreement).read_text(encoding="utf-8"))
        report["primary_audit"] = evaluate_primary_audit(spec, primary)
    anchor = args.pilot_dir or args.controls_dir
    target = (
        Path(args.report)
        if args.report
        else (root / anchor / "judge-validation-report.json" if anchor else None)
    )  # noqa: E501
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if target is not None:
        write_text_atomically(target, text + "\n")
    print(text)
    primary_fail = report.get("primary_audit", {}).get("status") == "FAIL"
    if report["overall"] == "FAIL" or primary_fail:
        return 3
    return 0 if report["overall"] == "PILOT_CRITERIA_MET" else 4


def _cmd_judge_compare(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    out_dir = root / args.output_dir
    stored = load_protocol(root / args.protocol)
    pilot = _pilot(args, root, stored)
    protocol = _resolved_for_audit(args, root, out_dir)
    dir_a = root / args.dir_a if args.dir_a else out_dir
    dir_b = root / args.dir_b
    if dir_a.resolve() == dir_b.resolve():
        print("error: --dir-a and --dir-b are the same directory", file=sys.stderr)
        return 2
    results: dict[str, object] = {}
    invalid = False
    for spec in _selected(protocol, args.repository):
        bench, _ = _check_benchmark(spec, root, settings)
        first = load_judgements(judgement_path(dir_a, spec.repository_id))
        second = load_judgements(judgement_path(dir_b, spec.repository_id))
        outcome = compare_judgements(first, second, {q.id: q for q in bench.questions})
        results[spec.repository_id] = outcome
        invalid = invalid or not outcome["identity_verified"]
        print(f"== {spec.repository_id}\n{render_comparison(outcome)}")
    target = dir_b / f"judge-compare{'-PILOT' if pilot is not None else ''}.json"
    write_text_atomically(target, json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {target}")
    return 3 if invalid else 0


def _repo_inputs(protocol: PrimaryProtocol, root: Path, out_dir: Path, settings: Settings,
                 names: Sequence[str], pilot: PilotSpec | None = None,
                 judgements_dir: Path | None = None, use_judgements: bool = True,
                 ) -> tuple[list[RepoInputs], dict[str, str]]:  # fmt: skip
    inputs: list[RepoInputs] = []
    files_by_label: dict[str, str] = {}
    for spec in _selected(protocol, names):
        bench, files = _check_benchmark(spec, root, settings)
        run_file = run_path(out_dir, spec.repository_id)
        run = load_run(run_file)
        _expect_scope(run, pilot, spec.repository_id)
        if run.protocol_hash != protocol_hash(protocol):
            raise RunMismatchError(f"{spec.repository_id}: the run belongs to another protocol")
        judge_file = judgement_path(judgements_dir or out_dir, spec.repository_id)
        judgements = (
            load_judgements(judge_file) if use_judgements and judge_file.is_file() else None
        )
        inputs.append(
            RepoInputs(
                run=run,
                questions={
                    q.id: q
                    for q in bench.questions
                    if pilot is None or q.id in pilot.questions.get(spec.repository_id, ())
                },
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
    pilot = _pilot(args, root, stored)
    # the run's protocol hash includes the confirmed model; recover it from the run files
    first = load_run(run_path(out_dir, _selected(stored, args.repository)[0].repository_id))
    _expect_scope(first, pilot, "first run file")
    protocol = stored.with_model(first.llm_provider, first.llm_model)
    judgements_dir = root / args.judgements_dir if args.judgements_dir else None
    if judgements_dir is not None:
        check_judgements_dir(
            judgements_dir, pilot=pilot is not None, primary_dir=root / DEFAULT_OUTPUT_DIR,
            pilot_dir=root / DEFAULT_PILOT_OUTPUT_DIR,
        )  # fmt: skip
    inputs, labels = _repo_inputs(
        protocol, root, out_dir, settings, args.repository, pilot, judgements_dir
    )
    report = build_report(
        inputs, protocol, input_files=labels, scope="pilot" if pilot is not None else "primary"
    )
    suffix = ("-PILOT" if pilot is not None else "") + (
        "-TEST-ARTIFACT" if report["test_artifact"] else ""
    )
    # A report that uses another judge's judgements is written next to those judgements, so it
    # can never replace the report of a different judge.
    report_dir = judgements_dir or out_dir
    json_path = report_dir / f"report{suffix}.json"
    md_path = report_dir / f"report{suffix}.md"
    write_text_atomically(json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    write_text_atomically(md_path, render_markdown(report))
    print(f"wrote {json_path} and {md_path}")
    return 0


def _paired_answers(inputs: Sequence[RepoInputs]):
    """Questions both arms answered with model text (a system abstention is verified separately)."""
    for item in inputs:
        for entry in item.run.entries:
            if entry.plain.has_text and entry.rag.has_text:
                yield item, entry


def _suffix(inputs: Sequence[RepoInputs], provisional: bool) -> str:
    return ("-PROVISIONAL" if provisional else "") + (
        "-TEST-ARTIFACT" if any(i.run.test_artifact for i in inputs) else ""
    )


def _resolved_for_audit(args: argparse.Namespace, root: Path, out_dir: Path):
    stored = load_protocol(root / args.protocol)
    first = load_run(run_path(out_dir, _selected(stored, args.repository)[0].repository_id))
    return stored.with_model(first.llm_provider, first.llm_model)


def _cmd_audit_worksheet(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    out_dir = root / args.output_dir
    stored = load_protocol(root / args.protocol)
    pilot = _pilot(args, root, stored)
    protocol = _resolved_for_audit(args, root, out_dir)
    # The pilot worksheet is drawn BEFORE any judging: no judgement file is read, so the human
    # ratings cannot be influenced by (or depend on) any judge. Every answer with text is rated.
    inputs, _ = _repo_inputs(
        protocol, root, out_dir, settings, args.repository, pilot, use_judgements=pilot is None
    )
    statuses = classify_questions(inputs, require_judge_rating=pilot is None)
    problems = pilot_problems(inputs) if pilot is not None else finalization_problems(inputs)
    provisional = bool(problems)
    if provisional and not args.provisional:
        print(
            f"error: the experiment is not finalized ({len(problems)} problem(s)); an official "
            "audit sample may only be drawn after generation and judging are complete. First "
            f"problem: {problems[0]}. Re-run with --provisional for a development-only sample.",
            file=sys.stderr,
        )
        return 2
    suffix = ("-PILOT" if pilot is not None else "") + _suffix(inputs, provisional)
    label = "PROVISIONAL (development only)" if provisional else None
    if pilot is not None:
        label = "PILOT (every answer, judge-independent)" + (" PROVISIONAL" if provisional else "")
    fraction = 1.0 if pilot is not None else protocol.audit_fraction
    eligible = {(s.repository_id, s.question_id) for s in statuses if s.eligible}
    paired = [(i, e) for i, e in _paired_answers(inputs)
              if (i.run.repository_id, e.question_id) in eligible]  # fmt: skip
    units = [AuditUnit(i.run.repository_id, e.question_id, e.answerable) for i, e in paired]
    sample = sample_audit(units, fraction=fraction, seed=protocol.audit_seed)
    report = population_report(inputs, statuses, sample, problems=problems, provisional=provisional)
    write_text_atomically(
        out_dir / f"audit-population{suffix}.json", json.dumps(report, indent=2) + "\n"
    )
    if not sample:
        print(f"error: {describe_empty_sample(report)}", file=sys.stderr)
        return 2  # never an empty worksheet reported as a success
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
    sheet = build_worksheet(
        answers, questions, references, seed=protocol.audit_seed, status_label=label
    )
    write_text_atomically(out_dir / f"audit-worksheet{suffix}.csv", sheet.to_csv())
    write_text_atomically(out_dir / f"audit-key{suffix}.json", sheet.key_json())
    rule_csv, rule_key, rule_sampled = build_rule_worksheet(
        rule_items(inputs), fraction=fraction, seed=protocol.audit_seed, status_label=label,
    )  # fmt: skip
    if rule_sampled:
        write_text_atomically(out_dir / f"audit-rule-worksheet{suffix}.csv", rule_csv)
        write_text_atomically(out_dir / f"audit-rule-key{suffix}.json", rule_key)
    total = report["total_benchmark_questions"]
    print(
        f"{'PROVISIONAL ' if provisional else ''}audit sample: {len(chosen)} question(s) "
        f"({len(answers)} answers) = {len(chosen) / len(eligible):.0%} of {len(eligible)} eligible "
        f"pair(s) and {len(chosen) / total:.0%} of {total} benchmark question(s); "
        f"{report['excluded_pairs']} excluded; population report: audit-population{suffix}.json"
    )
    print(
        f"rule-scored system abstentions: {report['rule_scored_system_abstentions']} "
        f"({len(rule_sampled)} sampled for verification in audit-rule-worksheet{suffix}.csv)"
    )
    print(f"the key is audit-key{suffix}.json - do not open it while rating")
    if pilot is not None:
        print("PILOT worksheet: drawn before any judging; rate every answer, then run "
              "audit-agree --pilot --judgements-dir ... after the judge has run")  # fmt: skip
    return 0


def _read_csv(path: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8"))))


def _cmd_audit_agree(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    out_dir = root / args.output_dir
    stored = load_protocol(root / args.protocol)
    pilot = _pilot(args, root, stored)
    judgements_dir = root / args.judgements_dir if args.judgements_dir else out_dir
    if args.judgements_dir:
        check_judgements_dir(
            judgements_dir, pilot=pilot is not None, primary_dir=root / DEFAULT_OUTPUT_DIR,
            pilot_dir=root / DEFAULT_PILOT_OUTPUT_DIR,
        )  # fmt: skip
    protocol = _resolved_for_audit(args, root, out_dir)
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    rows = _read_csv(args.worksheet)
    questions: dict[str, dict] = {}
    judged: dict[str, dict] = {}
    for spec in _selected(protocol, args.repository):
        bench, _ = _check_benchmark(spec, root, settings)
        questions[spec.repository_id] = {q.id: q for q in bench.questions}
        judged[spec.repository_id] = load_judgements(
            judgement_path(judgements_dir, spec.repository_id)
        ).by_key()
    result = evaluate_agreement(
        rows, key, questions, judged, rater=args.rater, pilot=pilot is not None
    )
    # Written beside the judgements it evaluates, so two judges' agreement files never collide.
    path = judgements_dir / f"audit-agreement{'-PILOT' if pilot is not None else ''}.json"
    write_text_atomically(path, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["rows_accounted_for"]:
        print("error: invalid, duplicate, unexpected or missing worksheet rows (see the report)",
              file=sys.stderr)  # fmt: skip
        return 3
    if result["rated_answers"] == 0:
        print("error: no eligible rated rows (a human rating and a valid judge rating)",
              file=sys.stderr)  # fmt: skip
        return 2
    return 0


def _cmd_audit_rule_check(args: argparse.Namespace, settings: Settings) -> int:
    root = Path(args.root)
    out_dir = root / args.output_dir
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    result = evaluate_rule_check(_read_csv(args.worksheet), key)
    write_text_atomically(out_dir / "audit-rule-check.json", json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    problems = (
        result["invalid_ratings"] or result["duplicate_item_ids"] or result["unexpected_item_ids"]
        or result["key_items_missing_from_worksheet"]
    )  # fmt: skip
    return 3 if problems else (2 if result["rated"] == 0 else 0)


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

    def pilot_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--pilot", action="store_true",
            help="the bounded pilot: predefined questions, separate output directory",
        )  # fmt: skip
        p.add_argument("--pilot-spec", default=PILOT_SPEC_PATH)

    init = sub.add_parser("protocol-init", help="write the frozen primary protocol")
    common(init)
    init.add_argument("--force", action="store_true")

    pilot_init = sub.add_parser("pilot-init", help="record the pilot questions (offline)")
    common(pilot_init)
    pilot_init.add_argument("--pilot-spec", default=PILOT_SPEC_PATH)

    plan = sub.add_parser("plan", help="dry run (no provider)")
    common(plan)
    pilot_flags(plan)
    plan.add_argument("--confirm-model", help="PROVIDER/MODEL (optional for a plan)")
    plan.add_argument("--historical", action="append", default=[], help="a comparison file")

    run = sub.add_parser("run", help="generate answers (hosted)")
    common(run)
    pilot_flags(run)
    retry_flags(run)
    run.add_argument("--confirm-model", help="PROVIDER/MODEL that generates both arms")
    run.add_argument("--historical", action="append", default=[])
    run.add_argument("--delay-seconds", type=float, default=0.0)
    run.add_argument("--max-consecutive-rate-limit-failures", type=int, default=3)

    def judge_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--judge-provider", choices=["gemini", "ollama"],
                       help="the judge's provider (default: COPILOT_LLM_PROVIDER); the generation "
                       "model is never changed")  # fmt: skip
        p.add_argument("--judge-model", help="the judge model id, e.g. llama3.1:8b")
        p.add_argument("--judgements-dir", help="where this judge's artifacts live (required for "
                       "a local judge; never the generation output directory)")  # fmt: skip
        p.add_argument("--judge-timeout", type=float, help="seconds per request (local: 300)")
        p.add_argument("--judge-num-ctx", type=int, help="local context window (default 8192)")
        p.add_argument("--judge-seed", type=int, help="local sampling seed (default 20260)")
        p.add_argument("--judge-keep-alive", help="how long the local model stays loaded, e.g. 30m")
        p.add_argument("--judge-max-output-tokens", type=int,
                       help="judge output budget (local default 1024)")  # fmt: skip
        p.add_argument("--no-cache", action="store_true", help="bypass the judge cache entirely "
                       "(repeatability checks)")  # fmt: skip

    judge = sub.add_parser("judge", help="judge answers (hosted Gemini or local Ollama)")
    common(judge)
    pilot_flags(judge)
    retry_flags(judge)
    judge_flags(judge)
    judge.add_argument("--confirm-model", help="the generation model (to resolve the protocol)")

    preflight = sub.add_parser("judge-preflight", help="check the local judge runtime (no pull)")
    common(preflight)
    judge_flags(preflight)
    preflight.add_argument("--confirm-model", help="unused; accepted for symmetry with judge")
    preflight.add_argument("--expect-quantization", help="e.g. Q4_K_M; exit 3 on a mismatch")
    preflight.add_argument("--require-gpu", action="store_true",
                           help="exit 3 unless the loaded model is fully GPU resident")  # fmt: skip
    preflight.add_argument("--warm-up", action="store_true",
                           help="load the model with one constant, "
                           "repository-free prompt")  # fmt: skip

    controls = sub.add_parser("judge-controls", help="run the synthetic judge controls")
    common(controls)
    retry_flags(controls)
    judge_flags(controls)
    controls.add_argument("--confirm-model", help="unused; accepted for symmetry with judge")
    controls.add_argument("--repeat", action="store_true",
                          help="independent second pass: no cache, model unloaded between "
                          "requests")  # fmt: skip

    validate = sub.add_parser(
        "judge-validate", help="evaluate artifacts against the predeclared plan"
    )
    common(validate)
    validate.add_argument("--controls-dir", help="directory of the first control pass")
    validate.add_argument("--controls-repeat-dir", help="directory of the repeat control pass")
    validate.add_argument("--pilot-dir", help="the local judge's pilot judgements directory")
    validate.add_argument("--pilot-repeat-dir", help="the repeat pilot judgements directory")
    validate.add_argument("--agreement", help="pilot audit-agree result (default: in --pilot-dir)")
    validate.add_argument("--primary-agreement", help="also apply the primary-audit criteria to "
                          "this audit-agree result")  # fmt: skip
    validate.add_argument("--report", help="where to write the report")

    compare = sub.add_parser("judge-compare", help="per-answer comparison of two judges (offline)")
    common(compare)
    pilot_flags(compare)
    compare.add_argument("--dir-a", help="judgements of judge A (default: the output directory)")
    compare.add_argument("--dir-b", required=True, help="judgements of judge B")

    report = sub.add_parser("report", help="analysis report (offline)")
    common(report)
    pilot_flags(report)
    report.add_argument("--judgements-dir", help="use (and write beside) another judge's files")

    worksheet = sub.add_parser("audit-worksheet", help="sample and write the blinded worksheet")
    common(worksheet)
    pilot_flags(worksheet)

    worksheet.add_argument(
        "--provisional", action="store_true",
        help="allow a development-only sample from an unfinished experiment",
    )  # fmt: skip

    agree = sub.add_parser("audit-agree", help="agreement between a filled worksheet and the judge")
    common(agree)
    pilot_flags(agree)
    agree.add_argument("--worksheet", required=True)
    agree.add_argument("--key", required=True)
    agree.add_argument("--rater", default="rater-1")
    agree.add_argument("--judgements-dir", help="the judge's directory (default: output dir)")

    rules = sub.add_parser(
        "audit-rule-check", help="verify the rule-scored system abstentions (not judge agreement)"
    )
    common(rules)
    rules.add_argument("--worksheet", required=True)
    rules.add_argument("--key", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "pilot", False) and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_PILOT_OUTPUT_DIR  # a pilot never defaults into the primary dir
    setup_logging()
    settings = get_settings()
    commands = {
        "protocol-init": _cmd_protocol_init,
        "pilot-init": _cmd_pilot_init,
        "plan": _cmd_plan,
        "run": _cmd_run,
        "judge": _cmd_judge,
        "judge-preflight": _cmd_judge_preflight,
        "judge-controls": _cmd_judge_controls,
        "judge-validate": _cmd_judge_validate,
        "judge-compare": _cmd_judge_compare,
        "report": _cmd_report,
        "audit-worksheet": _cmd_audit_worksheet,
        "audit-agree": _cmd_audit_agree,
        "audit-rule-check": _cmd_audit_rule_check,
    }
    try:
        return commands[args.command](args, settings)
    except KeyboardInterrupt:
        print("\ninterrupted: finished sides are checkpointed; re-run to resume", file=sys.stderr)
        return 130
    except (
        SecurityError, LLMError, ValueError, BenchmarkFormatError, RunMismatchError,
        JudgementMismatchError, EvaluationError, FileNotFoundError, PilotMismatchError,
        ControlsError, ValidationSpecError,
    ) as exc:  # fmt: skip
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
