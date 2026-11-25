"""Choosing the judge independently of the generation model (Milestone 11B).

The generation model is fixed by the protocol (Gemini). A judge is chosen on the command line:
``--judge-provider`` / ``--judge-model`` build a *separate* settings copy for the judge client, so
neither the generation model nor the process-wide settings change. Judgement artifacts of different
judges never share a directory: a local judge must be given its own ``--judgements-dir`` (never the
generation output directory), so an existing Gemini judgement file can neither be overwritten nor
accidentally resumed by another judge (resuming also re-checks the judge identity and refuses on
any difference).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from copilot.config import Settings

LOCAL_PROVIDER = "ollama"
#: Maximum judge output for the local judge. The protocol's 4096 is an upper bound; the judge's
#: JSON verdict is a few hundred tokens, and a smaller reserve leaves more of the context window
#: for the prompt. Recorded in the runtime fingerprint, not hidden.
LOCAL_JUDGE_MAX_OUTPUT_TOKENS = 1024
LOCAL_JUDGE_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class JudgeSelection:
    provider: str
    model: str
    settings: Settings  # a copy for the judge client; the caller's settings are untouched
    max_output_tokens: int
    judgements_dir: Path
    cache_dir: Path | None
    local: bool


def _within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def check_judgements_dir(
    judgements_dir: Path, *, pilot: bool, primary_dir: Path, pilot_dir: Path
) -> None:
    """Pilot judgements may not land in the primary namespace and vice versa."""
    target = judgements_dir.resolve()
    if pilot and _within(target, primary_dir.resolve()):
        raise ValueError("a pilot judgements directory must not be inside the primary directory")
    if not pilot and _within(target, pilot_dir.resolve()):
        raise ValueError("a primary judgements directory must not be inside the pilot directory")


def resolve_judge(
    args: argparse.Namespace,
    settings: Settings,
    *,
    protocol_max_output_tokens: int,
    root: Path,
    out_dir: Path,
    pilot: bool,
    primary_dir: Path,
    pilot_dir: Path,
    need_directory: bool = True,
) -> JudgeSelection:
    """Validate the judge options and build the judge's own settings and directories.

    Raises ``ValueError`` (reported as exit code 2) for any invalid or unsafe combination.
    """
    if not args.judge_model:
        raise ValueError(
            "pass --judge-model (a different model family from the generator is recommended)"
        )
    provider = args.judge_provider or settings.llm_provider
    local = provider == LOCAL_PROVIDER
    configured = Path(args.judgements_dir) if args.judgements_dir else None
    if need_directory and local and configured is None:
        raise ValueError(
            "a local judge needs its own --judgements-dir so it cannot touch another judge's "
            "artifacts (for example data/generation-eval/pilot-judge-ollama-llama3.1-8b)"
        )
    judgements_dir = (root / configured) if configured is not None else out_dir
    if need_directory:
        if local and judgements_dir.resolve() == out_dir.resolve():
            raise ValueError(
                "--judgements-dir must differ from the generation output directory for a local "
                "judge"
            )
        check_judgements_dir(
            judgements_dir, pilot=pilot, primary_dir=primary_dir, pilot_dir=pilot_dir
        )

    max_tokens = (
        args.judge_max_output_tokens
        if args.judge_max_output_tokens is not None
        else (LOCAL_JUDGE_MAX_OUTPUT_TOKENS if local else protocol_max_output_tokens)
    )
    if not 16 <= max_tokens <= protocol_max_output_tokens:
        raise ValueError(
            f"--judge-max-output-tokens must be between 16 and the protocol's judge maximum "
            f"({protocol_max_output_tokens})"
        )
    update: dict[str, object] = {"llm_provider": provider, "llm_model": args.judge_model}
    if local:
        timeout = (
            args.judge_timeout if args.judge_timeout is not None else LOCAL_JUDGE_TIMEOUT_SECONDS
        )
        if not 0 < timeout <= 3600:
            raise ValueError("--judge-timeout must be between 0 and 3600 seconds")
        update["llm_timeout_seconds"] = timeout
        if args.judge_num_ctx is not None:
            if not 1024 <= args.judge_num_ctx <= 131072:
                raise ValueError("--judge-num-ctx must be between 1024 and 131072")
            update["ollama_num_ctx"] = args.judge_num_ctx
        if args.judge_seed is not None:
            update["ollama_seed"] = args.judge_seed
        if args.judge_keep_alive is not None:
            update["ollama_keep_alive"] = args.judge_keep_alive
    judge_settings = settings.model_copy(update=update)
    cache_dir = None if args.no_cache else judgements_dir / "cache-judge"
    return JudgeSelection(
        provider=provider,
        model=args.judge_model,
        settings=judge_settings,
        max_output_tokens=max_tokens,
        judgements_dir=judgements_dir,
        cache_dir=cache_dir,
        local=local,
    )
