"""Measuring the provisional token estimator against a real tokenizer, and testing chunk-size caps.

Pure functions: they take counts produced by an embedder's ``count_tokens`` so they can be tested
with fake counts. Nothing here reads chunk contents for output; examples carry only file path and
line range.

Terminology: *estimated* = ``copilot.utils.tokens.estimate_tokens`` (the heuristic behind
``chunk_max_tokens``); *actual* = the embedding model's tokenizer, including special tokens.
``estimated - actual`` is the signed error: **negative means underestimated**, which is the unsafe
direction because a chunk could then exceed the cap (and possibly the model limit).
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from copilot.chunking import LineChunker, chunk_repository
from copilot.chunking.stats import percentile
from copilot.embeddings.representation import TextStyle, embedding_text
from copilot.models.chunk import Chunk
from copilot.models.ingestion import IngestionResult


class TokenExample(BaseModel):
    """One chunk in an error ranking (metadata only, never content)."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    start_line: int
    end_line: int
    estimated: int
    actual: int


class TokenComparison(BaseModel):
    """Estimated-vs-actual token statistics over a set of chunks."""

    model_config = ConfigDict(frozen=True)

    sample_count: int
    total_estimated: int
    total_actual: int
    mean_abs_error: float
    median_abs_error: float
    p95_abs_error: float
    max_abs_error: int
    mean_signed_error: float  # estimated - actual; negative = underestimated on average
    pct_underestimated: float  # estimated < actual (unsafe direction)
    pct_overestimated: float  # estimated > actual
    pct_exact: float
    actual_over_estimated_median: float
    actual_over_estimated_p95: float
    actual_over_estimated_max: float
    largest_underestimates: tuple[TokenExample, ...]
    largest_overestimates: tuple[TokenExample, ...]


class CapAssessment(BaseModel):
    """What one candidate ``chunk_max_tokens`` value does to this repository's chunks.

    Fields marked *tokenizer-free* need no model. ``None`` in an ``actual_*``/``over_*`` field
    means no tokenizer was supplied.

    **Tokenizer-independent safety bound.** A tokenizer that emits at least one token per
    character (WordPiece, BPE) or per byte (byte-level BPE) can never produce more than
    ``len(utf-8 bytes) + 2`` tokens (the ``+2`` covers start/end special tokens). So a chunk whose
    embedding text has ``bytes + 2 <= model_limit`` is *provably* within the model limit, without
    knowing the tokenizer. ``bytes_bound_exceeds_limit`` counts chunks for which that proof fails
    (they may still be fine; only a real tokenizer can say).
    """

    model_config = ConfigDict(frozen=True)

    cap: int
    chunk_count: int  # tokenizer-free
    fragment_chunks: int  # tokenizer-free
    cap_shortened_pct: (
        float  # tokenizer-free: % of non-final whole-line chunks cut short by the cap
    )
    estimated_median: float  # tokenizer-free: estimated tokens of the raw chunk content
    estimated_p95: float
    estimated_max: int
    embedded_bytes_median: float  # tokenizer-free: UTF-8 bytes of the embedding text
    embedded_bytes_p95: float
    embedded_bytes_max: int
    bytes_bound_exceeds_limit: int  # tokenizer-free: chunks where bytes + 2 > model_limit
    actual_median: float | None = None  # actual tokens of the raw chunk content
    actual_p95: float | None = None
    actual_max: int | None = None
    embedded_max: int | None = None  # actual tokens of the embedding text (per style)
    over_model_limit: int | None = None  # embedding text exceeds the model input limit
    over_512_actual: int | None = None
    at_least_90pct_of_limit: int | None = None


def _example(chunk: Chunk, estimated: int, actual: int) -> TokenExample:
    return TokenExample(
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        estimated=estimated,
        actual=actual,
    )


def compare_token_counts(
    chunks: Sequence[Chunk], actual_counts: Sequence[int], *, top: int = 5
) -> TokenComparison:
    """Compare each chunk's ``token_estimate`` with its actual token count."""
    if len(chunks) != len(actual_counts):
        raise ValueError("chunks and actual_counts must have the same length")
    if not chunks:
        raise ValueError("no chunks to compare")

    estimated = [c.token_estimate for c in chunks]
    signed = [e - a for e, a in zip(estimated, actual_counts, strict=True)]
    abs_err = sorted(abs(s) for s in signed)
    n = len(chunks)
    ratios = sorted(a / e for e, a in zip(estimated, actual_counts, strict=True) if e > 0)

    ranked = sorted(range(n), key=lambda i: signed[i])  # most negative first
    under = [i for i in ranked if signed[i] < 0][:top]
    over = [i for i in reversed(ranked) if signed[i] > 0][:top]

    def ratio_pct(q: float) -> float:
        if not ratios:
            return 0.0
        return ratios[max(0, min(len(ratios) - 1, round(q / 100 * len(ratios)) - 1))]

    return TokenComparison(
        sample_count=n,
        total_estimated=sum(estimated),
        total_actual=sum(actual_counts),
        mean_abs_error=statistics.fmean(abs_err),
        median_abs_error=float(statistics.median(abs_err)),
        p95_abs_error=percentile(abs_err, 95),
        max_abs_error=abs_err[-1],
        mean_signed_error=statistics.fmean(signed),
        pct_underestimated=100 * sum(s < 0 for s in signed) / n,
        pct_overestimated=100 * sum(s > 0 for s in signed) / n,
        pct_exact=100 * sum(s == 0 for s in signed) / n,
        actual_over_estimated_median=float(statistics.median(ratios)) if ratios else 0.0,
        actual_over_estimated_p95=ratio_pct(95),
        actual_over_estimated_max=ratios[-1] if ratios else 0.0,
        largest_underestimates=tuple(
            _example(chunks[i], estimated[i], actual_counts[i]) for i in under
        ),
        largest_overestimates=tuple(
            _example(chunks[i], estimated[i], actual_counts[i]) for i in over
        ),
    )


def assess_caps(
    ingestion: IngestionResult,
    count_tokens: Callable[[Sequence[str]], list[int]] | None,
    *,
    caps: Sequence[int],
    size_lines: int,
    overlap_lines: int,
    model_limit: int,
    style: TextStyle = "prefixed",
) -> list[CapAssessment]:
    """Re-chunk ``ingestion`` at each candidate cap and measure the resulting chunk sizes.

    Uses the same line size and overlap for every cap, so the only variable is the cap. Pass
    ``count_tokens=None`` for the tokenizer-free measurements only.
    """
    results: list[CapAssessment] = []
    for cap in caps:
        chunker = LineChunker(size_lines=size_lines, overlap_lines=overlap_lines, max_tokens=cap)
        chunks = list(chunk_repository(ingestion, chunker).chunks)
        if not chunks:
            raise ValueError("no chunks produced; nothing to assess")
        texts = [embedding_text(c, style) for c in chunks]
        text_bytes = sorted(len(t.encode("utf-8")) for t in texts)
        estimated = sorted(c.token_estimate for c in chunks)

        last_index: dict[str, int] = {}
        for c in chunks:
            last_index[c.file_path] = max(last_index.get(c.file_path, -1), c.chunk_index)
        non_final = [
            c for c in chunks if not c.is_fragment and c.chunk_index != last_index[c.file_path]
        ]
        shortened = [c for c in non_final if c.end_line - c.start_line + 1 < size_lines]

        actual_fields: dict[str, float | int | None] = {}
        if count_tokens is not None:
            raw_actual = sorted(count_tokens([c.content for c in chunks]))
            embedded_actual = count_tokens(texts)
            actual_fields = {
                "actual_median": float(statistics.median(raw_actual)),
                "actual_p95": percentile(raw_actual, 95),
                "actual_max": raw_actual[-1],
                "embedded_max": max(embedded_actual),
                "over_model_limit": sum(a > model_limit for a in embedded_actual),
                "over_512_actual": sum(a > 512 for a in embedded_actual),
                "at_least_90pct_of_limit": sum(a >= 0.9 * model_limit for a in embedded_actual),
            }

        results.append(
            CapAssessment(
                cap=cap,
                chunk_count=len(chunks),
                fragment_chunks=sum(c.is_fragment for c in chunks),
                cap_shortened_pct=100 * len(shortened) / len(non_final) if non_final else 0.0,
                estimated_median=float(statistics.median(estimated)),
                estimated_p95=percentile(estimated, 95),
                estimated_max=estimated[-1],
                embedded_bytes_median=float(statistics.median(text_bytes)),
                embedded_bytes_p95=percentile(text_bytes, 95),
                embedded_bytes_max=text_bytes[-1],
                bytes_bound_exceeds_limit=sum(b + 2 > model_limit for b in text_bytes),
                **actual_fields,  # type: ignore[arg-type]
            )
        )
    return results
