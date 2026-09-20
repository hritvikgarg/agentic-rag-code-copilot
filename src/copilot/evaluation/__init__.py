"""Retrieval evaluation (Milestone 5b): a transparent benchmark, Hit@k / MRR, and a config matrix.

No LLM is involved. Ground truth is *source regions* (file path + line range), matched by
line overlap, never by chunk id, so it stays valid when chunking parameters change the chunk ids.
"""

from copilot.evaluation.benchmark import (
    BENCHMARK_SCHEMA,
    Benchmark,
    BenchmarkFormatError,
    BenchmarkMeta,
    BenchmarkQuestion,
    Region,
    RegionProblem,
    dump_benchmark,
    load_benchmark,
    region_text,
    seal_regions,
    verify_regions,
)
from copilot.evaluation.matching import first_hit_rank, region_overlaps
from copilot.evaluation.metrics import DEFAULT_KS, QuestionOutcome, Summary, summarize
from copilot.evaluation.runner import (
    ConfigResult,
    MatrixResult,
    evaluate_retriever,
    run_configuration,
    run_matrix,
)

__all__ = [
    "BENCHMARK_SCHEMA",
    "DEFAULT_KS",
    "Benchmark",
    "BenchmarkFormatError",
    "BenchmarkMeta",
    "BenchmarkQuestion",
    "ConfigResult",
    "MatrixResult",
    "QuestionOutcome",
    "Region",
    "RegionProblem",
    "Summary",
    "dump_benchmark",
    "evaluate_retriever",
    "first_hit_rank",
    "load_benchmark",
    "region_overlaps",
    "region_text",
    "run_configuration",
    "run_matrix",
    "seal_regions",
    "summarize",
    "verify_regions",
]
