import pytest

from copilot.embeddings.token_validation import assess_caps, compare_token_counts
from copilot.ingestion import IngestionPolicy, ingest_repository
from copilot.utils.tokens import estimate_tokens
from tests.chunking_helpers import make_file
from tests.unit.test_chunk_model import make_chunk


def chunk_with(estimate, path="a.py", start=1, end=1):
    return make_chunk(
        chunk_id=f"{abs(hash((path, start, estimate))) % 16**16:016x}",
        file_path=path,
        start_line=start,
        end_line=end,
        token_estimate=estimate,
    )


def test_comparison_statistics_hand_computed():
    chunks = [
        chunk_with(10, "a.py", 1, 5),
        chunk_with(20, "b.py", 1, 5),
        chunk_with(30, "c.py", 1, 5),
        chunk_with(40, "d.py", 1, 5),
    ]
    actual = [12, 20, 25, 60]
    # signed error (estimated - actual) = [-2, 0, +5, -20]; absolute sorted = [0, 2, 5, 20]
    c = compare_token_counts(chunks, actual, top=2)
    assert c.sample_count == 4
    assert (c.total_estimated, c.total_actual) == (100, 117)
    assert c.mean_abs_error == pytest.approx(6.75)
    assert c.median_abs_error == pytest.approx(3.5)
    assert c.p95_abs_error == 20.0  # nearest rank: ceil(0.95 * 4) = 4th value
    assert c.max_abs_error == 20
    assert c.mean_signed_error == pytest.approx(-4.25)
    assert (c.pct_underestimated, c.pct_overestimated, c.pct_exact) == (50.0, 25.0, 25.0)
    # actual / estimated = [1.2, 1.0, 0.8333, 1.5]
    assert c.actual_over_estimated_median == pytest.approx(1.1)
    assert c.actual_over_estimated_max == pytest.approx(1.5)
    assert [(e.file_path, e.estimated, e.actual) for e in c.largest_underestimates] == [
        ("d.py", 40, 60),
        ("a.py", 10, 12),
    ]
    assert [(e.file_path, e.estimated, e.actual) for e in c.largest_overestimates] == [
        ("c.py", 30, 25)
    ]


def test_examples_carry_only_metadata():
    c = compare_token_counts([chunk_with(5, "x.py", 3, 9)], [9], top=1)
    example = c.largest_underestimates[0]
    assert set(example.model_dump()) == {
        "file_path",
        "start_line",
        "end_line",
        "estimated",
        "actual",
    }
    assert (example.start_line, example.end_line) == (3, 9)


def test_perfect_estimator_has_zero_error():
    chunks = [chunk_with(n, f"{n}.py") for n in (3, 7, 11)]
    c = compare_token_counts(chunks, [3, 7, 11])
    assert c.max_abs_error == 0 and c.pct_exact == 100.0
    assert c.largest_underestimates == () and c.largest_overestimates == ()


def test_length_mismatch_and_empty_input_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        compare_token_counts([chunk_with(1)], [1, 2])
    with pytest.raises(ValueError, match="no chunks"):
        compare_token_counts([], [])


# ---- cap assessment ------------------------------------------------------------------------
@pytest.fixture
def ten_line_repo(tmp_path):
    (tmp_path / "m.py").write_text("a b c d e f g h i j\n" * 10)  # 10 lines x 10 tokens
    return ingest_repository(tmp_path, IngestionPolicy())


def identity_counts(texts):
    return [estimate_tokens(t) for t in texts]


def test_cap_assessment_hand_computed(ten_line_repo):
    small, big = assess_caps(
        ten_line_repo,
        identity_counts,
        caps=[25, 1000],
        size_lines=10,
        overlap_lines=0,
        model_limit=30,
        style="raw",
    )
    # cap 25 -> 2 lines (20 tokens) per chunk -> 5 chunks; the 4 non-final ones are cap-shortened
    assert (small.cap, small.chunk_count, small.cap_shortened_pct) == (25, 5, 100.0)
    assert (small.actual_max, small.over_model_limit, small.at_least_90pct_of_limit) == (20, 0, 0)
    assert (small.estimated_median, small.estimated_max) == (20.0, 20)
    # cap 1000 -> one 10-line chunk of 100 tokens, over the pretend model limit of 30
    assert (big.chunk_count, big.cap_shortened_pct, big.actual_max) == (1, 0.0, 100)
    assert (big.over_model_limit, big.over_512_actual, big.at_least_90pct_of_limit) == (1, 0, 1)


def test_prefixed_style_adds_the_prefix_to_the_embedded_size(ten_line_repo):
    raw, prefixed = (
        assess_caps(
            ten_line_repo,
            identity_counts,
            caps=[1000],
            size_lines=10,
            overlap_lines=0,
            model_limit=500,
            style=style,
        )[0]
        for style in ("raw", "prefixed")
    )
    assert prefixed.embedded_max > raw.embedded_max
    assert prefixed.actual_max == raw.actual_max  # raw content size is unaffected by the prefix


def test_cap_assessment_on_empty_repository_is_rejected(tmp_path):
    (tmp_path / "empty.py").write_text("")
    empty = ingest_repository(tmp_path, IngestionPolicy())
    with pytest.raises(ValueError, match="no chunks"):
        assess_caps(
            empty, identity_counts, caps=[100], size_lines=10, overlap_lines=0, model_limit=5
        )


def test_make_file_helper_is_importable():  # keeps the shared helper import honest
    assert make_file("x = 1\n").language == "python"


def test_tokenizer_free_bytes_bound_hand_computed(ten_line_repo):
    """10 lines of 'a b c d e f g h i j' (19 bytes each): raw chunk = 10*19 + 9 newlines = 199.

    The prefix "File: m.py\nLanguage: python\nLines: 1-10\n\n" is 11 + 17 + 12 + 1 = 41 bytes.
    """
    raw, prefixed = (
        assess_caps(
            ten_line_repo,
            None,  # no tokenizer at all
            caps=[1000],
            size_lines=10,
            overlap_lines=0,
            model_limit=240,
            style=style,
        )[0]
        for style in ("raw", "prefixed")
    )
    assert raw.embedded_bytes_max == 199 and prefixed.embedded_bytes_max == 240
    assert raw.bytes_bound_exceeds_limit == 0  # 199 + 2 <= 240: provably within the limit
    assert prefixed.bytes_bound_exceeds_limit == 1  # 240 + 2 > 240: cannot be proven safe
    assert raw.actual_max is None and raw.over_model_limit is None  # token fields need a tokenizer
