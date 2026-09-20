"""The hit rule and the metrics, with hand-computed expectations."""

from types import SimpleNamespace

import pytest

from copilot.evaluation import (
    QuestionOutcome,
    Region,
    first_hit_rank,
    region_overlaps,
    summarize,
)


def at(path: str, start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(file_path=path, start_line=start, end_line=end)


def region(path: str, start: int, end: int) -> Region:
    return Region(file_path=path, start_line=start, end_line=end)


# ---- overlap rule -----------------------------------------------------------------------------


def test_identical_ranges_overlap():
    assert region_overlaps(at("a.py", 10, 20), region("a.py", 10, 20))


def test_a_different_file_never_overlaps_even_with_identical_lines():
    assert not region_overlaps(at("b.py", 10, 20), region("a.py", 10, 20))


def test_path_comparison_is_exact_not_case_folded():
    assert not region_overlaps(at("A.py", 1, 5), region("a.py", 1, 5))


@pytest.mark.parametrize(
    ("chunk", "target", "expected"),
    [
        ((1, 10), (11, 20), False),  # adjacent: the chunk ends one line before the region
        ((11, 20), (1, 10), False),  # adjacent the other way round
        ((1, 10), (10, 20), True),  # exactly one shared line (the chunk's last)
        ((10, 20), (1, 10), True),  # exactly one shared line (the chunk's first)
        ((5, 5), (5, 5), True),  # a single shared line
        ((5, 5), (6, 6), False),
        ((1, 100), (40, 42), True),  # chunk contains the region
        ((40, 42), (1, 100), True),  # region contains the chunk
        ((1, 10), (50, 60), False),  # far apart
    ],
)
def test_line_overlap_cases(chunk, target, expected):
    assert region_overlaps(at("a.py", *chunk), region("a.py", *target)) is expected


def test_first_hit_rank_is_the_first_matching_result():
    results = [at("x.py", 1, 5), at("a.py", 30, 40), at("a.py", 8, 12), at("a.py", 9, 11)]
    assert first_hit_rank(results, [region("a.py", 10, 11)]) == 3


def test_any_of_several_regions_counts():
    results = [at("x.py", 1, 5), at("b.py", 100, 120)]
    regions = [region("a.py", 1, 10), region("b.py", 110, 111)]
    assert first_hit_rank(results, regions) == 2


def test_the_earliest_rank_wins_across_regions():
    results = [at("a.py", 1, 3), at("b.py", 1, 3)]
    assert first_hit_rank(results, [region("b.py", 1, 3), region("a.py", 1, 3)]) == 1


def test_no_match_gives_none():
    assert first_hit_rank([at("a.py", 1, 5)], [region("a.py", 6, 9)]) is None
    assert first_hit_rank([], [region("a.py", 1, 5)]) is None


# ---- metrics ----------------------------------------------------------------------------------


def outcome(rank, hit_lines=None, retrieved=()):
    return QuestionOutcome(
        question_id="q",
        first_hit_rank=rank,
        first_hit_lines=hit_lines,
        retrieved_locations=tuple(f"f.py:1-{n}" for n in retrieved),
        retrieved_lines=tuple(retrieved),
    )


HAND = [
    outcome(1, 10, (10, 20)),
    outcome(3, 30, (5, 5, 30)),
    outcome(None, None, (40, 60)),
    outcome(10, 50, (50,)),
]


def test_hit_at_k_by_hand():
    s = summarize(HAND, ks=(1, 3, 5, 10))
    assert s.n_questions == 4
    assert s.hit_counts == {1: 1, 3: 2, 5: 2, 10: 3}
    assert s.hit_at == {1: 0.25, 3: 0.5, 5: 0.5, 10: 0.75}


def test_mrr_by_hand():
    assert summarize(HAND).mrr == pytest.approx((1 + 1 / 3 + 0 + 1 / 10) / 4)


def test_mean_lines_per_hit_uses_the_first_matching_chunk_of_questions_that_hit():
    assert summarize(HAND).mean_lines_per_hit == pytest.approx((10 + 30 + 50) / 3)


def test_mean_retrieved_lines_covers_every_retrieved_chunk():
    assert summarize(HAND).mean_retrieved_lines == pytest.approx(
        (10 + 20 + 5 + 5 + 30 + 40 + 60 + 50) / 8
    )


def test_all_misses_give_zero_scores_and_no_line_average():
    s = summarize([outcome(None, None, (7,)), outcome(None, None, (9,))])
    assert s.mrr == 0.0 and s.hit_at[10] == 0.0 and s.mean_lines_per_hit is None
    assert s.mean_retrieved_lines == pytest.approx(8.0)


def test_a_single_perfect_question():
    s = summarize([outcome(1, 12, (12,))])
    assert s.mrr == 1.0 and s.hit_at == {1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0}


def test_a_hit_deeper_than_the_depth_is_a_miss_for_mrr():
    s = summarize([outcome(7, 5, (5,))], ks=(1, 3), depth=5)
    assert s.mrr == 0.0


def test_custom_ks_are_honoured():
    s = summarize(HAND, ks=(2, 4))
    assert set(s.hit_at) == {2, 4} and s.hit_at[4] == 0.5


def test_no_outcomes_and_invalid_arguments_are_refused():
    with pytest.raises(ValueError, match="zero"):
        summarize([])
    with pytest.raises(ValueError, match="k"):
        summarize(HAND, ks=(0,))
    with pytest.raises(ValueError, match="1-based"):
        summarize([outcome(0)])


def test_the_confound_is_visible_in_the_line_columns():
    """Same hit ranks, but one configuration's hit chunks are ten times larger."""
    small = summarize([outcome(1, 20, (20,)), outcome(2, 30, (25, 30))])
    large = summarize([outcome(1, 200, (200,)), outcome(2, 300, (250, 300))])
    assert small.hit_at == large.hit_at and small.mrr == large.mrr
    assert large.mean_lines_per_hit == pytest.approx(10 * small.mean_lines_per_hit)
