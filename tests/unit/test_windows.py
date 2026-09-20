import random

import pytest

from copilot.chunking.windows import split_into_windows, split_lines, split_long_line
from copilot.utils.tokens import estimate_tokens
from tests.chunking_helpers import assert_chunk_invariants, make_file

BIG = 10_000  # effectively no token cap


def spans(windows):
    return [(w.start_line, w.end_line) for w in windows]


def numbered(n):
    return [f"line{i}" for i in range(1, n + 1)]


class TestSplitLines:
    def test_empty_text_has_no_lines(self):
        assert split_lines("") == []

    def test_trailing_newline_terminates_last_line(self):
        assert split_lines("a\nb\n") == ["a", "b"]
        assert split_lines("a\nb") == ["a", "b"]

    def test_blank_lines_are_kept(self):
        assert split_lines("a\n\n\nb\n") == ["a", "", "", "b"]
        assert split_lines("\n") == [""]

    def test_only_newline_splits_lines(self):
        # str.splitlines() would split on these and desynchronise line numbers
        assert split_lines("a\x0cb c\nd\n") == ["a\x0cb c", "d"]


class TestWindows:
    def test_overlap_and_boundaries_hand_computed(self):
        windows = split_into_windows(numbered(30), size=10, overlap=3, max_tokens=BIG)
        assert spans(windows) == [(1, 10), (8, 17), (15, 24), (22, 30)]

    def test_consecutive_windows_overlap_by_exactly_overlap_lines(self):
        windows = split_into_windows(numbered(100), size=12, overlap=4, max_tokens=BIG)
        for a, b in zip(windows, windows[1:], strict=False):
            assert a.end_line - b.start_line + 1 == 4

    def test_zero_overlap_partitions_the_file(self):
        windows = split_into_windows(numbered(25), size=10, overlap=0, max_tokens=BIG)
        assert spans(windows) == [(1, 10), (11, 20), (21, 25)]

    def test_file_shorter_than_window_is_one_chunk(self):
        assert spans(split_into_windows(numbered(3), size=10, overlap=2, max_tokens=BIG)) == [
            (1, 3)
        ]

    def test_file_exactly_one_window_has_no_redundant_tail(self):
        assert spans(split_into_windows(numbered(10), size=10, overlap=3, max_tokens=BIG)) == [
            (1, 10)
        ]

    def test_one_line_file(self):
        assert spans(split_into_windows(["x = 1"], size=10, overlap=3, max_tokens=BIG)) == [(1, 1)]

    def test_no_lines_no_windows(self):
        assert split_into_windows([], size=10, overlap=3, max_tokens=BIG) == []

    def test_first_line_number_offsets_results(self):
        windows = split_into_windows(
            numbered(5), size=3, overlap=1, max_tokens=BIG, first_line_number=41
        )
        assert spans(windows) == [(41, 43), (43, 45)]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"size": 0, "overlap": 0, "max_tokens": 5},
            {"size": 5, "overlap": 5, "max_tokens": 5},
            {"size": 5, "overlap": -1, "max_tokens": 5},
            {"size": 5, "overlap": 1, "max_tokens": 0},
        ],
    )
    def test_invalid_parameters_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            split_into_windows(["a"], **kwargs)

    def test_whitespace_only_windows_are_dropped_but_code_stays_covered(self):
        lines = ["a = 1"] + [""] * 20 + ["b = 2"]
        windows = split_into_windows(lines, size=5, overlap=1, max_tokens=BIG)
        assert all(w.content.strip() for w in windows)
        covered = {n for w in windows for n in range(w.start_line, w.end_line + 1)}
        assert 1 in covered and 22 in covered

    def test_whitespace_only_file_gives_no_windows(self):
        assert split_into_windows(["", "  ", "\t"], size=5, overlap=1, max_tokens=BIG) == []


class TestTokenCap:
    def test_window_is_shortened_to_respect_cap(self):
        lines = ["a b c d e f g h i j"] * 10  # 10 tokens per line
        windows = split_into_windows(lines, size=10, overlap=0, max_tokens=25)
        assert spans(windows) == [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
        assert all(estimate_tokens(w.content) <= 25 for w in windows)

    def test_overlap_is_reduced_when_cap_forces_small_windows(self):
        lines = ["a b c d e f g h i j"] * 5
        windows = split_into_windows(lines, size=5, overlap=3, max_tokens=25)
        assert spans(windows) == [(1, 2), (2, 3), (3, 4), (4, 5)]  # slides by one line

    def test_giant_line_between_normal_lines_causes_no_duplicate_cascade(self):
        lines = numbered(30) + ["x " * 200] + numbered(5)  # line 31 is far over the cap
        windows = split_into_windows(lines, size=10, overlap=4, max_tokens=30)
        whole = [(w.start_line, w.end_line) for w in windows if w.fragment_index is None]
        for i, (a1, b1) in enumerate(whole):
            for j, (a2, b2) in enumerate(whole):
                assert i == j or not (a2 <= a1 and b1 <= b2)
        assert all(
            w.start_line == w.end_line == 31 for w in windows if w.fragment_index is not None
        )
        assert any(w.fragment_index is not None for w in windows)

    def test_long_line_is_split_into_fragments_that_reassemble(self):
        line = "    data = [" + ", ".join(str(i) for i in range(400)) + "]"
        windows = split_into_windows(
            ["before = 1", line, "after = 2"], size=10, overlap=2, max_tokens=50
        )
        fragments = [w for w in windows if w.fragment_index is not None]
        assert len(fragments) > 1
        assert "".join(w.content for w in fragments) == line
        assert all(estimate_tokens(w.content) <= 50 for w in windows)
        assert all(w.start_line == w.end_line == 2 for w in fragments)
        assert [w.fragment_index for w in fragments] == list(range(len(fragments)))
        assert {w.fragment_count for w in fragments} == {len(fragments)}

    def test_split_long_line_pieces_respect_cap_and_reassemble(self):
        line = "  " + "abc def(x, y) " * 100
        pieces = split_long_line(line, 7)
        assert "".join(pieces) == line
        assert all(0 < estimate_tokens(p) <= 7 for p in pieces)

    def test_unicode_long_line(self):
        line = "变量" * 200  # 400 tokens under the per-character estimate
        pieces = split_long_line(line, 100)
        assert len(pieces) == 4
        assert "".join(pieces) == line


@pytest.mark.parametrize("seed", range(25))
def test_invariants_hold_on_random_files(seed):
    """Randomised (seeded, deterministic) files including blank and over-long lines."""
    rng = random.Random(seed)
    size = rng.randint(2, 15)
    overlap = rng.randint(0, size - 1)
    max_tokens = rng.randint(8, 60)
    lines = []
    for _ in range(rng.randint(0, 120)):
        kind = rng.random()
        if kind < 0.15:
            lines.append("")
        elif kind < 0.2:
            lines.append("tok " * rng.randint(60, 200))  # over the cap
        else:
            lines.append(
                "    " + " ".join(f"w{rng.randint(0, 99)}" for _ in range(rng.randint(1, 12)))
            )
    content = "\n".join(lines) + ("\n" if lines else "")

    from copilot.chunking import LineChunker

    file = make_file(content)
    chunks = LineChunker(size_lines=size, overlap_lines=overlap, max_tokens=max_tokens).chunk_file(
        file
    )
    assert_chunk_invariants(file, chunks, max_tokens=max_tokens)
