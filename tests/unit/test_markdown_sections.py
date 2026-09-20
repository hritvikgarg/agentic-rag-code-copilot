from copilot.chunking.markdown_sections import split_markdown_sections
from copilot.chunking.windows import split_lines


def sections(text):
    lines = split_lines(text)
    return [(s.start + 1, s.end, s.heading) for s in split_markdown_sections(lines)]


def test_no_headings_is_one_section():
    assert sections("just text\nmore\n") == [(1, 2, None)]


def test_empty_text_has_no_sections():
    assert sections("") == []


def test_sections_split_at_headings_with_heading_paths():
    text = "# A\ntext a\n## B\ntext b\n## C\ntext c\n# D\ntext d\n"
    assert sections(text) == [
        (1, 2, "A"),
        (3, 4, "A > B"),
        (5, 6, "A > C"),
        (7, 8, "D"),
    ]


def test_preamble_before_first_heading_has_no_heading():
    assert sections("intro\n\n# A\nbody\n") == [(1, 2, None), (3, 4, "A")]


def test_blank_preamble_is_dropped():
    assert sections("\n\n# A\nbody\n") == [(3, 4, "A")]


def test_heading_only_section_is_merged_into_the_next():
    assert sections("# A\n\n## B\ntext b\n") == [(1, 4, "A > B")]


def test_trailing_heading_only_section_is_kept():
    assert sections("# A\ntext\n## B\n") == [(1, 2, "A"), (3, 3, "A > B")]


def test_headings_inside_fenced_code_are_ignored():
    text = "# A\n```python\n# not a heading\n```\n## B\nx\n"
    assert sections(text) == [(1, 4, "A"), (5, 6, "A > B")]


def test_tilde_and_longer_fences_and_unclosed_fence():
    assert sections("# A\n~~~\n# no\n~~~\n# B\nx\n") == [(1, 4, "A"), (5, 6, "B")]
    assert sections("# A\n````\n```\n# no\n````\n# B\nx\n") == [(1, 5, "A"), (6, 7, "B")]
    assert sections("# A\n```\n# never closed\n") == [(1, 3, "A")]


def test_closing_hashes_and_empty_titles():
    assert sections("## Title ##\nbody\n") == [(1, 2, "Title")]
    assert sections("# A\nbody\n##\nmore\n") == [(1, 2, "A"), (3, 4, "A")]


def test_hash_without_space_is_not_a_heading():
    assert sections("#hashtag\ntext\n") == [(1, 2, None)]


def test_seven_hashes_is_not_a_heading():
    assert sections("####### seven\ntext\n") == [(1, 2, None)]


def test_heading_level_reset_pops_deeper_levels():
    text = "# A\nx\n### deep\ny\n## B\nz\n"
    assert sections(text) == [(1, 2, "A"), (3, 4, "A > deep"), (5, 6, "A > B")]


def test_setext_headings_are_not_recognised():  # documented limitation
    assert sections("Title\n=====\ntext\n") == [(1, 3, None)]
