from copilot.utils.tokens import estimate_tokens, token_starts


def test_empty_and_whitespace_are_zero_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens(" \n\t  \n") == 0


def test_words_and_punctuation():
    assert estimate_tokens("def add(a, b):") == 8  # def add ( a , b ) :
    assert estimate_tokens("snake_case_name camelCase123") == 2


def test_non_ascii_characters_count_individually():
    assert estimate_tokens("你好世界") == 4  # not one "word": would badly under-count CJK
    assert estimate_tokens("größe") == 4  # gr + ö + ß + e (conservative)


def test_additive_over_lines():
    a, b = "x = foo(1)", "    return [a, b]"
    assert estimate_tokens(a + "\n" + b) == estimate_tokens(a) + estimate_tokens(b)


def test_token_starts_match_count_and_are_increasing():
    text = "a = b + 12\n"
    starts = list(token_starts(text))
    assert len(starts) == estimate_tokens(text) == 5
    assert starts == sorted(starts)
    assert starts[0] == 0
