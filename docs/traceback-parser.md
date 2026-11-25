# Traceback parser (Milestone 12, slice S1)

`copilot.parsing` turns pasted Python traceback text into typed data. It is the first stage of the
evidence-based debugging feature planned in [`m12-debugging-plan.md`](m12-debugging-plan.md). It does
**only** parsing: mapping frames to repository files is slice S2, evidence collection is S3.

```python
from copilot.parsing import DebugLimits, looks_like_traceback, parse_traceback

result = parse_traceback(text, limits=DebugLimits())  # never raises on content
result.status  # complete | partial | failed
result.segments  # in input order; chained exceptions: cause first, final exception last
result.final_segment  # the exception that actually propagated
result.diagnostics  # fixed codes, severities and positions - never input text
looks_like_traceback(text)  # cheap routing hint, not a validator
```

## Trust model

The text is untrusted **data**. The parser splits and slices strings and uses a few anchored
regular expressions without nested quantifiers (linear time). It never evaluates, compiles, imports
or executes anything, opens no file and looks at no path. Tests enforce this (static checks on the
parser's imports and calls, a test that forbids filesystem access while parsing real paths, a
deterministic fuzz suite).

* Text copied from the traceback (paths, function names, source lines, exception type/message) is
  excluded from `repr`, `str` and diagnostics. Diagnostics carry a fixed code and position only.
  The only log record is a DEBUG line with counts, the status and a 12-character SHA-256 digest.
* Control, format and line-separator characters (including bidi overrides) are replaced by U+FFFD
  and reported. ANSI colour sequences are removed and reported.
* Nothing is truncated silently: a cut field has a `*_truncated` flag **and** a diagnostic. A path
  that is truncated or has an unterminated quote has `path_truncated=True` and must never be used
  as a mapping candidate.

## Supported

Standard tracebacks with any number of frames; chained exceptions (`cause`/`context`); Windows,
POSIX, UNC and relative paths (lexical classification only); `SyntaxError`, `IndentationError` and
`TabError` blocks with or without the `Traceback` header (with the caret position inside the
displayed source text); frames without source lines; Python 3.11+ caret lines; recursion-elided
output; pseudo-files (`<string>`, `<stdin>`, `<frozen ...>`); exception notes and "Did you mean"
hints (continuation lines until a blank line); CRLF/CR line endings; malformed or incomplete input
(`partial` plus diagnostics).

This is **not** a claim of complete support for every Python version or traceback format. The golden
fixtures are real CPython 3.13 output plus documented path-style transformations of it.

## Not supported (reported, never guessed)

ExceptionGroup output, pytest-style, IPython-style, JavaScript and Java traces, and log-prefixed
lines. A bare word without `:` is accepted as an exception line only if it ends like an exception
name (`...Error`, `...Exception`, `...Warning`, `...Exit`, `...Interrupt`, `...Iteration`).

## Limits (`DebugLimits`, all enforced)

| Limit | Default | Behaviour beyond the limit |
|---|---|---|
| input characters | 50,000 | rejected (`failed`, `input_too_large`), nothing parsed |
| input lines | 3,000 | rejected (`failed`, `too_many_lines`) |
| line length | 2,000 | line cut; flagged; a cut frame line never yields a trusted path/number |
| frames per segment | 200 | first 10 + last 190 kept, original `index` values keep the gap |
| segments | 20 | last 20 kept, `segments_omitted` counts the rest |
| path / function / source line | 1,024 / 200 / 300 | cut and flagged |
| exception type name | 200 | the line is malformed |
| message | 500 characters, 20 lines | cut and flagged |
| diagnostics | 50 | a final `diagnostics_truncated` marker; status still reflects the worst severity |

## Examples (real CPython 3.13 output)

```text
Traceback (most recent call last):
  File "/tmp/tbgen/nm.py", line 5, in <module>
    Svc().go(None)
    ~~~~~~~~^^^^^^
  File "/tmp/tbgen/nm.py", line 3, in go
    return x.value + 1
           ^^^^^^^
AttributeError: 'NoneType' object has no attribute 'value'
```

parses to one segment, two frames (`raw_path="/tmp/tbgen/nm.py"`, `path_style=posix_absolute`,
`line=5`/`3`, `function="<module>"`/`"go"`, claimed `source_line`), exception `AttributeError`, status
`complete`, no diagnostics. A header-less syntax error:

```text
  File "/tmp/tbgen/syn.py", line 1
    def broken(:
               ^
SyntaxError: invalid syntax
```

gives one segment (`has_header=False`), one frame (`function=None`), `is_syntax_error=True` and
`syntax.caret_offset=11`. The traceback is a *claim by the user*: line numbers and source lines are
unverified until slice S2/S3 compare them with the indexed repository.
