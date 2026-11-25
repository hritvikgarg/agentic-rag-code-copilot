"""Golden tracebacks for the Milestone 12 parser tests.

The ``REAL_*`` constants are verbatim output of CPython 3.13 produced by running tiny throw-away
scripts (never repository code) in a scratch directory; only the directory name was fixed to
``/tmp/tbgen``. The variants below them are produced by *documented text transformations* of those
real outputs (path style changes), so no traceback format is invented. Hand-written constants are
marked ``HAND_*`` and mimic output that cannot be produced without third-party packages.
"""

from __future__ import annotations

REAL_ATTRIBUTE_NONE = """\
Traceback (most recent call last):
  File "/tmp/tbgen/nm.py", line 5, in <module>
    Svc().go(None)
    ~~~~~~~~^^^^^^
  File "/tmp/tbgen/nm.py", line 3, in go
    return x.value + 1
           ^^^^^^^
AttributeError: 'NoneType' object has no attribute 'value'
"""

REAL_CHAINED = """\
Traceback (most recent call last):
  File "/tmp/tbgen/chain.py", line 6, in connect
    return load(cfg)
  File "/tmp/tbgen/chain.py", line 2, in load
    return cfg["db"]["url"]
           ~~~^^^^^^
KeyError: 'db'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/tmp/tbgen/chain.py", line 12, in run
    connect({})
    ~~~~~~~^^^^
  File "/tmp/tbgen/chain.py", line 8, in connect
    raise RuntimeError("cannot connect") from exc
RuntimeError: cannot connect

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/tmp/tbgen/chain.py", line 16, in <module>
    run()
    ~~~^^
  File "/tmp/tbgen/chain.py", line 14, in run
    raise ValueError("during handling")
ValueError: during handling
"""

REAL_RECURSION = """\
Traceback (most recent call last):
  File "/tmp/tbgen/rec.py", line 3, in <module>
    f(0)
    ~^^^
  File "/tmp/tbgen/rec.py", line 2, in f
    return f(n + 1)
  File "/tmp/tbgen/rec.py", line 2, in f
    return f(n + 1)
  [Previous line repeated 996 more times]
RecursionError: maximum recursion depth exceeded
"""

REAL_SYNTAX_HEADERLESS = """\
  File "/tmp/tbgen/syn.py", line 1
    def broken(:
               ^
SyntaxError: invalid syntax
"""

REAL_INDENTATION_HEADERLESS = """\
  File "/tmp/tbgen/ind.py", line 2
    x = 1
    ^
IndentationError: expected an indented block after function definition on line 1
"""

REAL_SYNTAX_IN_IMPORT = """\
Traceback (most recent call last):
  File "/tmp/tbgen/impsyn.py", line 1, in <module>
    import syn
  File "/tmp/tbgen/syn.py", line 1
    def broken(:
               ^
SyntaxError: invalid syntax
"""

REAL_CIRCULAR_IMPORT = """\
Traceback (most recent call last):
  File "/tmp/tbgen/ca.py", line 1, in <module>
    import cb
  File "/tmp/tbgen/cb.py", line 1, in <module>
    from ca import thing
ImportError: cannot import name 'thing' from 'ca' (/tmp/tbgen/ca.py)
"""

REAL_EXEC_STRING = """\
Traceback (most recent call last):
  File "/tmp/tbgen/ex.py", line 1, in <module>
    exec("1/0")
    ~~~~^^^^^^^
  File "<string>", line 1, in <module>
ZeroDivisionError: division by zero
"""

REAL_NOTES = """\
Traceback (most recent call last):
  File "/tmp/tbgen/note.py", line 6, in <module>
    f()
    ~^^
  File "/tmp/tbgen/note.py", line 5, in f
    raise e
ValueError: bad value
while loading config
second note
"""

REAL_SUGGESTION = """\
Traceback (most recent call last):
  File "/tmp/tbgen/sug.py", line 2, in <module>
    s.strp()
    ^^^^^^
AttributeError: 'str' object has no attribute 'strp'. Did you mean: 'strip'?
"""

REAL_STDLIB_FRAMES = """\
Traceback (most recent call last):
  File "/tmp/tbgen/lib.py", line 2, in <module>
    json.loads("{bad")
    ~~~~~~~~~~^^^^^^^^
  File "/usr/local/lib/python3.13/json/__init__.py", line 352, in loads
    return _default_decoder.decode(s)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/usr/local/lib/python3.13/json/decoder.py", line 345, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/json/decoder.py", line 361, in raw_decode
    obj, end = self.scan_once(s, idx)
               ~~~~~~~~~~~~~~^^^^^^^^
json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
"""  # noqa: E501

# Hand-written: a third-party frame (needs a package that is not installed here).
HAND_THIRD_PARTY = """\
Traceback (most recent call last):
  File "/srv/app/service/client.py", line 41, in fetch
    return session.get(url, timeout=5)
  File "/srv/app/.venv/lib/python3.12/site-packages/requests/sessions.py", line 602, in get
    return self.request("GET", url, **kwargs)
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='example.org'): Max retries exceeded
"""

# Hand-written: frames without source lines and exceptions without a message.
HAND_NO_SOURCE = """\
Traceback (most recent call last):
  File "/srv/app/main.py", line 10, in <module>
  File "/srv/app/worker.py", line 22, in run
KeyboardInterrupt
"""

# Hand-written: ANSI colour codes as some terminals/loggers emit them.
HAND_ANSI = (
    "\x1b[1;31mTraceback (most recent call last):\x1b[0m\n"
    '  File "/srv/app/main.py", line 3, in <module>\n'
    "    boom()\n"
    "\x1b[1;31mRuntimeError\x1b[0m: boom\n"
)


def windows_variant(text: str) -> str:
    """Transformation: ``/tmp/tbgen/x.py`` -> ``C:\\Users\\alice\\proj\\x.py`` (real separators)."""
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith('File "/tmp/tbgen/'):
            head, _, tail = line.partition('File "')
            path, _, rest = tail.partition('"')
            name = path.rsplit("/", 1)[-1]
            line = f'{head}File "C:\\Users\\alice\\proj\\{name}"{rest}'
        out.append(line)
    return "\n".join(out)


def relative_variant(text: str) -> str:
    """Transformation: ``/tmp/tbgen/x.py`` -> ``app/x.py`` (a traceback of a relative run)."""
    return text.replace('File "/tmp/tbgen/', 'File "app/')


def many_frames(count: int, *, repeat_function: str = "f") -> str:
    """A synthetic traceback with ``count`` distinct frames (for limit tests)."""
    lines = ["Traceback (most recent call last):"]
    for n in range(count):
        lines.append(f'  File "/srv/app/m{n}.py", line {n + 1}, in {repeat_function}{n}')
        lines.append(f"    call_{n}()")
    lines.append("RuntimeError: deep")
    return "\n".join(lines) + "\n"
