"""Builder for a synthetic repository used by the ingestion tests.

The tree is generated at test time (into ``tmp_path``) rather than committed, because it must
contain things that must never live in Git: ``.env``-style files, key-like files, a binary and an
oversized file. Everything is fake; nothing here resembles a real credential, and ingestion never
reads the sensitive files anyway (they are rejected by name).

Run ``python -m tests.fixtures.synthetic_repo OUTPUT_DIR`` to materialise it for a manual demo.

The ``EXPECTED_*`` tables are written by hand from the design of the tree, not derived from the
ingestion output, so tests compare the implementation against an independent statement of intent.
"""

from __future__ import annotations

import sys
from pathlib import Path

from copilot.models.ingestion import SkipReason

# Policy limit the tests use; huge_module.py deliberately exceeds it, everything else is far below.
SYNTHETIC_MAX_FILE_SIZE = 4096

MAIN_PY_MARKER = "SYNTHETIC_CONTENT_MARKER_7f3a"  # lets logging tests prove contents are not logged
UTF16_TEXT = "def größe():\n    return 'ünïcode'\n"

# relative path -> content (str is written UTF-8, bytes verbatim)
FILES: dict[str, str | bytes] = {
    # ---- accepted -------------------------------------------------------------------------
    "README.md": "# Synthetic repo\n\nUsed only by ingestion tests.\n",
    "docs/guide.md": "# Guide\n\nHow it works.\n",
    "src/app/__init__.py": "",
    "src/app/main.py": (
        f'"""Entry point."""\n\nMARKER = "{MAIN_PY_MARKER}"\n\n\ndef main():\n    return 0\n'
    ),
    "src/app/utils/helpers.py": "def helper(x):\n    return x + 1\n",
    "src/app/secrets.py": "def load_secret():\n    raise NotImplementedError\n",
    "src/app/crlf_module.py": b"def a():\r\n    return 1\r\n\r\n\r\ndef b():\r\n    return 2\r\n",
    "src/app/utf8_bom.py": b"\xef\xbb\xbfVALUE = 1\n",
    "src/app/utf16_module.py": UTF16_TEXT.encode("utf-16"),  # includes a BOM
    "web/index.js": "console.log('hi');\n",
    "web/App.jsx": "export const App = () => <div/>;\n",
    "web/api.ts": "export const x: number = 1;\n",
    "web/Widget.tsx": "export const W = () => <span/>;\n",
    "lib/Main.java": "public class Main { public static void main(String[] a) {} }\n",
    "native/util.c": "int add(int a, int b) { return a + b; }\n",
    "native/engine.cpp": "int run() { return 0; }\n",
    "config/settings.json": '{"debug": false}\n',
    "config/ci.yml": "steps:\n  - run: pytest\n",
    "config/app.yaml": "name: synthetic\n",
    "models/tokenizer.json": '{"vocab": {}}\n',  # a name that merely *starts* with "token"
    "UPPER/SCRIPT.PY": "print('upper-case extension')\n",
    # ---- skipped: unsupported / binary / oversized / encoding -------------------------------
    "notes.txt": "plain text is not a supported type\n",
    "Makefile": "all:\n\techo hi\n",
    "assets/logo.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
    "assets/fake_binary.py": b"MZ\x90\x00\x03\x00\x00\x00binary\x00\x00\x01\x02",
    "big/huge_module.py": "x = 1\n" * 2000,  # 12,000 bytes > SYNTHETIC_MAX_FILE_SIZE
    "bad_encoding.py": b"name = '\xe9'\n",  # lone 0xE9: valid latin-1, invalid UTF-8
    # ---- skipped: sensitive (all contents are obviously fake) -------------------------------
    ".env": "FAKE_SETTING=not-a-real-value\n",
    ".env.production": "FAKE_SETTING=not-a-real-value\n",
    "deploy/server.pem": "fake pem placeholder, not a certificate\n",
    "keys/private.key": "fake key placeholder\n",
    "keys/id_rsa": "fake ssh key placeholder\n",
    "config/secrets.yaml": "placeholder: not-a-real-value\n",
    "config/credentials.json": '{"placeholder": "not-a-real-value"}\n',
    # ---- skipped: generated / vendored ------------------------------------------------------
    "package-lock.json": '{"lockfileVersion": 3}\n',
    "web/vendor.min.js": "!function(){}();\n",
    # ---- inside pruned directories: must never be enumerated -------------------------------
    "node_modules/left-pad/index.js": "module.exports = 1;\n",
    ".git/notes.md": "not a real git dir\n",
    "venv/lib/site.py": "x = 1\n",
    ".venv/lib/site.py": "x = 1\n",
    "__pycache__/mod.py": "x = 1\n",
    "src/app/__pycache__/main.py": "x = 1\n",
    "build/out.py": "x = 1\n",
    "dist/bundle.js": "x = 1;\n",
    ".pytest_cache/README.md": "cache\n",
    ".mypy_cache/cache.json": "{}\n",
    "coverage/report.md": "report\n",
    ".idea/workspace.json": "{}\n",
    ".ssh/notes.md": "sensitive directory\n",
    "mypkg.egg-info/PKG-INFO.md": "meta\n",
}

PRUNED_PREFIXES = (
    "node_modules/", ".git/", "venv/", ".venv/", "__pycache__/", "src/app/__pycache__/",
    "build/", "dist/", ".pytest_cache/", ".mypy_cache/", "coverage/", ".idea/", ".ssh/",
    "mypkg.egg-info/",
)  # fmt: skip

EXPECTED_PRUNED: dict[str, int] = {
    ".git": 1, ".idea": 1, ".mypy_cache": 1, ".pytest_cache": 1, ".ssh": 1, ".venv": 1,
    "__pycache__": 2, "build": 1, "coverage": 1, "dist": 1, "mypkg.egg-info": 1,
    "node_modules": 1, "venv": 1,
}  # fmt: skip

EXPECTED_ACCEPTED: tuple[str, ...] = tuple(
    sorted(
        [
            "README.md", "docs/guide.md", "src/app/__init__.py", "src/app/main.py",
            "src/app/utils/helpers.py", "src/app/secrets.py", "src/app/crlf_module.py",
            "src/app/utf8_bom.py", "src/app/utf16_module.py", "web/index.js", "web/App.jsx",
            "web/api.ts", "web/Widget.tsx", "lib/Main.java", "native/util.c", "native/engine.cpp",
            "config/settings.json", "config/ci.yml", "config/app.yaml", "models/tokenizer.json",
            "UPPER/SCRIPT.PY",
        ]
    )
)  # fmt: skip

EXPECTED_SKIPPED: dict[str, SkipReason] = {
    "notes.txt": SkipReason.UNSUPPORTED_EXTENSION,
    "Makefile": SkipReason.UNSUPPORTED_EXTENSION,
    "assets/logo.png": SkipReason.UNSUPPORTED_EXTENSION,
    "assets/fake_binary.py": SkipReason.BINARY,
    "big/huge_module.py": SkipReason.OVERSIZED,
    "bad_encoding.py": SkipReason.ENCODING_ERROR,
    ".env": SkipReason.SENSITIVE,
    ".env.production": SkipReason.SENSITIVE,
    "deploy/server.pem": SkipReason.SENSITIVE,
    "keys/private.key": SkipReason.SENSITIVE,
    "keys/id_rsa": SkipReason.SENSITIVE,
    "config/secrets.yaml": SkipReason.SENSITIVE,
    "config/credentials.json": SkipReason.SENSITIVE,
    "package-lock.json": SkipReason.IGNORED_FILE,
    "web/vendor.min.js": SkipReason.IGNORED_FILE,
}

EXPECTED_LANGUAGE_COUNTS: dict[str, int] = {
    "python": 8, "javascript": 2, "typescript": 2, "java": 1, "c": 1, "cpp": 1,
    "markdown": 2, "json": 2, "yaml": 2,
}  # fmt: skip


def build_synthetic_repo(root: Path) -> Path:
    """Create the synthetic repository under ``root`` (created if needed) and return it."""
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in FILES.items():
        target = root.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8", newline="")
    return root


if __name__ == "__main__":  # pragma: no cover - manual helper
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m tests.fixtures.synthetic_repo OUTPUT_DIR")
    print(build_synthetic_repo(Path(sys.argv[1])))
