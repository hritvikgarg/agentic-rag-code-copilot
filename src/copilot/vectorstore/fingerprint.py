"""Deterministic repository fingerprint: "what accepted source state was indexed?".

Formula (``FINGERPRINT_SCHEMA = "repo-fingerprint/1"``)::

    entries = sorted( [relative_path, sha256_of_raw_bytes, sha256_of_normalised_text, language]
                      for every accepted file )
    fingerprint = SHA-256( json.dumps([FINGERPRINT_SCHEMA, entries],
                                      ensure_ascii=True, separators=(",", ":")) )   # 64 hex chars

Design decisions:

* **Content-addressed, not location-addressed.** No absolute path, no timestamps, no random ids.
  Moving or re-cloning the repository to another directory gives the same fingerprint.
* **Order-independent input, deterministic output.** Entries are sorted (Python string order of the
  canonical POSIX path), so filesystem traversal order can never matter.
* **Both hashes.** The raw-bytes hash is what chunk ids are built from; the hash of the decoded,
  newline-normalised text is what the chunkers actually see, so a change in decoding/normalisation
  is detected even when the bytes are unchanged.
* **Language is included** because the default embedding representation writes it into the
  embedded text; a changed extension-to-language mapping therefore changes the vectors.
* **The repository name is NOT part of the fingerprint** (it is a separate field of the index
  identity), so the fingerprint answers only "is this the same source content?".
* Accepted files only. Skipped files (binary, secrets, ...) do not influence it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from copilot.models.chunk import sha256_text
from copilot.models.ingestion import SourceFile
from copilot.vectorstore.errors import IndexBuildError

FINGERPRINT_SCHEMA = "repo-fingerprint/1"


def repository_fingerprint(files: Iterable[SourceFile]) -> str:
    """SHA-256 hex fingerprint of the accepted source state (see the module docstring)."""
    entries = [[f.relative_path, f.sha256, sha256_text(f.content), f.language] for f in files]
    entries.sort()
    paths = [entry[0] for entry in entries]
    if len(set(paths)) != len(paths):
        raise IndexBuildError("duplicate relative paths in the accepted files")
    text = json.dumps([FINGERPRINT_SCHEMA, entries], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
