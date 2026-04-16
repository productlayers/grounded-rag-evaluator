"""Content hashing for drift detection.

Produces SHA-256 hashes prefixed with 'sha256:' to enable staleness checks
between ingested chunks and their source documents.
"""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """Return a prefixed SHA-256 hex digest of *text*.

    >>> content_hash("hello")
    'sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
