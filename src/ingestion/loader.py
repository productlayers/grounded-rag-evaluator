"""Document loader — reads source files from a docs directory.

Walks the directory tree, reads each file as UTF-8, and returns a list of
(text, metadata) tuples.  The metadata includes everything needed to derive
a doc_id and trace back to the original file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# File extensions we know how to ingest.  Extendable later.
_SUPPORTED_EXTENSIONS: set[str] = {".md", ".txt"}


@dataclass(frozen=True)
class LoadedDocument:
    """A single source document loaded into memory."""

    text: str
    source_path: str  # relative to project root, e.g. "data/docs/support/faq.md"
    doc_id: str  # relative to docs_dir, no extension, e.g. "support/faq"


def load_documents(docs_dir: Path, project_root: Path | None = None) -> list[LoadedDocument]:
    """Walk *docs_dir* and return all supported documents.

    Parameters
    ----------
    docs_dir:
        Root directory containing source documents (e.g. ``data/docs/``).
    project_root:
        Used to compute ``source_path`` as a relative path.  Defaults to
        ``docs_dir.parent.parent`` (assumes ``<root>/data/docs/``).

    Returns
    -------
    list[LoadedDocument]
        One entry per successfully loaded file, sorted by ``doc_id`` for
        deterministic ordering.

    Raises
    ------
    FileNotFoundError
        If *docs_dir* does not exist.
    """
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"docs directory not found: {docs_dir}")

    if project_root is None:
        project_root = docs_dir.parent.parent  # data/docs/ → project root

    documents: list[LoadedDocument] = []

    for file_path in sorted(docs_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            logger.debug("Skipping unsupported file: %s", file_path)
            continue

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF-8 file: %s", file_path)
            continue

        # doc_id: relative path from docs_dir, no extension
        relative = file_path.relative_to(docs_dir)
        doc_id = str(relative.with_suffix(""))

        # source_path: relative path from project root
        source_path = str(file_path.relative_to(project_root))

        documents.append(LoadedDocument(text=text, source_path=source_path, doc_id=doc_id))
        logger.info("Loaded %s (%d chars)", doc_id, len(text))

    logger.info("Loaded %d documents from %s", len(documents), docs_dir)
    return documents
