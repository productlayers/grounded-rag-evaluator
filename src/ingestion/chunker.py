"""Character-offset-aware text chunker.

Splits a document into fixed-size chunks (measured in characters) while
tracking the exact start/end offsets in the original text.  This is the
core mechanism for satisfying the Full Traceability requirement: every
chunk maps back to a precise character range in its source file.

Design decisions
----------------
* **Fixed-size, not semantic:** We split on character count, not sentence
  boundaries.  This is intentional — semantic splitting introduces an
  NLP dependency and non-deterministic behaviour.  The eval harness will
  tell us if this hurts retrieval quality, and we can iterate.
* **Overlap:** Adjacent chunks share ``overlap`` characters to avoid
  splitting mid-sentence.  Tracked in ``chunking_config`` for
  reproducibility.
* **Runt merging:** If the final chunk is shorter than ``min_chunk_chars``,
  it is merged into the previous chunk rather than emitted as a low-signal
  standalone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class ChunkingConfig:
    """Parameters that control chunking behaviour."""

    max_chars: int = 500
    overlap: int = 100
    min_chunk_chars: int = 50


@dataclass(frozen=True)
class Chunk:
    """A single chunk with full traceability metadata."""

    chunk_id: str
    doc_id: str
    source_path: str
    chunk_index: int
    total_chunks: int  # set after all chunks are produced
    char_start: int
    char_end: int
    text: str
    section_heading: str | None
    content_hash: str
    chunking_config: dict[str, int] = field(default_factory=dict)
    created_at: str = ""


# ---------------------------------------------------------------------------
# Heading extraction
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _find_heading_at(text: str, offset: int) -> str | None:
    """Return the nearest markdown heading at or before *offset*.

    Scans all headings in *text* and returns the one whose position is
    closest to (but not after) *offset*.  Returns ``None`` if no heading
    precedes the offset.
    """
    best: str | None = None
    for match in _HEADING_RE.finditer(text):
        if match.start() <= offset:
            best = match.group(2).strip()
        else:
            break  # headings are in document order; past our offset
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    text: str,
    doc_id: str,
    source_path: str,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Split *text* into chunks with character-offset metadata.

    Parameters
    ----------
    text:
        Full text content of the source document.
    doc_id:
        Document identifier (relative path, no extension).
    source_path:
        Path to the source file relative to the project root.
    config:
        Chunking parameters.  Uses defaults if not provided.

    Returns
    -------
    list[Chunk]
        Ordered list of chunks.  Each chunk's ``char_start`` / ``char_end``
        can be used to slice *text* and recover the chunk content exactly.
    """
    from src.utils.hashing import content_hash

    if config is None:
        config = ChunkingConfig()

    now = datetime.now(timezone.utc).isoformat()
    config_dict = {
        "max_chars": config.max_chars,
        "overlap": config.overlap,
        "min_chunk_chars": config.min_chunk_chars,
    }

    if not text:
        return []

    # --- Build raw spans ------------------------------------------------
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + config.max_chars, len(text))
        spans.append((start, end))
        if end == len(text):
            break
        start = end - config.overlap

    # --- Merge runt (small final chunk) into previous -------------------
    if len(spans) >= 2:
        last_start, last_end = spans[-1]
        if (last_end - last_start) < config.min_chunk_chars:
            prev_start, _prev_end = spans[-2]
            spans[-2] = (prev_start, last_end)
            spans.pop()

    # --- Build Chunk objects -------------------------------------------
    total = len(spans)
    chunks: list[Chunk] = []

    for idx, (char_start, char_end) in enumerate(spans):
        chunk_text = text[char_start:char_end]
        heading = _find_heading_at(text, char_start)

        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::chunk_{idx:03d}",
                doc_id=doc_id,
                source_path=source_path,
                chunk_index=idx,
                total_chunks=total,
                char_start=char_start,
                char_end=char_end,
                text=chunk_text,
                section_heading=heading,
                content_hash=content_hash(chunk_text),
                chunking_config=config_dict,
                created_at=now,
            )
        )

    return chunks
