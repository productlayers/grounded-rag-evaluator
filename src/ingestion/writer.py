"""Chunk writer — serialises chunks to JSONL and prints summary stats.

This is the final stage of the ingestion pipeline.  It converts Chunk
dataclasses to dicts, writes them as JSONL, and logs a summary table
showing per-document chunk counts and character coverage.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from src.ingestion.chunker import Chunk
from src.utils.io import write_jsonl

logger = logging.getLogger(__name__)


def write_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Serialise *chunks* to JSONL at *output_path* and log summary stats.

    Parameters
    ----------
    chunks:
        Ordered list of Chunk objects from the chunker.
    output_path:
        Destination file (e.g. ``data/processed/chunks.jsonl``).
    """
    records = [asdict(c) for c in chunks]
    write_jsonl(records, output_path)

    # --- Summary statistics --------------------------------------------
    doc_ids = sorted({c.doc_id for c in chunks})
    total_chars = sum(c.char_end - c.char_start for c in chunks)

    logger.info("=" * 60)
    logger.info("Ingestion Summary")
    logger.info("=" * 60)
    logger.info("  Output:      %s", output_path)
    logger.info("  Documents:   %d", len(doc_ids))
    logger.info("  Chunks:      %d", len(chunks))
    logger.info("  Total chars: %d", total_chars)
    logger.info("-" * 60)

    for doc_id in doc_ids:
        doc_chunks = [c for c in chunks if c.doc_id == doc_id]
        char_range = f"{doc_chunks[0].char_start}–{doc_chunks[-1].char_end}"
        logger.info("  %-35s %2d chunks  chars %s", doc_id, len(doc_chunks), char_range)

    logger.info("=" * 60)
