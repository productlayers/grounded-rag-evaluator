"""Retrieval index — build from chunks, query with top-k.

The index is a JSON file containing:
  - ``model``: the embedding model name (for mismatch detection)
  - ``dim``: vector dimensionality
  - ``chunks``: list of chunk metadata (id, doc_id, text, offsets, etc.)
  - ``embeddings``: list of float lists (one per chunk)

Query flow:
  1. Embed the user query using the same model
  2. Compute cosine similarity against all chunk vectors
  3. Rank by score, return top-k above min_score
  4. Each result carries full traceability metadata
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.retrieval.embedder import get_embedder
from src.utils.io import read_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
_DEFAULT_INDEX_PATH = Path("data/processed/retrieval_index.json")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieval result with score and full chunk metadata."""

    chunk_id: str
    doc_id: str
    score: float
    text: str
    section_heading: str | None
    source_path: str
    char_start: int
    char_end: int


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------


def build_index(
    chunks_path: Path = _DEFAULT_CHUNKS_PATH,
    index_path: Path = _DEFAULT_INDEX_PATH,
    mode: str = "local",
) -> None:
    """Embed all chunks and save the retrieval index.

    Parameters
    ----------
    chunks_path:
        Path to the ``chunks.jsonl`` produced by the ingestion pipeline.
    index_path:
        Output path for the retrieval index JSON.
    mode:
        Embedding mode: ``"local"`` or ``"openai"``.
    """
    chunks = read_jsonl(chunks_path)
    if not chunks:
        logger.warning("No chunks found in %s", chunks_path)
        return

    logger.info("Building index from %d chunks (%s)", len(chunks), chunks_path)

    embedder = get_embedder(mode)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)

    # Strip the text and large fields from the stored metadata to keep
    # the index file focused.  We store enough to reconstruct results.
    chunk_meta = []
    for c in chunks:
        chunk_meta.append(
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "source_path": c["source_path"],
                "char_start": c["char_start"],
                "char_end": c["char_end"],
                "text": c["text"],
                "section_heading": c.get("section_heading"),
            }
        )

    index_data = {
        "model": embedder.model_name,
        "dim": embedder.dim,
        "num_chunks": len(chunks),
        "chunks": chunk_meta,
        "embeddings": embeddings.tolist(),
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False)

    logger.info(
        "Index saved to %s (%d chunks, dim=%d, model=%s)",
        index_path,
        len(chunks),
        embedder.dim,
        embedder.model_name,
    )


# ---------------------------------------------------------------------------
# Query index
# ---------------------------------------------------------------------------


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix of vectors.

    If vectors are already L2-normalised (as sentence-transformers produces
    with ``normalize_embeddings=True``), this reduces to a dot product.
    We normalise here anyway for safety in case OpenAI vectors aren't normalised.
    """
    # Normalise query
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    # Normalise matrix rows
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix_norm = matrix / row_norms
    # Dot product = cosine similarity on normalised vectors
    return matrix_norm @ query_norm


def query_index(
    query: str,
    index_path: Path = _DEFAULT_INDEX_PATH,
    top_k: int = 3,
    min_score: float = 0.0,
    mode: str | None = None,
) -> list[RetrievalResult]:
    """Retrieve the top-k most relevant chunks for a query.

    Parameters
    ----------
    query:
        The user's question.
    index_path:
        Path to the retrieval index JSON.
    top_k:
        Number of results to return.
    min_score:
        Minimum cosine similarity score to include a result.
    mode:
        Embedding mode override.  If ``None``, inferred from the index's
        stored model name.

    Returns
    -------
    list[RetrievalResult]
        Top-k results ordered by descending similarity score.
    """
    with index_path.open("r", encoding="utf-8") as f:
        index_data = json.load(f)

    stored_model = index_data["model"]
    chunks = index_data["chunks"]
    embeddings = np.asarray(index_data["embeddings"], dtype=np.float32)

    # Determine embedding mode from stored model if not specified
    if mode is None:
        if stored_model.startswith("text-embedding"):
            mode = "openai"
        else:
            mode = "local"

    embedder = get_embedder(mode)

    # Validate model match
    if embedder.model_name != stored_model:
        raise ValueError(
            f"Model mismatch: index was built with '{stored_model}' "
            f"but current embedder uses '{embedder.model_name}'.  "
            f"Rebuild the index with: python -m src.app.main retrieve-build --mode {mode}"
        )

    # Embed query
    query_vec = embedder.embed_texts([query])[0]

    # Compute similarities
    scores = _cosine_similarity(query_vec, embeddings)

    # Rank and filter
    ranked_indices = np.argsort(scores)[::-1]
    results: list[RetrievalResult] = []

    for idx in ranked_indices[:top_k]:
        score = float(scores[idx])
        if score < min_score:
            break
        chunk = chunks[idx]
        results.append(
            RetrievalResult(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                score=round(score, 4),
                text=chunk["text"],
                section_heading=chunk.get("section_heading"),
                source_path=chunk["source_path"],
                char_start=chunk["char_start"],
                char_end=chunk["char_end"],
            )
        )

    logger.info(
        "Query: %r → %d results (top score: %.4f)",
        query[:60],
        len(results),
        results[0].score if results else 0.0,
    )

    return results
