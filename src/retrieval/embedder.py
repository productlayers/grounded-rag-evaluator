"""Embedding backends — turn text into vectors.

Two implementations behind a common interface:

* **LocalEmbedder** (default): Uses ``sentence-transformers`` with
  ``all-MiniLM-L6-v2`` (384-dimensional vectors).  Runs entirely on CPU.
  The model is downloaded once (~90 MB) and cached by Hugging Face.

* **OpenAIEmbedder**: Calls the OpenAI ``text-embedding-3-small`` API
  (1536-dimensional vectors).  Requires ``OPENAI_API_KEY`` in the
  environment or ``.env`` file.  Install the optional ``openai`` extra:
  ``uv sync --extra openai``.

Design note: We batch all texts in a single call for efficiency rather
than embedding one-by-one.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Embedder(ABC):
    """Common interface for embedding backends."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier stored in the index for model-mismatch detection."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the output vectors."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), self.dim)``, dtype ``float32``.
        """


# ---------------------------------------------------------------------------
# Local: sentence-transformers
# ---------------------------------------------------------------------------

_DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"


class LocalEmbedder(Embedder):
    """Embed text using a local sentence-transformers model.

    The model is downloaded on first use (~90 MB for the default) and
    cached by the Hugging Face hub.  No API key or internet is required
    after the first download.
    """

    def __init__(self, model_name: str = _DEFAULT_LOCAL_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dim: int = self._model.get_embedding_dimension()  # type: ignore[assignment]
        logger.info("Model loaded (dim=%d)", self._dim)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the local model.

        Uses ``normalize_embeddings=True`` so cosine similarity reduces
        to a simple dot product.
        """
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

_DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


class OpenAIEmbedder(Embedder):
    """Embed text using the OpenAI Embeddings API.

    Requires ``OPENAI_API_KEY`` in the environment.  Install the
    optional extra with ``uv sync --extra openai``.
    """

    # text-embedding-3-small produces 1536-dim vectors
    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model_name: str = _DEFAULT_OPENAI_MODEL) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set.  Either:\n"
                "  1. Set it in your .env file and export it, or\n"
                "  2. Use --mode local (default) for local embeddings."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is not installed.\nRun: uv sync --extra openai"
            ) from None

        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._dim = self._DIMS.get(model_name, 1536)
        logger.info("Using OpenAI embedding model: %s (dim=%d)", model_name, self._dim)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Call the OpenAI embeddings API with batched input."""
        response = self._client.embeddings.create(input=texts, model=self._model_name)
        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_embedder_cache: dict[str, Embedder] = {}


def get_embedder(mode: str = "local") -> Embedder:
    """Create an embedder based on *mode*.

    Parameters
    ----------
    mode:
        ``"local"`` (default) for sentence-transformers, or ``"openai"``
        for the OpenAI API.
    """
    if mode in _embedder_cache:
        return _embedder_cache[mode]

    if mode == "local":
        embedder = LocalEmbedder()
    elif mode == "openai":
        embedder = OpenAIEmbedder()
    else:
        raise ValueError(f"Unknown embedding mode: {mode!r}. Use 'local' or 'openai'.")

    _embedder_cache[mode] = embedder
    return embedder
