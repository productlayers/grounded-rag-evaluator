"""Tests for the retrieval pipeline.

Tests cover:
- Cosine similarity math
- Index build → query round-trip
- Top-k ordering and min_score filtering
- Model mismatch detection
- Traceability metadata in results
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.retrieval.index import (
    RetrievalResult,
    _cosine_similarity,
    build_index,
    query_index,
)

# ---------------------------------------------------------------------------
# Cosine similarity unit tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        matrix = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        scores = _cosine_similarity(query, matrix)
        assert abs(scores[0] - 1.0) < 1e-5

    def test_orthogonal_vectors_score_zero(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.array([[0.0, 1.0]], dtype=np.float32)
        scores = _cosine_similarity(query, matrix)
        assert abs(scores[0]) < 1e-5

    def test_opposite_vectors_score_negative(self) -> None:
        query = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.array([[-1.0, 0.0]], dtype=np.float32)
        scores = _cosine_similarity(query, matrix)
        assert scores[0] < -0.99

    def test_multiple_vectors_ranked(self) -> None:
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        matrix = np.array(
            [
                [0.0, 1.0, 0.0],  # orthogonal → ~0
                [1.0, 0.1, 0.0],  # very similar → ~0.99
                [0.5, 0.5, 0.0],  # somewhat similar → ~0.7
            ],
            dtype=np.float32,
        )
        scores = _cosine_similarity(query, matrix)
        # Index 1 should have highest score, index 0 lowest
        assert scores[1] > scores[2] > scores[0]


# ---------------------------------------------------------------------------
# Index round-trip tests (with mocked embedder)
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """A deterministic embedder for testing: hashes text into a fixed-dim vector."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._model_name = "fake-test-model"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Create deterministic vectors from text: use char ordinals."""
        vectors = []
        for text in texts:
            # Simple deterministic embedding based on character values
            vec = np.zeros(self._dim, dtype=np.float32)
            for i, ch in enumerate(text[: self._dim]):
                vec[i % self._dim] += ord(ch)
            # Normalise
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


@pytest.fixture()
def sample_chunks_path(tmp_path: Path) -> Path:
    """Create a minimal chunks.jsonl for testing."""
    import json

    chunks = [
        {
            "chunk_id": "doc-a::chunk_000",
            "doc_id": "doc-a",
            "source_path": "data/docs/doc-a.md",
            "chunk_index": 0,
            "total_chunks": 1,
            "char_start": 0,
            "char_end": 50,
            "text": "How to reset your password and recover account access",
            "section_heading": "Account Recovery",
            "content_hash": "sha256:fake1",
            "chunking_config": {"max_chars": 500, "overlap": 100, "min_chunk_chars": 50},
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "chunk_id": "doc-b::chunk_000",
            "doc_id": "doc-b",
            "source_path": "data/docs/doc-b.md",
            "chunk_index": 0,
            "total_chunks": 1,
            "char_start": 0,
            "char_end": 45,
            "text": "Refund policy for annual and monthly subscriptions",
            "section_heading": "Refund Policy",
            "content_hash": "sha256:fake2",
            "chunking_config": {"max_chars": 500, "overlap": 100, "min_chunk_chars": 50},
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "chunk_id": "doc-c::chunk_000",
            "doc_id": "doc-c",
            "source_path": "data/docs/doc-c.md",
            "chunk_index": 0,
            "total_chunks": 1,
            "char_start": 0,
            "char_end": 40,
            "text": "Data privacy and encryption requirements for users",
            "section_heading": "Data Privacy",
            "content_hash": "sha256:fake3",
            "chunking_config": {"max_chars": 500, "overlap": 100, "min_chunk_chars": 50},
            "created_at": "2026-01-01T00:00:00Z",
        },
    ]

    path = tmp_path / "chunks.jsonl"
    with path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return path


@pytest.fixture()
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder(dim=8)


class TestIndexRoundTrip:
    def test_build_creates_index_file(
        self,
        sample_chunks_path: Path,
        fake_embedder: _FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        index_path = tmp_path / "index.json"
        monkeypatch.setattr("src.retrieval.index.get_embedder", lambda mode: fake_embedder)

        build_index(
            chunks_path=sample_chunks_path,
            index_path=index_path,
            mode="local",
        )

        assert index_path.exists()
        import json

        with index_path.open() as f:
            data = json.load(f)
        assert data["model"] == "fake-test-model"
        assert data["num_chunks"] == 3
        assert len(data["embeddings"]) == 3
        assert len(data["embeddings"][0]) == 8

    def test_query_returns_ranked_results(
        self,
        sample_chunks_path: Path,
        fake_embedder: _FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        index_path = tmp_path / "index.json"
        monkeypatch.setattr("src.retrieval.index.get_embedder", lambda mode: fake_embedder)

        build_index(
            chunks_path=sample_chunks_path,
            index_path=index_path,
            mode="local",
        )

        results = query_index(
            query="password reset account",
            index_path=index_path,
            top_k=3,
            mode="local",
        )

        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)
        # Results should be sorted by descending score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(
        self,
        sample_chunks_path: Path,
        fake_embedder: _FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        index_path = tmp_path / "index.json"
        monkeypatch.setattr("src.retrieval.index.get_embedder", lambda mode: fake_embedder)

        build_index(
            chunks_path=sample_chunks_path,
            index_path=index_path,
            mode="local",
        )

        results = query_index(
            query="test query",
            index_path=index_path,
            top_k=1,
            mode="local",
        )
        assert len(results) == 1

    def test_min_score_filters_results(
        self,
        sample_chunks_path: Path,
        fake_embedder: _FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        index_path = tmp_path / "index.json"
        monkeypatch.setattr("src.retrieval.index.get_embedder", lambda mode: fake_embedder)

        build_index(
            chunks_path=sample_chunks_path,
            index_path=index_path,
            mode="local",
        )

        # Very high min_score should return no results
        results = query_index(
            query="test query",
            index_path=index_path,
            top_k=3,
            min_score=0.9999,
            mode="local",
        )
        assert len(results) == 0

    def test_results_carry_traceability_metadata(
        self,
        sample_chunks_path: Path,
        fake_embedder: _FakeEmbedder,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        index_path = tmp_path / "index.json"
        monkeypatch.setattr("src.retrieval.index.get_embedder", lambda mode: fake_embedder)

        build_index(
            chunks_path=sample_chunks_path,
            index_path=index_path,
            mode="local",
        )

        results = query_index(
            query="account recovery",
            index_path=index_path,
            top_k=1,
            mode="local",
        )

        r = results[0]
        # Every result must carry traceability metadata
        assert r.chunk_id is not None
        assert r.doc_id is not None
        assert r.source_path is not None
        assert isinstance(r.char_start, int)
        assert isinstance(r.char_end, int)
        assert r.char_end > r.char_start
        assert len(r.text) > 0
        assert isinstance(r.score, float)

    def test_model_mismatch_raises(
        self,
        sample_chunks_path: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If query model differs from build model, raise ValueError."""
        index_path = tmp_path / "index.json"

        build_embedder = _FakeEmbedder(dim=8)
        build_embedder._model_name = "model-alpha"

        query_embedder = _FakeEmbedder(dim=8)
        query_embedder._model_name = "model-beta"

        call_count = {"n": 0}

        def mock_get_embedder(mode: str) -> _FakeEmbedder:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return build_embedder
            return query_embedder

        monkeypatch.setattr("src.retrieval.index.get_embedder", mock_get_embedder)

        build_index(
            chunks_path=sample_chunks_path,
            index_path=index_path,
            mode="local",
        )

        with pytest.raises(ValueError, match="Model mismatch"):
            query_index(
                query="test",
                index_path=index_path,
                mode="local",
            )
