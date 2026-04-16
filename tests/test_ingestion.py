"""Smoke tests for the ingestion pipeline.

These tests verify the core traceability invariant:
  source_text[char_start:char_end] == chunk.text

and exercise the loader, chunker, and writer against known inputs.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.ingestion.chunker import ChunkingConfig, chunk_document
from src.ingestion.loader import load_documents
from src.ingestion.writer import write_chunks
from src.utils.hashing import content_hash
from src.utils.io import read_jsonl

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_text() -> str:
    """A ~600-char document with two markdown headings."""
    return textwrap.dedent("""\
        # Alpha Section

        This is the first section of the document. It contains enough text to
        exercise the chunker with our default settings. We need several lines
        of content to ensure that at least two chunks are produced when using
        a max_chars value of 300.

        ## Beta Section

        The second section covers different material. It includes details that
        a retrieval system should be able to locate independently from the
        first section. This tests that section_heading extraction works for
        chunks that start mid-document.

        End of document.
    """)


@pytest.fixture()
def tmp_docs(tmp_path: Path, sample_text: str) -> Path:
    """Create a temporary docs directory with sample files."""
    docs_dir = tmp_path / "data" / "docs"
    docs_dir.mkdir(parents=True)

    (docs_dir / "alpha.md").write_text(sample_text, encoding="utf-8")

    sub = docs_dir / "sub"
    sub.mkdir()
    (sub / "beta.md").write_text("# Beta\n\nShort doc.\n", encoding="utf-8")

    return docs_dir


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestLoader:
    def test_loads_supported_files(self, tmp_docs: Path) -> None:
        docs = load_documents(tmp_docs, project_root=tmp_docs.parent.parent)
        assert len(docs) == 2

    def test_doc_id_uses_relative_path(self, tmp_docs: Path) -> None:
        docs = load_documents(tmp_docs, project_root=tmp_docs.parent.parent)
        ids = {d.doc_id for d in docs}
        assert "alpha" in ids
        assert "sub/beta" in ids

    def test_skips_unsupported_extensions(self, tmp_docs: Path) -> None:
        (tmp_docs / "image.png").write_bytes(b"\x89PNG")
        docs = load_documents(tmp_docs, project_root=tmp_docs.parent.parent)
        assert all(d.doc_id != "image" for d in docs)

    def test_raises_on_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_documents(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------


class TestChunker:
    def test_single_chunk_for_short_text(self) -> None:
        chunks = chunk_document("Hello world", doc_id="test", source_path="test.md")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 11

    def test_traceability_invariant(self, sample_text: str) -> None:
        """The core contract: text[char_start:char_end] == chunk.text."""
        config = ChunkingConfig(max_chars=300, overlap=50, min_chunk_chars=50)
        chunks = chunk_document(sample_text, doc_id="test", source_path="test.md", config=config)

        assert len(chunks) >= 2, "Expected multiple chunks for a 600-char doc with max_chars=300"

        for chunk in chunks:
            actual = sample_text[chunk.char_start : chunk.char_end]
            assert actual == chunk.text, (
                f"Traceability broken for {chunk.chunk_id}: "
                f"source[{chunk.char_start}:{chunk.char_end}] != chunk.text"
            )

    def test_chunk_ids_are_unique(self, sample_text: str) -> None:
        config = ChunkingConfig(max_chars=200, overlap=50)
        chunks = chunk_document(sample_text, doc_id="test", source_path="test.md", config=config)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_content_hash_matches(self, sample_text: str) -> None:
        chunks = chunk_document(sample_text, doc_id="test", source_path="test.md")
        for chunk in chunks:
            assert chunk.content_hash == content_hash(chunk.text)

    def test_total_chunks_is_correct(self, sample_text: str) -> None:
        config = ChunkingConfig(max_chars=200, overlap=50)
        chunks = chunk_document(sample_text, doc_id="test", source_path="test.md", config=config)
        for chunk in chunks:
            assert chunk.total_chunks == len(chunks)

    def test_section_heading_extracted(self, sample_text: str) -> None:
        config = ChunkingConfig(max_chars=300, overlap=50)
        chunks = chunk_document(sample_text, doc_id="test", source_path="test.md", config=config)
        # First chunk should have "Alpha Section" heading
        assert chunks[0].section_heading == "Alpha Section"

    def test_runt_merging(self) -> None:
        """Chunks shorter than min_chunk_chars merge into the previous chunk."""
        text = "A" * 520  # 500 + 20 → runt of 20 chars (with overlap, slightly different)
        config = ChunkingConfig(max_chars=500, overlap=0, min_chunk_chars=50)
        chunks = chunk_document(text, doc_id="test", source_path="test.md", config=config)

        # Should merge into a single chunk since the runt (20 chars) < 50
        assert len(chunks) == 1
        assert chunks[0].char_end == 520

    def test_empty_text_returns_no_chunks(self) -> None:
        chunks = chunk_document("", doc_id="test", source_path="test.md")
        assert chunks == []

    def test_chunking_config_recorded(self, sample_text: str) -> None:
        config = ChunkingConfig(max_chars=300, overlap=75, min_chunk_chars=40)
        chunks = chunk_document(sample_text, doc_id="test", source_path="test.md", config=config)
        for chunk in chunks:
            assert chunk.chunking_config["max_chars"] == 300
            assert chunk.chunking_config["overlap"] == 75
            assert chunk.chunking_config["min_chunk_chars"] == 40


# ---------------------------------------------------------------------------
# Writer tests
# ---------------------------------------------------------------------------


class TestWriter:
    def test_writes_valid_jsonl(self, sample_text: str, tmp_path: Path) -> None:
        config = ChunkingConfig(max_chars=300, overlap=50)
        chunks = chunk_document(sample_text, doc_id="test", source_path="test.md", config=config)
        output = tmp_path / "chunks.jsonl"
        write_chunks(chunks, output)

        # Verify every line is valid JSON with expected fields
        records = read_jsonl(output)
        assert len(records) == len(chunks)

        required_fields = {
            "chunk_id",
            "doc_id",
            "source_path",
            "chunk_index",
            "total_chunks",
            "char_start",
            "char_end",
            "text",
            "section_heading",
            "content_hash",
            "chunking_config",
            "created_at",
        }
        for record in records:
            assert required_fields.issubset(record.keys()), (
                f"Missing fields: {required_fields - record.keys()}"
            )


# ---------------------------------------------------------------------------
# Hashing tests
# ---------------------------------------------------------------------------


class TestHashing:
    def test_hash_is_deterministic(self) -> None:
        assert content_hash("hello") == content_hash("hello")

    def test_hash_has_prefix(self) -> None:
        h = content_hash("hello")
        assert h.startswith("sha256:")

    def test_different_input_different_hash(self) -> None:
        assert content_hash("hello") != content_hash("world")
