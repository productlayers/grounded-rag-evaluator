"""Tests for the evaluation metrics and harness.

Tests cover:
- Individual metric functions (retrieval hit, OOD decline, citations,
  groundedness)
- Edge cases (empty citations, missing doc_ids)
"""

from __future__ import annotations

from src.evals.metrics import (
    citations_grounded_strict,
    has_citations,
    ood_declined,
    retrieval_hit,
    retrieval_hit_from_scores,
)
from src.generation.grounded_answer import Citation, GenerationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    citations: list[Citation] | None = None,
    insufficient_evidence: bool = False,
    answer: str = "Some answer.",
) -> GenerationResult:
    """Create a GenerationResult for testing."""
    return GenerationResult(
        question="test question",
        answer=answer,
        citations=citations or [],
        insufficient_evidence=insufficient_evidence,
        retrieval_scores=[0.5],
        mode="retrieval",
    )


def _make_citation(
    *,
    chunk_id: str = "doc::chunk_000",
    doc_id: str = "doc",
    cited_text: str = "Some cited text.",
) -> Citation:
    return Citation(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_path="data/docs/doc.md",
        char_start=0,
        char_end=50,
        cited_text=cited_text,
    )


# ---------------------------------------------------------------------------
# Retrieval hit
# ---------------------------------------------------------------------------


class TestRetrievalHit:
    def test_hit_when_expected_doc_in_citations(self) -> None:
        c = _make_citation(doc_id="support/faq")
        result = _make_result(citations=[c])
        assert retrieval_hit(result, "support/faq") is True

    def test_miss_when_wrong_doc(self) -> None:
        c = _make_citation(doc_id="billing/refund-policy")
        result = _make_result(citations=[c])
        assert retrieval_hit(result, "support/faq") is False

    def test_miss_when_no_citations(self) -> None:
        result = _make_result(citations=[])
        assert retrieval_hit(result, "support/faq") is False

    def test_hit_from_scores_list(self) -> None:
        doc_ids = ["support/faq", "billing/refund-policy"]
        assert retrieval_hit_from_scores(doc_ids, "support/faq") is True

    def test_miss_from_scores_list(self) -> None:
        doc_ids = ["billing/refund-policy", "security/data-privacy"]
        assert retrieval_hit_from_scores(doc_ids, "support/faq") is False


# ---------------------------------------------------------------------------
# OOD decline
# ---------------------------------------------------------------------------


class TestOODDecline:
    def test_correctly_declined(self) -> None:
        result = _make_result(insufficient_evidence=True)
        assert ood_declined(result) is True

    def test_incorrectly_answered(self) -> None:
        result = _make_result(insufficient_evidence=False)
        assert ood_declined(result) is False


# ---------------------------------------------------------------------------
# Citation rate
# ---------------------------------------------------------------------------


class TestHasCitations:
    def test_has_citations(self) -> None:
        c = _make_citation()
        result = _make_result(citations=[c])
        assert has_citations(result) is True

    def test_no_citations(self) -> None:
        result = _make_result(citations=[])
        assert has_citations(result) is False


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------


class TestGroundedness:
    def test_grounded_when_text_in_chunk(self) -> None:
        c = _make_citation(chunk_id="doc::chunk_000", cited_text="Hello world")
        result = _make_result(citations=[c])
        chunk_texts = {"doc::chunk_000": "Prefix Hello world suffix"}
        assert citations_grounded_strict(result, chunk_texts) is True

    def test_not_grounded_when_text_missing(self) -> None:
        c = _make_citation(chunk_id="doc::chunk_000", cited_text="Invented text")
        result = _make_result(citations=[c])
        chunk_texts = {"doc::chunk_000": "Completely different content here"}
        assert citations_grounded_strict(result, chunk_texts) is False

    def test_grounded_with_no_citations(self) -> None:
        result = _make_result(citations=[])
        assert citations_grounded_strict(result, {}) is True

    def test_not_grounded_when_chunk_id_missing(self) -> None:
        c = _make_citation(chunk_id="nonexistent::chunk_999", cited_text="Anything")
        result = _make_result(citations=[c])
        chunk_texts = {"doc::chunk_000": "Some content"}
        assert citations_grounded_strict(result, chunk_texts) is False
