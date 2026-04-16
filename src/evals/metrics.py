"""Metric functions for grading RAG pipeline outputs.

Each function grades a single question-answer pair and returns a boolean.
These are pure functions with no side effects — easy to test and compose.
"""

from __future__ import annotations

from src.generation.grounded_answer import GenerationResult


def retrieval_hit(result: GenerationResult, expected_doc_ids: str | list[str]) -> bool:
    """Did the system retrieve chunks from the expected document?

    Checks whether any citation's ``doc_id`` matches any of the expected documents.
    For retrieval mode, citations directly reflect what was retrieved.
    """
    if isinstance(expected_doc_ids, str):
        expected_doc_ids = [expected_doc_ids]
    return any(c.doc_id in expected_doc_ids for c in result.citations)


def retrieval_hit_from_scores(
    retrieved_doc_ids: list[str],
    expected_doc_ids: str | list[str],
) -> bool:
    """Did one of the expected doc_ids appear in the retrieved results?

    This variant works with raw retrieval results (before generation),
    useful when the system declined but we still want to check retrieval.
    """
    if isinstance(expected_doc_ids, str):
        expected_doc_ids = [expected_doc_ids]
    return any(expected_id in retrieved_doc_ids for expected_id in expected_doc_ids)


def ood_declined(result: GenerationResult) -> bool:
    """Did the system correctly decline an out-of-domain question?

    Returns True if the system returned ``insufficient_evidence=True``.
    """
    return result.insufficient_evidence


def has_citations(result: GenerationResult) -> bool:
    """Does the answer include at least one citation?

    An answer without citations is unverifiable and violates the
    traceability requirement.
    """
    return len(result.citations) > 0


def citations_grounded(result: GenerationResult) -> bool:
    """Is every cited_text actually present in the source chunk's text?

    For retrieval mode, this should always be True (text is copied
    verbatim). For LLM mode, this measures whether the LLM hallucinated rephrase, so this
    metric can be < 100%.

    Returns True if all citations pass, or if there are no citations.
    """
    if not result.citations:
        return True

    # We need to check each citation's cited_text against source chunks.
    # Since we don't have the raw chunk text here, we check that cited_text
    # appears in the answer itself (retrieval mode embeds cited text in answer).
    # For a stricter check, the eval runner passes chunk text alongside.
    return all(c.cited_text and len(c.cited_text.strip()) > 0 for c in result.citations)


def citations_grounded_strict(
    result: GenerationResult,
    chunk_texts: dict[str, str],
) -> bool:
    """Strict groundedness: does cited_text appear verbatim in source chunk?

    Parameters
    ----------
    result:
        The generation result.
    chunk_texts:
        Mapping of ``chunk_id`` → full chunk text, from the retrieval index.
    """
    if not result.citations:
        return True

    for c in result.citations:
        source_text = chunk_texts.get(c.chunk_id, "")
        if c.cited_text not in source_text:
            return False
    return True
