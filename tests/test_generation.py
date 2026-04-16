"""Tests for the generation pipeline.

Tests cover:
- Sentence splitting
- Extractive generation produces citations
- Every citation maps to text from a real chunk
- Insufficient-evidence gate triggers on low scores
- GenerationResult has all required fields
"""

from __future__ import annotations

import os
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import numpy as np

from src.generation.grounded_answer import (
    generate_answer,
)
from src.generation.prompts import build_user_prompt, format_context
from src.retrieval.index import RetrievalResult

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


class TestPromptFormatting:
    def test_format_context_includes_chunk_ids(self) -> None:
        chunks = [
            {
                "chunk_id": "doc-a::chunk_000",
                "doc_id": "doc-a",
                "section_heading": "Overview",
                "text": "Test content here.",
            }
        ]
        context = format_context(chunks)
        assert "[doc-a::chunk_000]" in context
        assert "Test content here." in context
        assert "Overview" in context

    def test_build_user_prompt_has_question(self) -> None:
        chunks = [
            {
                "chunk_id": "doc-a::chunk_000",
                "doc_id": "doc-a",
                "section_heading": None,
                "text": "Content.",
            }
        ]
        prompt = build_user_prompt("What is this?", chunks)
        assert "What is this?" in prompt
        assert "Content." in prompt


# ---------------------------------------------------------------------------
# Extractive generation (mocked retrieval + embedder)
# ---------------------------------------------------------------------------


_MOCK_RESULTS = [
    RetrievalResult(
        chunk_id="support/faq::chunk_000",
        doc_id="support/faq",
        score=0.55,
        text=(
            "Use the account recovery flow from the sign-in page. "
            "You will need access to your registered email."
        ),
        section_heading="How do I reset my access?",
        source_path="data/docs/support/faq.md",
        char_start=0,
        char_end=100,
    ),
    RetrievalResult(
        chunk_id="support/account::chunk_001",
        doc_id="support/account",
        score=0.50,
        text=(
            "Navigate to Settings then Security. Click Reset Password and follow the instructions."
        ),
        section_heading="Password Reset",
        source_path="data/docs/support/account.md",
        char_start=100,
        char_end=200,
    ),
]


class _FakeEmbedder:
    """Deterministic embedder for testing."""

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
        vectors = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            for i, ch in enumerate(text[: self._dim * 2]):
                vec[i % self._dim] += ord(ch)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class TestExtractiveGeneration:
    def test_produces_answer_with_citations(self) -> None:
        with (
            patch("src.generation.grounded_answer.query_index", return_value=_MOCK_RESULTS),
            patch("src.retrieval.embedder.get_embedder", return_value=_FakeEmbedder()),
        ):
            result = generate_answer(
                question="How do I reset my password?",
                mode="retrieval",
                min_score=0.1,
            )

        assert not result.insufficient_evidence
        assert result.mode == "retrieval"
        assert len(result.answer) > 0
        assert len(result.citations) > 0

    def test_citations_reference_real_chunks(self) -> None:
        with (
            patch("src.generation.grounded_answer.query_index", return_value=_MOCK_RESULTS),
            patch("src.retrieval.embedder.get_embedder", return_value=_FakeEmbedder()),
        ):
            result = generate_answer(
                question="How do I reset my password?",
                mode="retrieval",
                min_score=0.1,
            )

        valid_chunk_ids = {r.chunk_id for r in _MOCK_RESULTS}
        for c in result.citations:
            assert c.chunk_id in valid_chunk_ids, (
                f"Citation {c.chunk_id} doesn't match any retrieved chunk"
            )

    def test_cited_text_appears_in_source_chunk(self) -> None:
        """Extractive answers must contain text verbatim from chunks."""
        with (
            patch("src.generation.grounded_answer.query_index", return_value=_MOCK_RESULTS),
            patch("src.retrieval.embedder.get_embedder", return_value=_FakeEmbedder()),
            patch(
                "src.generation.grounded_answer._heal_chunk_boundaries",
                side_effect=lambda r: r.text,
            ),
        ):
            result = generate_answer(
                question="How do I reset my password?",
                mode="retrieval",
                min_score=0.1,
            )

        # Every cited_text should appear in one of the source chunks
        all_chunk_text = " ".join(r.text for r in _MOCK_RESULTS)
        for c in result.citations:
            assert c.cited_text in all_chunk_text, (
                f"Cited text not found in source chunks: {c.cited_text[:50]}"
            )

    def test_generation_result_has_all_fields(self) -> None:
        with (
            patch("src.generation.grounded_answer.query_index", return_value=_MOCK_RESULTS),
            patch("src.retrieval.embedder.get_embedder", return_value=_FakeEmbedder()),
        ):
            result = generate_answer(
                question="test",
                mode="retrieval",
                min_score=0.1,
            )

        d = asdict(result)
        required = {
            "question",
            "answer",
            "citations",
            "insufficient_evidence",
            "retrieval_scores",
            "mode",
        }
        assert required <= d.keys()


class TestLLMGeneration:
    def test_llm_mode_calls_openai_and_parses_citations(self) -> None:
        """LLM mode should build a prompt, call OpenAI, and find citation IDs."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Wait, [support/faq::chunk_000] says to use recovery.")
            )
        ]

        with (
            patch("src.generation.grounded_answer.query_index", return_value=_MOCK_RESULTS),
            patch("src.retrieval.embedder.get_embedder", return_value=_FakeEmbedder()),
            patch("openai.OpenAI") as mock_openai_class,
        ):
            # Setup the mocked client
            mock_client = mock_openai_class.return_value
            mock_client.chat.completions.create.return_value = mock_response

            with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
                result = generate_answer(
                    question="How do I reset my password?",
                    mode="llm",
                    min_score=0.1,
                )

            assert result.mode == "llm"
            assert "recovery" in result.answer
            assert len(result.citations) == 1
            assert result.citations[0].chunk_id == "support/faq::chunk_000"

            # Verify the call was correct
            mock_client.chat.completions.create.assert_called_once()

    def test_no_api_key_with_custom_url(self) -> None:
        """Verify that a missing API key is permitted if a custom base_url is set."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Local answer"))]

        with (
            patch("src.generation.grounded_answer.query_index", return_value=_MOCK_RESULTS),
            patch("src.retrieval.embedder.get_embedder", return_value=_FakeEmbedder()),
            patch("openai.OpenAI") as mock_openai_class,
        ):
            mock_client = mock_openai_class.return_value
            mock_client.chat.completions.create.return_value = mock_response

            # No OPENAI_API_KEY, but has custom base_url
            with patch.dict(
                os.environ,
                {"OPENAI_BASE_URL": "http://localhost:11434/v1"},
                clear=True,
            ):
                # Ensure the patch.dict doesn't leak secrets or fail if key is empty
                if "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]

                result = generate_answer(
                    question="test",
                    mode="llm",
                    min_score=0.1,
                )

            assert result.answer == "Local answer"
            # Verify it initialized with placeholder key
            mock_openai_class.assert_called_once()
            assert mock_openai_class.call_args[1]["api_key"] == "not-required"

    def test_smart_default_groq(self) -> None:
        """Verify that detection of 'groq' in URL sets Llama default."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Groq answer"))]

        with (
            patch("src.generation.grounded_answer.query_index", return_value=_MOCK_RESULTS),
            patch("src.retrieval.embedder.get_embedder", return_value=_FakeEmbedder()),
            patch("openai.OpenAI") as mock_openai_class,
        ):
            mock_client = mock_openai_class.return_value
            mock_client.chat.completions.create.return_value = mock_response

            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "fake-groq-key",
                    "OPENAI_BASE_URL": "https://api.groq.com/openai/v1",
                },
                clear=True,
            ):
                # Remove OPENAI_MODEL to force the default logic
                if "OPENAI_MODEL" in os.environ:
                    del os.environ["OPENAI_MODEL"]

                generate_answer(question="test", mode="llm", min_score=0.1)

                # Verify Llama was sent to Groq
                args, kwargs = mock_client.chat.completions.create.call_args
                assert kwargs["model"] == "llama-3.3-70b-versatile"

    def test_hybrid_response_triggers_decline(self) -> None:
        """Verify that INSUFFICIENT_EVIDENCE is caught even if not at the start of the response."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=(
                        "To cancel a subscription, close your account. "
                        "INSUFFICIENT_EVIDENCE: specific steps are missing"
                    )
                )
            )
        ]

        with (
            patch("src.generation.grounded_answer.query_index", return_value=_MOCK_RESULTS),
            patch("src.retrieval.embedder.get_embedder", return_value=_FakeEmbedder()),
            patch("openai.OpenAI") as mock_openai_class,
        ):
            mock_client = mock_openai_class.return_value
            mock_client.chat.completions.create.return_value = mock_response

            with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
                result = generate_answer(question="test", mode="llm", min_score=0.1)

            assert result.insufficient_evidence is True
            assert len(result.citations) == 0
            assert "do not contain enough information" in result.answer


# ---------------------------------------------------------------------------
# Insufficient-evidence gate
# ---------------------------------------------------------------------------


class TestInsufficientEvidence:
    def test_low_scores_trigger_decline(self) -> None:
        """When top retrieval score < min_score, system should decline."""
        low_score_results = [
            RetrievalResult(
                chunk_id="doc::chunk_000",
                doc_id="doc",
                score=0.04,
                text="Irrelevant content.",
                section_heading=None,
                source_path="data/docs/doc.md",
                char_start=0,
                char_end=20,
            ),
        ]
        with patch(
            "src.generation.grounded_answer.query_index",
            return_value=low_score_results,
        ):
            result = generate_answer(
                question="What is the capital of France?",
                mode="retrieval",
                min_score=0.15,
            )

        assert result.insufficient_evidence is True
        assert "INSUFFICIENT_EVIDENCE" in result.answer
        assert len(result.citations) == 0

    def test_empty_retrieval_triggers_decline(self) -> None:
        """When retrieval returns nothing, system should decline."""
        with patch(
            "src.generation.grounded_answer.query_index",
            return_value=[],
        ):
            result = generate_answer(
                question="Random nonsense",
                mode="retrieval",
                min_score=0.15,
            )

        assert result.insufficient_evidence is True

    def test_high_scores_do_not_decline(self) -> None:
        """Above-threshold scores should produce a real answer."""
        with (
            patch("src.generation.grounded_answer.query_index", return_value=_MOCK_RESULTS),
            patch("src.retrieval.embedder.get_embedder", return_value=_FakeEmbedder()),
        ):
            result = generate_answer(
                question="How do I reset my password?",
                mode="retrieval",
                min_score=0.15,
            )

        assert result.insufficient_evidence is False
        assert "INSUFFICIENT_EVIDENCE" not in result.answer
