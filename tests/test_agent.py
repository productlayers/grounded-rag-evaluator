"""Tests for the Agentic RAG loop.

These tests mock the LLM client so they run entirely offline with no
API keys required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.loop import run_agent_loop
from src.agent.tools import format_results_for_llm
from src.retrieval.index import RetrievalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CHUNK = RetrievalResult(
    chunk_id="support/faq::chunk_001",
    doc_id="support/faq",
    score=0.85,
    text="To reset your password, navigate to Settings > Security.",
    section_heading="Password Reset",
    source_path="data/docs/support.md",
    char_start=100,
    char_end=200,
)


def _make_tool_call(name: str, arguments: str) -> MagicMock:
    """Create a mock OpenAI tool call object."""
    tc = MagicMock()
    tc.id = f"call_{name}"
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_assistant_msg(tool_calls: list) -> MagicMock:
    msg = MagicMock()
    msg.tool_calls = tool_calls
    msg.content = None
    return msg


def _make_response(msg: MagicMock) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = msg
    return resp


# ---------------------------------------------------------------------------
# Tool unit tests
# ---------------------------------------------------------------------------


class TestFormatResultsForLLM:
    def test_empty_returns_no_results_message(self) -> None:
        assert "No relevant results" in format_results_for_llm([])

    def test_single_result_includes_chunk_id(self) -> None:
        output = format_results_for_llm([_FAKE_CHUNK])
        assert "support/faq::chunk_001" in output
        assert "reset your password" in output

    def test_multiple_results_separated_by_divider(self) -> None:
        output = format_results_for_llm([_FAKE_CHUNK, _FAKE_CHUNK])
        assert "---" in output


# ---------------------------------------------------------------------------
# Agentic loop tests
# ---------------------------------------------------------------------------


class TestAgentLoop:
    @patch("src.agent.loop._get_client")
    @patch("src.agent.loop.execute_search")
    def test_single_turn_success(
        self, mock_search: MagicMock, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        """Agent finds evidence on the first search and finalizes."""
        mock_search.return_value = [_FAKE_CHUNK]

        import json

        # Turn 1: agent calls search_knowledge_base
        search_call = _make_tool_call(
            "search_knowledge_base",
            json.dumps({"query": "how to reset password", "top_k": 3}),
        )
        # Turn 2: agent finalizes
        finalize_call = _make_tool_call(
            "finalize_answer",
            json.dumps(
                {
                    "answer": "Navigate to Settings > Security [support/faq::chunk_001].",
                    "cited_chunk_ids": ["support/faq::chunk_001"],
                }
            ),
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_response(_make_assistant_msg([search_call])),
            _make_response(_make_assistant_msg([finalize_call])),
        ]
        mock_get_client.return_value = (mock_client, "test-model")

        index_path = tmp_path / "index.json"
        result = run_agent_loop(question="How do I reset my password?", index_path=index_path)

        assert not result.insufficient_evidence
        assert "support/faq::chunk_001" in result.cited_chunk_ids
        assert result.iterations == 2  # search + finalize
        assert len(result.trace) == 2

    @patch("src.agent.loop._get_client")
    @patch("src.agent.loop.execute_search")
    def test_insufficient_evidence_declared(
        self, mock_search: MagicMock, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        """Agent declares INSUFFICIENT_EVIDENCE when results are empty."""
        mock_search.return_value = []

        import json

        search_call = _make_tool_call(
            "search_knowledge_base",
            json.dumps({"query": "alien invasion protocol", "top_k": 3}),
        )
        finalize_call = _make_tool_call(
            "finalize_answer",
            json.dumps(
                {
                    "answer": "INSUFFICIENT_EVIDENCE: No relevant documents found.",
                    "cited_chunk_ids": [],
                }
            ),
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_response(_make_assistant_msg([search_call])),
            _make_response(_make_assistant_msg([finalize_call])),
        ]
        mock_get_client.return_value = (mock_client, "test-model")

        index_path = tmp_path / "index.json"
        result = run_agent_loop(
            question="What is the alien invasion protocol?", index_path=index_path
        )

        assert result.insufficient_evidence
        assert result.cited_chunk_ids == []

    @patch("src.agent.loop._get_client")
    @patch("src.agent.loop.execute_search")
    def test_max_iterations_guard(
        self, mock_search: MagicMock, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        """Agent never calls finalize_answer: hits max_iterations, returns INSUFFICIENT_EVIDENCE."""
        mock_search.return_value = [_FAKE_CHUNK]

        import json

        # Agent always calls search, never finalize
        search_call = _make_tool_call(
            "search_knowledge_base",
            json.dumps({"query": "looping query", "top_k": 3}),
        )
        # Return same search call every time
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_response(
            _make_assistant_msg([search_call])
        )
        mock_get_client.return_value = (mock_client, "test-model")

        index_path = tmp_path / "index.json"
        result = run_agent_loop(
            question="Keep searching forever?",
            index_path=index_path,
            max_iterations=3,
        )

        assert result.insufficient_evidence
        assert result.iterations == 3  # hit the cap

    @patch("src.agent.loop._get_client")
    @patch("src.agent.loop.execute_search")
    def test_refine_query_step(
        self, mock_search: MagicMock, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        """Agent calls refine_query when first search is insufficient."""
        mock_search.return_value = [_FAKE_CHUNK]

        import json

        refine_call = _make_tool_call(
            "refine_query",
            json.dumps(
                {
                    "original_query": "reset",
                    "feedback": "Too vague, need to specify password reset",
                }
            ),
        )
        search_call = _make_tool_call(
            "search_knowledge_base",
            json.dumps({"query": "password reset steps", "top_k": 3}),
        )
        finalize_call = _make_tool_call(
            "finalize_answer",
            json.dumps(
                {
                    "answer": "Navigate to Settings [support/faq::chunk_001].",
                    "cited_chunk_ids": ["support/faq::chunk_001"],
                }
            ),
        )

        refine_llm_response = MagicMock()
        refine_llm_response.choices = [MagicMock()]
        refine_llm_response.choices[0].message.content = "password reset steps"

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_response(_make_assistant_msg([refine_call])),
            refine_llm_response,  # The inner refine_query LLM call
            _make_response(_make_assistant_msg([search_call])),
            _make_response(_make_assistant_msg([finalize_call])),
        ]
        mock_get_client.return_value = (mock_client, "test-model")

        index_path = tmp_path / "index.json"
        result = run_agent_loop(question="reset", index_path=index_path)

        assert not result.insufficient_evidence
        # Verify the trace recorded the refine step
        tool_names = [step.tool for step in result.trace]
        assert "refine_query" in tool_names
        assert "search_knowledge_base" in tool_names
        assert "finalize_answer" in tool_names
