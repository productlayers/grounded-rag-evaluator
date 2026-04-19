"""Agentic RAG loop controller.

Replaces the single-shot LLM call with a multi-turn tool-calling loop.

The flow
--------
1. Initialise conversation with a system prompt + the user question.
2. Call the LLM with the tool schemas attached.
3. If the LLM returns a tool call:
   - ``search_knowledge_base`` → run retrieval, add results to conversation.
   - ``refine_query``          → rewrite the query, run retrieval, add results.
   - ``finalize_answer``       → extract answer + cited IDs, exit loop.
4. Repeat until ``finalize_answer`` is called or ``max_iterations`` is hit.
5. Return an ``AgentLoopResult`` that the caller converts into a
   ``GenerationResult`` (same contract as the rest of the pipeline).

Security / Groundedness guarantees
-----------------------------------
* The agent operates corpus-only: the only external calls are to the local
  vector index via ``search_knowledge_base``.
* ``finalize_answer`` enforces the same INSUFFICIENT_EVIDENCE gate used by
  the standard LLM mode.
* ``max_iterations`` prevents runaway API spend.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.agent.tools import (
    TOOL_SCHEMAS,
    execute_search,
    format_results_for_llm,
)
from src.retrieval.index import RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 4
_DEFAULT_INDEX_PATH = Path("data/processed/retrieval_index.json")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """A single step in the agent's reasoning trace."""

    tool: str
    args: dict[str, Any]
    result_summary: str  # human-readable summary for the UI trace


@dataclass
class AgentLoopResult:
    """Output from the agentic loop, before citation rehydration."""

    answer: str  # may be INSUFFICIENT_EVIDENCE: ...
    cited_chunk_ids: list[str]
    all_retrieved: list[RetrievalResult]  # union of all search calls
    trace: list[AgentStep] = field(default_factory=list)
    iterations: int = 0
    insufficient_evidence: bool = False


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = """\
You are a grounded Q&A agent with access to tools for searching a knowledge base.

Your mission: answer the user's question using ONLY information retrieved from the
knowledge base. You may search multiple times with different query phrasings to
gather evidence.

Rules:
1. Always start by calling search_knowledge_base with the user's question.
2. If results are weak or off-topic, call refine_query and then search again.
3. Every factual claim in your final answer MUST be cited with [chunk_id].
4. When you have sufficient evidence, call finalize_answer.
5. If after all searches evidence is still insufficient, call finalize_answer with:
   INSUFFICIENT_EVIDENCE: <brief description of what information is missing>
6. Do NOT use outside knowledge. Do NOT speculate.
"""


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


def _get_client() -> tuple[OpenAI, str]:
    """Return an OpenAI-compatible client and model name."""
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY") or ("local-placeholder" if base_url else None)
    client = OpenAI(api_key=api_key, base_url=base_url)

    if base_url and "groq" in base_url.lower():
        model = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")
    else:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    return client, model


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------


def run_agent_loop(
    question: str,
    index_path: Path = _DEFAULT_INDEX_PATH,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> AgentLoopResult:
    """Run the agentic RAG loop for a single user question.

    Parameters
    ----------
    question:
        The user's natural-language question.
    index_path:
        Path to the pre-built retrieval index.
    max_iterations:
        Hard cap on LLM tool-call rounds to prevent runaway spend.

    Returns
    -------
    AgentLoopResult
        Contains the final answer, cited chunk IDs, full retrieval history,
        and a human-readable trace of every tool call made.
    """
    client, model = _get_client()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    all_retrieved: list[RetrievalResult] = []
    trace: list[AgentStep] = []
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        logger.info("Agent iteration %d/%d", iterations, max_iterations)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1000,
        )

        msg = response.choices[0].message

        # If no tool call, treat as a plain-text finalisation (fallback)
        if not msg.tool_calls:
            logger.warning(
                "Agent returned plain text instead of a tool call — treating as finalize_answer."
            )
            return AgentLoopResult(
                answer=msg.content or "INSUFFICIENT_EVIDENCE: Agent did not call finalize_answer.",
                cited_chunk_ids=[],
                all_retrieved=all_retrieved,
                trace=trace,
                iterations=iterations,
                insufficient_evidence=True,
            )

        # Append the assistant's tool-call message to history
        messages.append(msg)

        # Process each tool call (typically one at a time)
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            logger.info("Agent called tool: %s(%s)", tool_name, args)

            # ----------------------------------------------------------------
            # Tool: search_knowledge_base
            # ----------------------------------------------------------------
            if tool_name == "search_knowledge_base":
                query = args["query"]
                top_k = int(args.get("top_k", 3))
                results = execute_search(query=query, top_k=top_k, index_path=index_path)
                all_retrieved.extend(results)

                formatted = format_results_for_llm(results)
                summary = f'Searched "{query}" → {len(results)} result(s)'
                trace.append(AgentStep(tool=tool_name, args=args, result_summary=summary))

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": formatted,
                    }
                )

            # ----------------------------------------------------------------
            # Tool: refine_query
            # ----------------------------------------------------------------
            elif tool_name == "refine_query":
                original = args["original_query"]
                feedback = args["feedback"]

                # Ask the LLM to rewrite the query (lightweight call, no tools)
                refine_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a search-query rewriter. Given an original "
                                "question and context about why it didn't retrieve "
                                "good results, produce a single improved search query. "
                                "Return ONLY the rewritten query, nothing else."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Original: {original}\nFeedback: {feedback}",
                        },
                    ],
                    temperature=0.0,
                    max_tokens=100,
                )
                refined = refine_response.choices[0].message.content or original
                summary = f'Refined query: "{original}" → "{refined}"'
                trace.append(AgentStep(tool=tool_name, args=args, result_summary=summary))

                # Feed the refined query back to the agent as a tool result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Refined query: {refined}",
                    }
                )

            # ----------------------------------------------------------------
            # Tool: finalize_answer  (exit condition)
            # ----------------------------------------------------------------
            elif tool_name == "finalize_answer":
                answer = args["answer"]
                cited_ids = args.get("cited_chunk_ids", [])
                is_insufficient = answer.strip().startswith("INSUFFICIENT_EVIDENCE")

                summary = (
                    "Finalised answer" if not is_insufficient else "Declared INSUFFICIENT_EVIDENCE"
                )
                trace.append(AgentStep(tool=tool_name, args=args, result_summary=summary))

                logger.info(
                    "Agent finalised after %d iteration(s). Insufficient=%s",
                    iterations,
                    is_insufficient,
                )
                return AgentLoopResult(
                    answer=answer,
                    cited_chunk_ids=cited_ids,
                    all_retrieved=all_retrieved,
                    trace=trace,
                    iterations=iterations,
                    insufficient_evidence=is_insufficient,
                )

            else:
                # Unknown tool — log and skip
                logger.warning("Agent called unknown tool: %s", tool_name)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: unknown tool '{tool_name}'",
                    }
                )

    # ---- max_iterations hit without finalize_answer -------------------------
    logger.warning("Agent hit max_iterations (%d) without finalising.", max_iterations)
    return AgentLoopResult(
        answer=(
            "INSUFFICIENT_EVIDENCE: Agent could not find a definitive answer "
            "within the allowed search budget."
        ),
        cited_chunk_ids=[],
        all_retrieved=all_retrieved,
        trace=trace,
        iterations=iterations,
        insufficient_evidence=True,
    )
