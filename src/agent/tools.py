"""Tool registry for the Agentic RAG loop.

Defines the three tools the agent can call, along with their OpenAI-compatible
JSON schemas so the LLM knows when and how to invoke them.

Tools
-----
search_knowledge_base
    Runs a cosine-similarity retrieval pass against the local vector index.
    The agent may call this multiple times with different queries.
refine_query
    Asks the LLM to rephrase the original question to improve retrieval
    (query expansion / decomposition).  The result is fed back as a new
    ``search_knowledge_base`` call.
finalize_answer
    Signals that the agent has gathered sufficient evidence.  Triggers the
    existing citation-rehydration pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.index import RetrievalResult, query_index

# ---------------------------------------------------------------------------
# OpenAI-compatible tool schemas
# Passed verbatim to client.chat.completions.create(tools=TOOL_SCHEMAS)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the local knowledge base for chunks relevant to a query. "
                "Returns ranked text snippets with their chunk IDs. "
                "Call this when you need information to answer the user's question. "
                "You may call it multiple times with different query phrasings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve (default: 3, max: 5).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refine_query",
            "description": (
                "Rephrase or decompose the original user question into a better "
                "search query. Use this when the initial search returned low-quality "
                "or irrelevant results. Returns an improved query string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "original_query": {
                        "type": "string",
                        "description": "The original user question.",
                    },
                    "feedback": {
                        "type": "string",
                        "description": (
                            "Brief explanation of why the previous search was "
                            "insufficient and what angle to try instead."
                        ),
                    },
                },
                "required": ["original_query", "feedback"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_answer",
            "description": (
                "Produce the final grounded answer once you have sufficient evidence "
                "from search results. You MUST cite every factual claim using "
                "[chunk_id] notation. If evidence is insufficient, set answer to "
                "INSUFFICIENT_EVIDENCE: <brief description of what is missing>."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "The final answer with [chunk_id] citations, OR "
                            "INSUFFICIENT_EVIDENCE: <reason> if evidence is lacking."
                        ),
                    },
                    "cited_chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of chunk_ids referenced in the answer.",
                    },
                },
                "required": ["answer", "cited_chunk_ids"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution functions
# ---------------------------------------------------------------------------


def execute_search(
    query: str,
    top_k: int = 3,
    index_path: Path = Path("data/processed/retrieval_index.json"),
    min_score: float = 0.0,
) -> list[RetrievalResult]:
    """Execute a knowledge-base search and return ranked RetrievalResult objects."""
    top_k = min(top_k, 5)  # hard cap to prevent runaway token usage
    return query_index(
        query=query,
        index_path=index_path,
        top_k=top_k,
        min_score=min_score,
    )


def format_results_for_llm(results: list[RetrievalResult]) -> str:
    """Format retrieval results into a string the LLM can reason about."""
    if not results:
        return "No relevant results found."
    lines = []
    for r in results:
        lines.append(f"[{r.chunk_id}] (score={r.score:.3f}, source={r.source_path})\n{r.text}")
    return "\n\n---\n\n".join(lines)
