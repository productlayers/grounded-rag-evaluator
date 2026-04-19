"""Grounded answer generation with citations.

Two generation backends:

* **Extractive** (default): Selects the most relevant sentences from
  retrieved chunks using embedding similarity.  Zero cost, deterministic,
  every word in the answer comes verbatim from a source document.

* **OpenAI**: Sends retrieved chunks to GPT with a grounding prompt that
  mandates ``[chunk_id]`` citations and allows declining.  Requires
  ``OPENAI_API_KEY``.

Both share the same insufficient-evidence gate: if the top retrieval
score is below ``min_score``, the system declines with an explicit
message rather than guessing.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from src.retrieval.index import RetrievalResult, query_index

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_DEFAULT_MIN_SCORE = 0.15
_DEFAULT_TOP_K = 3
_DEFAULT_MAX_SENTENCES = 3


@dataclass(frozen=True)
class Citation:
    """A single citation linking part of an answer to a source chunk."""

    chunk_id: str
    doc_id: str
    source_path: str
    char_start: int
    char_end: int
    cited_text: str  # the sentence from the chunk used in the answer


@dataclass(frozen=True)
class GenerationResult:
    """Complete output from the generation pipeline."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    insufficient_evidence: bool = False
    retrieval_scores: list[float] = field(default_factory=list)
    mode: str = "retrieval"


def _heal_chunk_boundaries(r: RetrievalResult) -> str:
    """Expand chunk boundaries outwards to the nearest sentence ending or newline.

    This fixes the 'broken sentence' problem at retrieval time by looking up the
    original document and pulling the missing syntax, preserving mathematical
    precision while providing readable results.
    """
    try:
        with open(r.source_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return r.text  # fallback

    start = r.char_start
    end = r.char_end

    # Expand left until we hit a terminal punctuation mark or newline
    while start > 0:
        if content[start - 1] in ".!?" and content[start].isspace():
            break
        if content[start - 1] == "\n":
            break
        start -= 1

    # Trim any leading whitespace we backed into
    while start < len(content) and content[start].isspace():
        start += 1

    # Expand right until we finish the current sentence or hit a newline
    while end < len(content):
        if content[end - 1] in ".!?" and content[end].isspace():
            break
        if content[end] == "\n":
            break
        end += 1

    return content[start:end].strip()


# ---------------------------------------------------------------------------
# Retrieval generation
# ---------------------------------------------------------------------------


def _generate_retrieval(
    question: str,
    results: list[RetrievalResult],
    max_sentences: int = _DEFAULT_MAX_SENTENCES,
) -> GenerationResult:
    """Return the raw retrieved chunks.

    Process:
    1. Take the top retrieved chunks directly.
    2. Build citations mapping to the full chunk text.
    3. Return a static answer string pointing to the citations.
    """
    citations: list[Citation] = []

    for r in results:
        citations.append(
            Citation(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                source_path=r.source_path,
                char_start=r.char_start,
                char_end=r.char_end,
                cited_text=_heal_chunk_boundaries(r),
            )
        )

    return GenerationResult(
        question=question,
        answer="Here are the most relevant excerpts from the knowledge base:",
        citations=citations,
        insufficient_evidence=False,
        retrieval_scores=[r.score for r in results],
        mode="retrieval",
    )


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------


def _generate_llm(
    question: str,
    results: list[RetrievalResult],
) -> GenerationResult:
    """Generate a grounded answer using the OpenAI Chat Completions API.

    Sends retrieved chunks as context with a system prompt that mandates
    ``[chunk_id]`` citations and allows declining with
    ``INSUFFICIENT_EVIDENCE``.
    """
    from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")

    # Local model support: if base_url is set (custom endpoint), API key is optional.
    # We substitute a dummy key if missing to satisfy the SDK's initialization.
    if base_url and not api_key:
        api_key = "not-required"
        logger.info("Custom base_url detected; using placeholder API key for local/custom model.")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set.  Either:\n"
            "  1. Set it in your .env file and export it,\n"
            "  2. Use a local runner (Ollama/LM Studio) and set OPENAI_BASE_URL, or\n"
            "  3. Use --mode retrieval (default)."
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is not installed.\nRun: uv sync --extra openai"
        ) from None

    # Smart Defaults:
    # 1. If Groq is detected, use Llama.
    # 2. Otherwise use GPT-4o-mini.
    # 3. Always allow OPENAI_MODEL env var to override.
    is_groq = base_url and "groq" in base_url.lower()
    default_model = "llama-3.3-70b-versatile" if is_groq else "gpt-4o-mini"
    model = os.environ.get("OPENAI_MODEL", default_model)

    # Build context from retrieval results
    chunk_dicts = [
        {
            "chunk_id": r.chunk_id,
            "doc_id": r.doc_id,
            "section_heading": r.section_heading,
            "text": r.text,
        }
        for r in results
    ]

    user_prompt = build_user_prompt(question, chunk_dicts)

    logger.info("Calling OpenAI %s with %d context chunks", model, len(results))

    base_url = os.environ.get("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # deterministic for reproducibility
        max_tokens=500,
    )

    raw_answer = response.choices[0].message.content or ""

    # Check if the model declined anywhere in the response
    if "INSUFFICIENT_EVIDENCE" in raw_answer:
        # Extract everything from the flag onwards, or just return the decline explicitly
        # This prevents the UI from showing a partial answer + the warning tag
        return GenerationResult(
            question=question,
            answer=(
                "INSUFFICIENT_EVIDENCE: The available documents do not "
                "contain enough information to complete this answer."
            ),
            citations=[],
            insufficient_evidence=True,
            retrieval_scores=[r.score for r in results],
            mode="llm",
        )

    # Extract cited chunk_ids from the response
    cited_ids = set(re.findall(r"\[([^\]]+::\w+_\d+)\]", raw_answer))

    # Build citation objects for referenced chunks
    citations: list[Citation] = []
    for r in results:
        if r.chunk_id in cited_ids:
            citations.append(
                Citation(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    source_path=r.source_path,
                    char_start=r.char_start,
                    char_end=r.char_end,
                    cited_text=r.text[:200],  # first 200 chars as summary
                )
            )

    return GenerationResult(
        question=question,
        answer=raw_answer.strip(),
        citations=citations,
        insufficient_evidence=False,
        retrieval_scores=[r.score for r in results],
        mode="llm",
    )


# ---------------------------------------------------------------------------
# Agent-mode generator
# ---------------------------------------------------------------------------


def _generate_agent(
    question: str,
    index_path: Path,
) -> GenerationResult:
    """Delegate to the agentic loop and rehydrate citations.

    The agentic loop returns ``AgentLoopResult`` which contains the raw answer
    and a flat list of chunk IDs the LLM cited.  This function rehydrates those
    IDs into full ``Citation`` objects using the pool of ``RetrievalResult``s
    accumulated across all tool-call iterations.
    """
    from src.agent.loop import run_agent_loop  # local import avoids circular deps

    loop_result = run_agent_loop(question=question, index_path=index_path)

    # Build a lookup map from all chunks retrieved across all agent iterations
    chunk_map: dict[str, "RetrievalResult"] = {r.chunk_id: r for r in loop_result.all_retrieved}

    if loop_result.insufficient_evidence:
        return GenerationResult(
            question=question,
            answer=loop_result.answer,
            citations=[],
            insufficient_evidence=True,
            retrieval_scores=[r.score for r in loop_result.all_retrieved],
            mode="agent",
        )

    # Rehydrate citations using the same pattern as LLM mode
    cited_ids = set(loop_result.cited_chunk_ids)
    # Also extract any [chunk_id] tags from the answer text (belt-and-suspenders)
    cited_ids |= set(re.findall(r"\[([^\]]+::\w+_\d+)\]", loop_result.answer))

    citations: list[Citation] = []
    for chunk_id, r in chunk_map.items():
        if chunk_id in cited_ids:
            citations.append(
                Citation(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    source_path=r.source_path,
                    char_start=r.char_start,
                    char_end=r.char_end,
                    cited_text=r.text[:200],
                )
            )

    return GenerationResult(
        question=question,
        answer=loop_result.answer.strip(),
        citations=citations,
        insufficient_evidence=False,
        retrieval_scores=[r.score for r in loop_result.all_retrieved],
        mode="agent",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    mode: str = "retrieval",
    top_k: int = _DEFAULT_TOP_K,
    min_score: float = _DEFAULT_MIN_SCORE,
    index_path: Path | None = None,
    max_sentences: int = _DEFAULT_MAX_SENTENCES,
) -> GenerationResult:
    """Generate a grounded answer for a question.

    This is the main entry point for the generation pipeline.  It:

    1. Retrieves top-k chunks via the retrieval index.
    2. Checks the insufficient-evidence gate (top score < min_score).
    3. Dispatches to the retrieval or LLM generator.
    4. Returns a ``GenerationResult`` with answer, citations, and metadata.

    Parameters
    ----------
    question:
        The user's question.
    mode:
        ``"retrieval"`` (default), ``"llm"``, or ``"agent"``.
    top_k:
        Number of chunks to retrieve.
    min_score:
        Minimum retrieval score to proceed.  Below this triggers
        the insufficient-evidence decline.
    index_path:
        Path to the retrieval index.  Uses default if not specified.
    max_sentences:
        (Extractive only) Maximum sentences in the answer.

    Returns
    -------
    GenerationResult
        Contains the answer, citations, and whether evidence was
        insufficient.
    """
    from src.retrieval.index import _DEFAULT_INDEX_PATH

    if index_path is None:
        index_path = _DEFAULT_INDEX_PATH

    # Step 1: Retrieve
    results = query_index(
        query=question,
        index_path=index_path,
        top_k=top_k,
        min_score=0.0,  # retrieve everything; we gate below
    )

    retrieval_scores = [r.score for r in results]

    # Step 2: Insufficient-evidence gate
    top_score = retrieval_scores[0] if retrieval_scores else 0.0

    if top_score < min_score:
        logger.info(
            "Insufficient evidence for %r (top_score=%.4f < min_score=%.4f)",
            question[:60],
            top_score,
            min_score,
        )
        return GenerationResult(
            question=question,
            answer=(
                f"INSUFFICIENT_EVIDENCE: The available documents do not contain "
                f"enough information to answer this question. "
                f"(top retrieval score: {top_score:.4f}, threshold: {min_score:.4f})"
            ),
            citations=[],
            insufficient_evidence=True,
            retrieval_scores=retrieval_scores,
            mode=mode,
        )

    # Step 3: Generate
    logger.info(
        "Generating %s answer for %r (top_score=%.4f, %d chunks)",
        mode,
        question[:60],
        top_score,
        len(results),
    )

    if mode == "retrieval":
        return _generate_retrieval(question, results, max_sentences)
    elif mode == "llm":
        return _generate_llm(question, results)
    elif mode == "agent":
        return _generate_agent(question, index_path)
    else:
        raise ValueError(f"Unknown generation mode: {mode!r}. Use 'retrieval', 'llm', or 'agent'.")
