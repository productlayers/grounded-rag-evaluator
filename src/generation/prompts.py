"""Prompt templates for LLM-based grounded generation.

These prompts instruct the LLM to answer using ONLY the provided context
chunks, cite sources with ``[chunk_id]`` notation, and explicitly decline
when evidence is insufficient.

The prompts are intentionally kept as constants (not auto-generated)
so they can be reviewed, versioned, and tuned like any other configuration.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt — sets the model's behaviour
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a grounded Q&A assistant. You answer questions using ONLY the \
provided context chunks.

Rules:
1. Every factual claim MUST cite its source using [chunk_id] format.
2. If the context does not contain enough information to answer the \
question, respond ONLY with EXACTLY this format:
   INSUFFICIENT_EVIDENCE: <brief description of what information is missing>
   Do not provide partial or related answers.
3. Do NOT use any outside knowledge. Do NOT guess or speculate.
4. Keep answers concise, direct, and professional.
5. If multiple chunks support the same point, cite all of them.
"""

# ---------------------------------------------------------------------------
# User prompt — formats the context and question
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """\
Context chunks:

{context}

---

Question: {question}

Answer (cite every factual claim with [chunk_id]):"""

# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

_CHUNK_TEMPLATE = """\
[{chunk_id}] (from: {doc_id}, section: {section_heading})
{text}
"""


def format_context(chunks: list[dict[str, str | None]]) -> str:
    """Format retrieval results as numbered context for the LLM prompt.

    Parameters
    ----------
    chunks:
        List of dicts with keys: ``chunk_id``, ``doc_id``,
        ``section_heading``, ``text``.

    Returns
    -------
    str
        Formatted context block ready to insert into the user prompt.
    """
    parts = []
    for c in chunks:
        parts.append(
            _CHUNK_TEMPLATE.format(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                section_heading=c.get("section_heading") or "N/A",
                text=c["text"],
            )
        )
    return "\n".join(parts)


def build_user_prompt(question: str, chunks: list[dict[str, str | None]]) -> str:
    """Build the complete user prompt with formatted context.

    Parameters
    ----------
    question:
        The user's question.
    chunks:
        Retrieved context chunks.

    Returns
    -------
    str
        Complete user prompt ready to send to the LLM.
    """
    context = format_context(chunks)
    return USER_PROMPT_TEMPLATE.format(context=context, question=question)
