"""Evaluation harness — run all questions through the pipeline and grade.

Processes every question in ``data/eval/questions.jsonl``, runs the full
retrieve → generate pipeline, grades with metrics, and outputs
a scored report to ``results/baseline.json``.

Performance note: The embedding model is loaded ONCE and reused for
all 55 questions rather than loading per-question.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.evals.metrics import (
    citations_grounded_strict,
    has_citations,
    ood_declined,
    retrieval_hit_from_scores,
)
from src.generation.grounded_answer import generate_answer
from src.retrieval.index import query_index
from src.utils.io import read_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_QUESTIONS_PATH = Path("data/eval/questions.jsonl")
_DEFAULT_OUTPUT_PATH = Path("results/baseline.json")
_DEFAULT_INDEX_PATH = Path("data/processed/retrieval_index.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_chunk_texts(index_path: Path) -> dict[str, str]:
    """Load chunk_id → text mapping from the retrieval index."""
    with index_path.open("r", encoding="utf-8") as f:
        index_data = json.load(f)
    return {c["chunk_id"]: c["text"] for c in index_data["chunks"]}


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------


def run_eval(
    questions_path: Path = _DEFAULT_QUESTIONS_PATH,
    output_path: Path = _DEFAULT_OUTPUT_PATH,
    index_path: Path = _DEFAULT_INDEX_PATH,
    mode: str = "retrieval",
    top_k: int = 3,
    min_score: float = 0.15,
) -> dict:
    """Run the full evaluation over the question set.

    Parameters
    ----------
    questions_path:
        Path to the eval questions JSONL.
    output_path:
        Where to write the results JSON.
    index_path:
        Path to the retrieval index.
    mode:
        Generation mode (``"retrieval"`` or ``"llm"``).
    top_k:
        Number of chunks to retrieve per question.
    min_score:
        Insufficient-evidence threshold.

    Returns
    -------
    dict
        The complete evaluation report.
    """
    questions = read_jsonl(questions_path)
    chunk_texts = _load_chunk_texts(index_path)
    total = len(questions)

    logger.info("=" * 70)
    logger.info(
        "Evaluation run: %d questions, mode=%s, top_k=%d, min_score=%.2f",
        total,
        mode,
        top_k,
        min_score,
    )
    logger.info("=" * 70)

    details: list[dict] = []

    # Counters for aggregate metrics
    in_domain_count = 0
    ood_count = 0
    retrieval_hits = 0
    ood_correct = 0
    cited_count = 0
    grounded_count = 0
    answered_count = 0  # in-domain questions that got an answer (not declined)

    # Tag-level tracking
    tag_stats: dict[str, dict[str, int]] = {}

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        is_ood = q.get("expect_insufficient_evidence", False)
        expected_doc_id = q.get("expected_doc_id")
        acceptable_doc_ids = q.get(
            "acceptable_doc_ids", [expected_doc_id] if expected_doc_id else []
        )
        tags = q.get("tags", [])

        logger.info("  [%d/%d] %s: %s", i, total, qid, question[:60])

        # --- Run the full pipeline ---
        result = generate_answer(
            question=question,
            mode=mode,
            top_k=top_k,
            min_score=min_score,
            index_path=index_path,
        )

        # --- Also get raw retrieval results for retrieval hit rate ---
        raw_retrieval = query_index(
            query=question,
            index_path=index_path,
            top_k=top_k,
        )
        retrieved_doc_ids = [r.doc_id for r in raw_retrieval]
        top_score = raw_retrieval[0].score if raw_retrieval else 0.0

        # --- Grade ---
        record: dict = {
            "id": qid,
            "question": question,
            "tags": tags,
            "is_ood": is_ood,
            "top_score": round(top_score, 4),
            "retrieved_doc_ids": list(dict.fromkeys(retrieved_doc_ids)),  # unique, ordered
            "insufficient_evidence": result.insufficient_evidence,
        }

        if is_ood:
            ood_count += 1
            declined = ood_declined(result)
            record["ood_declined"] = declined
            if declined:
                ood_correct += 1
        else:
            in_domain_count += 1

            # Retrieval hit (independent of whether answer was generated)
            hit = retrieval_hit_from_scores(retrieved_doc_ids, acceptable_doc_ids)
            record["expected_doc_id"] = expected_doc_id
            record["acceptable_doc_ids"] = acceptable_doc_ids
            record["retrieval_hit"] = hit
            if hit:
                retrieval_hits += 1

            # Citation and groundedness (only if not declined)
            if not result.insufficient_evidence:
                answered_count += 1

                cited = has_citations(result)
                record["has_citations"] = cited
                if cited:
                    cited_count += 1

                grounded = citations_grounded_strict(result, chunk_texts)
                record["grounded"] = grounded
                if grounded:
                    grounded_count += 1
            else:
                record["has_citations"] = False
                record["grounded"] = None
                record["note"] = "Declined (insufficient evidence)"

            # Tag-level tracking
            for tag in tags:
                if tag not in tag_stats:
                    tag_stats[tag] = {"total": 0, "retrieval_hits": 0}
                tag_stats[tag]["total"] += 1
                if hit:
                    tag_stats[tag]["retrieval_hits"] += 1

        details.append(record)

    # --- Aggregate metrics ---
    aggregate = {
        "retrieval_hit_rate": round(retrieval_hits / in_domain_count, 4)
        if in_domain_count > 0
        else 0.0,
        "ood_decline_rate": round(ood_correct / ood_count, 4) if ood_count > 0 else 0.0,
        "citation_rate": round(cited_count / answered_count, 4) if answered_count > 0 else 0.0,
        "groundedness_proxy": round(grounded_count / answered_count, 4)
        if answered_count > 0
        else 0.0,
        "total_questions": total,
        "in_domain": in_domain_count,
        "in_domain_answered": answered_count,
        "in_domain_declined": in_domain_count - answered_count,
        "out_of_domain": ood_count,
    }

    # Tag-level breakdown
    by_tag = {}
    for tag, stats in sorted(tag_stats.items()):
        by_tag[tag] = {
            "retrieval_hit_rate": round(stats["retrieval_hits"] / stats["total"], 4)
            if stats["total"] > 0
            else 0.0,
            "count": stats["total"],
        }

    report = {
        "run_id": f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
        "config": {
            "mode": mode,
            "top_k": top_k,
            "min_score": min_score,
            "questions_path": str(questions_path),
            "index_path": str(index_path),
        },
        "aggregate": aggregate,
        "by_tag": by_tag,
        "details": details,
    }

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # --- Log summary ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info("")
    logger.info(
        "  Retrieval hit rate:    %d/%d  (%.0f%%)",
        retrieval_hits,
        in_domain_count,
        aggregate["retrieval_hit_rate"] * 100,
    )
    logger.info(
        "  OOD decline rate:      %d/%d  (%.0f%%)",
        ood_correct,
        ood_count,
        aggregate["ood_decline_rate"] * 100,
    )
    logger.info(
        "  Citation rate:         %d/%d  (%.0f%%)",
        cited_count,
        answered_count,
        aggregate["citation_rate"] * 100,
    )
    logger.info(
        "  Groundedness proxy:    %d/%d  (%.0f%%)",
        grounded_count,
        answered_count,
        aggregate["groundedness_proxy"] * 100,
    )
    logger.info("")

    # Log failures
    failures = [d for d in details if not d.get("retrieval_hit", True) and not d.get("is_ood")]
    if failures:
        logger.info("  Retrieval misses:")
        for f in failures:
            logger.info(
                "    ✗ %s: expected=%s, got=%s (%s)",
                f["id"],
                f.get("acceptable_doc_ids", [f["expected_doc_id"]]),
                f["retrieved_doc_ids"][:2],
                f["question"][:50],
            )
        logger.info("")

    logger.info("  Tag breakdown:")
    for tag, stats in by_tag.items():
        logger.info(
            "    %-20s  %d/%d  (%.0f%%)",
            tag,
            int(stats["retrieval_hit_rate"] * stats["count"]),
            stats["count"],
            stats["retrieval_hit_rate"] * 100,
        )

    logger.info("")
    logger.info("  Output: %s", output_path)
    logger.info("=" * 70)

    return report
