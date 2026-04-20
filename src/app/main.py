"""CLI entrypoint for the Grounded RAG Evaluator.

Usage::

    python -m src.app.main ingest [OPTIONS]
    python -m src.app.main retrieve-build [OPTIONS]
    python -m src.app.main retrieve-query --q "..." [OPTIONS]
    python -m src.app.main answer --q "..." [OPTIONS]
    python -m src.app.main eval [OPTIONS]

Run ``python -m src.app.main <command> --help`` for available options.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from src.ingestion.chunker import ChunkingConfig, chunk_document
from src.ingestion.loader import load_documents
from src.ingestion.writer import write_chunks

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DOCS_DIR = Path("data/docs")
_DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
_DEFAULT_INDEX_PATH = Path("data/processed/retrieval_index.json")
_DEFAULT_MAX_CHARS = 500
_DEFAULT_OVERLAP = 100
_DEFAULT_MIN_CHUNK_CHARS = 50
_DEFAULT_TOP_K = 3
_DEFAULT_MIN_SCORE = 0.0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _run_ingest(args: argparse.Namespace) -> None:
    """Execute the ingestion pipeline: load → chunk → write."""
    docs_dir = Path(args.docs_dir)
    output = Path(args.output)
    config = ChunkingConfig(
        max_chars=args.max_chars,
        overlap=args.overlap,
        min_chunk_chars=args.min_chunk_chars,
    )

    documents = load_documents(docs_dir)
    if not documents:
        logging.getLogger(__name__).warning("No documents found in %s", docs_dir)
        return

    all_chunks = []
    for doc in documents:
        chunks = chunk_document(
            text=doc.text,
            doc_id=doc.doc_id,
            source_path=doc.source_path,
            config=config,
        )
        all_chunks.extend(chunks)

    write_chunks(all_chunks, output)


def _run_retrieve_build(args: argparse.Namespace) -> None:
    """Build the retrieval index from chunks.jsonl."""
    from src.retrieval.index import build_index

    build_index(
        chunks_path=Path(args.chunks_path),
        index_path=Path(args.index_path),
        mode=args.mode,
    )


def _run_retrieve_query(args: argparse.Namespace) -> None:
    """Query the retrieval index and print results."""
    from src.retrieval.index import query_index

    results = query_index(
        query=args.q,
        index_path=Path(args.index_path),
        top_k=args.top_k,
        min_score=args.min_score,
    )

    log = logging.getLogger(__name__)

    if not results:
        log.info("No results above min_score=%.2f", args.min_score)
        return

    log.info("=" * 70)
    log.info("Query: %s", args.q)
    log.info("=" * 70)

    for i, r in enumerate(results, 1):
        log.info("")
        log.info("  #%d  score=%.4f  chunk=%s", i, r.score, r.chunk_id)
        log.info("       doc=%s  section=%s", r.doc_id, r.section_heading)
        log.info("       chars=%d–%d in %s", r.char_start, r.char_end, r.source_path)
        # Show first 150 chars of text
        preview = r.text[:150].replace("\n", " ")
        log.info("       text=%s...", preview)

    log.info("")
    log.info("=" * 70)

    # Also write JSON to stdout if requested
    if args.json:
        output = [asdict(r) for r in results]
        print(json.dumps(output, indent=2, ensure_ascii=False))


def _run_answer(args: argparse.Namespace) -> None:
    """Generate a grounded answer for a question."""
    from dataclasses import asdict

    from src.generation.grounded_answer import generate_answer

    result = generate_answer(
        question=args.q,
        mode=args.mode,
        top_k=args.top_k,
        min_score=args.min_score,
        max_sentences=args.max_sentences,
    )

    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("Question: %s", result.question)
    log.info("Mode:     %s", result.mode)
    log.info("=" * 70)

    if result.insufficient_evidence:
        log.info("")
        log.info("  ⚠ INSUFFICIENT EVIDENCE")
        log.info("  %s", result.answer)
        log.info("")
    else:
        if result.mode == "retrieval":
            log.info("")
            log.info("  %s", result.answer)
            for c in result.citations:
                log.info("")
                log.info("  --- Result (%s) ---", c.doc_id)
                for line in c.cited_text.splitlines():
                    log.info("  %s", line)
            log.info("")
        else:
            log.info("")
            log.info("  Answer:")
            log.info("  %s", result.answer)
            log.info("")
            log.info("  Citations (%d):", len(result.citations))
            for c in result.citations:
                log.info(
                    "    • %s  →  %s  chars %d–%d",
                    c.chunk_id,
                    c.doc_id,
                    c.char_start,
                    c.char_end,
                )
            log.info("")

    log.info("  Retrieval scores: %s", result.retrieval_scores)
    log.info("=" * 70)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        log.info("Output written to %s", output_path)


def _run_eval(args: argparse.Namespace) -> None:
    """Run the evaluation harness over the question set."""
    from src.evals.run_eval import run_eval

    run_eval(
        questions_path=Path(args.questions),
        output_path=Path(args.output),
        mode=args.mode,
        top_k=args.top_k,
        min_score=args.min_score,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grounded-rag",
        description="Grounded RAG Evaluator — production-grade RAG with citations",
    )
    sub = parser.add_subparsers(dest="command")

    # -- ingest -------------------------------------------------------------
    ingest = sub.add_parser("ingest", help="Ingest source documents into chunks.jsonl")
    ingest.add_argument(
        "--docs-dir",
        default=str(_DEFAULT_DOCS_DIR),
        help=f"Source documents directory (default: {_DEFAULT_DOCS_DIR})",
    )
    ingest.add_argument(
        "--output",
        default=str(_DEFAULT_CHUNKS_PATH),
        help=f"Output JSONL path (default: {_DEFAULT_CHUNKS_PATH})",
    )
    ingest.add_argument(
        "--max-chars",
        type=int,
        default=_DEFAULT_MAX_CHARS,
        help=f"Maximum characters per chunk (default: {_DEFAULT_MAX_CHARS})",
    )
    ingest.add_argument(
        "--overlap",
        type=int,
        default=_DEFAULT_OVERLAP,
        help=f"Overlap characters between chunks (default: {_DEFAULT_OVERLAP})",
    )
    ingest.add_argument(
        "--min-chunk-chars",
        type=int,
        default=_DEFAULT_MIN_CHUNK_CHARS,
        help=(
            "Minimum chunk size; smaller chunks merge into previous"
            f" (default: {_DEFAULT_MIN_CHUNK_CHARS})"
        ),
    )

    # -- retrieve-build ----------------------------------------------------
    rb = sub.add_parser("retrieve-build", help="Build retrieval index from chunks.jsonl")
    rb.add_argument(
        "--mode",
        choices=["local", "openai"],
        default="local",
        help="Embedding mode (default: local)",
    )
    rb.add_argument(
        "--chunks-path",
        default=str(_DEFAULT_CHUNKS_PATH),
        help=f"Input chunks JSONL (default: {_DEFAULT_CHUNKS_PATH})",
    )
    rb.add_argument(
        "--index-path",
        default=str(_DEFAULT_INDEX_PATH),
        help=f"Output index path (default: {_DEFAULT_INDEX_PATH})",
    )

    # -- retrieve-query ----------------------------------------------------
    rq = sub.add_parser("retrieve-query", help="Query the retrieval index")
    rq.add_argument(
        "--q",
        required=True,
        help="The query string",
    )
    rq.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help=f"Number of results to return (default: {_DEFAULT_TOP_K})",
    )
    rq.add_argument(
        "--min-score",
        type=float,
        default=_DEFAULT_MIN_SCORE,
        help=f"Minimum similarity score (default: {_DEFAULT_MIN_SCORE})",
    )
    rq.add_argument(
        "--index-path",
        default=str(_DEFAULT_INDEX_PATH),
        help=f"Index path (default: {_DEFAULT_INDEX_PATH})",
    )
    rq.add_argument(
        "--json",
        action="store_true",
        help="Also output results as JSON to stdout",
    )

    # -- answer ------------------------------------------------------------
    ans = sub.add_parser("answer", help="Generate a grounded answer with citations")
    ans.add_argument(
        "--q",
        required=True,
        help="The question to answer",
    )
    ans.add_argument(
        "--mode",
        choices=["retrieval", "llm", "agent"],
        default="retrieval",
        help="Generation mode (default: retrieval)",
    )
    ans.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help=f"Chunks to retrieve (default: {_DEFAULT_TOP_K})",
    )
    ans.add_argument(
        "--min-score",
        type=float,
        default=0.15,
        help="Min retrieval score; below triggers decline (default: 0.15)",
    )
    ans.add_argument(
        "--max-sentences",
        type=int,
        default=3,
        help="Max sentences in retrieval answer (default: 3)",
    )
    ans.add_argument(
        "--output",
        default=None,
        help="Write result JSON to this path",
    )

    # -- eval --------------------------------------------------------------
    ev = sub.add_parser("eval", help="Run evaluation harness over the question set")
    ev.add_argument(
        "--questions",
        default="data/eval/questions.jsonl",
        help="Path to eval questions JSONL (default: data/eval/questions.jsonl)",
    )
    ev.add_argument(
        "--output",
        default="results/baseline.json",
        help="Output path for results JSON (default: results/baseline.json)",
    )
    ev.add_argument(
        "--mode",
        choices=["retrieval", "llm", "agent"],
        default="retrieval",
        help="Generation mode (default: retrieval)",
    )
    ev.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help=f"Chunks to retrieve (default: {_DEFAULT_TOP_K})",
    )
    ev.add_argument(
        "--min-score",
        type=float,
        default=0.15,
        help="Min retrieval score (default: 0.15)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Parse args and dispatch to the appropriate command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    commands = {
        "ingest": _run_ingest,
        "retrieve-build": _run_retrieve_build,
        "retrieve-query": _run_retrieve_query,
        "answer": _run_answer,
        "eval": _run_eval,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
