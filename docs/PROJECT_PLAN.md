# Project Plan

Technical milestones for this repo.

## Milestones

1. **Ingestion** — Load `data/docs/`, chunk with stable IDs, write JSONL + stats.
2. **Retrieval** — Embeddings index, top-k retrieval with source metadata.
3. **Generation** — Answers grounded in retrieved chunks; citations required; insufficient-evidence path.
4. **Evaluation** — Fixed question set (`data/eval/questions.jsonl`), automated metrics + `python -m src.app.main eval` → `results/baseline.json` (and related proxies: retrieval hit rate, groundedness proxy, citation rate, OOD decline checks).
5. **Iteration** — One documented tuning pass with before/after metrics in `RESULTS.md`.

## Rules of thumb

- Prefer small, reviewable changes.
- Every answer path should cite sources or decline explicitly.
- Record baselines before “improvements.”
