# Project Brief — Grounded RAG Evaluator
*Retrospectively written after project completion as a reference example.
This is what should have been authored BEFORE the first line of code.*

---

## 1. Product Vision

A grounded Q&A system for internal policy and support documentation.
It answers user questions with cited, verifiable responses and explicitly
declines when the knowledge base contains insufficient evidence.
The target user is an HR, compliance, or support team member who needs
trustworthy answers without hallucination risk.

---

## 2. Definition of Done (Metrics)

| Metric | Target | How Measured |
|--------|--------|--------------|
| Retrieval Hit Rate | > 80% | % of golden questions where the correct `doc_id` is in top-3 retrieved chunks |
| OOD Decline Rate | > 95% | % of out-of-domain questions that return `INSUFFICIENT_EVIDENCE` |
| Citation Accuracy | 100% | Every answer in LLM mode must include at least one `[chunk_id]` tag |
| No Hallucinated Paths | 100% | Cited `source_path` must always point to a real file with correct char offsets |
| Test Coverage | All core modules | `test_retrieval.py`, `test_generation.py`, `test_evals.py`, `test_contract.py` all pass |

---

## 3. What Must Never Go Wrong

- [ ] A question completely outside the knowledge base must NEVER receive a fabricated answer. It must trigger `INSUFFICIENT_EVIDENCE`.
- [ ] Every cited source must map to a real document with correct character offsets. No hallucinated file paths.
- [ ] API keys must never appear in the UI, the code, or git history.
- [ ] Switching from OpenAI to Groq or a local model must not require code changes — only `.env` changes.
- [ ] A change to retrieval or prompting logic must always be accompanied by before/after eval metrics. No "it feels better" changes.
- [ ] A "hybrid" LLM response (partial answer + INSUFFICIENT_EVIDENCE flag buried at the end) must be treated as a decline — not an answer.

---

## 4. Eval Set

**55 cases across 4 categories:**

| Category | Count | Example |
|----------|-------|---------|
| FAQ (account/access) | 10 | "How do I reset my access?" |
| Policy (usage/content) | 10 | "Is sharing login credentials allowed?" |
| Privacy / billing / troubleshooting | 20 | "What encryption is used for data at rest?" |
| Out-of-domain (OOD) | 5 | "What is the capital of France?" |

**For v2:**
- Ambiguous questions that span two documents simultaneously
- Questions with deliberate typos or poor phrasing
- Questions where the answer is implied but not stated verbatim
- Adversarial phrasings designed to trick the retriever

**File:** `data/eval/questions.jsonl`

---

## 5. Prioritized Feature Roadmap

| Phase | Feature | Why This Phase | Success Criteria |
|-------|---------|----------------|-----------------|
| 1 | Ingestion pipeline | No pipeline = no data | Chunks saved to `data/processed/chunks.jsonl` with stable IDs |
| 2 | Retrieval index | Core capability | Top-k query returns correct doc for 80%+ of golden questions |
| 3 | Extractive generation | Zero-hallucination baseline | Every answer word comes verbatim from a source chunk |
| 4 | Insufficient-evidence gate | Safety non-negotiable | OOD questions decline at > 95% rate |
| 5 | LLM generation mode | User experience | Synthesized answers with `[chunk_id]` citations |
| 6 | Evaluation harness | Prove the metrics | Reproducible before/after reports from `run_eval.py` |
| 7 | Streamlit UI | Portfolio demo | Clean interface exposing retrieval, LLM, and results |

**Phase 2:**
- Agentic loop with tool calling 
- Query refinement tool