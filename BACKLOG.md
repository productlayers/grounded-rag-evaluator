# Product Backlog: Grounded RAG Evaluator

This document tracks the planned features, architectural improvements, and optimizations for the Grounded RAG system.

## Roadmap Overview

| Sl. No. | Feature Type | Feature | Status |
|---------|--------------|---------|--------|
| 1 | Quality Gate | Agent Quality Gate (Deterministic `min_score` for agent mode) | Planned |
| 2 | Architecture / Safety | Force First Tool Call Deterministically (Hybrid Agent Pattern) | Planned |
| 3 | CI/CD & Evaluation | Rate Limit Management for Automated Eval Suite | Planned |
| 4 | Optimization / Cost | Local Cross-Encoder Re-ranking (Token Optimization) | Planned |
| 5 | Resilience / Cost | Groq Compound Model Fallback Strategy | Planned |

---

## Detailed Feature Specifications

### 1. Agent Quality Gate
* **Feature Type:** Quality Gate
* **Gap:** The agent currently relies on the LLM's *subjective* judgment to decide if search results are "good enough" before finalizing. If the LLM thinks a `score=0.12` chunk is sufficient, it will finalize with weak evidence.
* **Planned Fix:** Add a deterministic Python check inside `loop.py` after each `search_knowledge_base` tool call. If `results[0].score < AGENT_MIN_SCORE`, inject a warning message into the tool result (e.g., *"Low confidence results. Consider refining your query."*) to programmatically nudge the agent toward a refinement step.

### 2. Force First Tool Call Deterministically
* **Feature Type:** Architecture / Safety
* **Gap:** The system prompt tells the agent "always start with `search_knowledge_base`," but this is a guideline, not enforced code. A misbehaving LLM could theoretically call `finalize_answer` on turn 1 without ever searching, producing an unsupported answer.
* **Planned Fix:** On the first iteration of the loop only, set `tool_choice={"type": "function", "function": {"name": "search_knowledge_base"}}` in the LLM call. This forces Python to own the first step deterministically. From iteration 2 onwards, revert to `tool_choice="auto"` to restore the agent's autonomous decision-making.

### 3. Rate Limit Management for Automated Eval Suite
* **Feature Type:** CI/CD & Evaluation
* **Gap:** Running the 55-question Ragas eval concurrently hits Groq's free-tier `Tokens Per Day` (TPD) limits, causing the script to crash with `429 Too Many Requests` errors.
* **Planned Fix:** Implement test-batching logic for local runs, or provision a dedicated higher-tier API key (e.g., OpenAI or Groq paid tier) specifically for CI/CD runs. Explore Ragas configuration to limit parallel workers if batching is not sufficient.

### 4. Local Cross-Encoder Re-ranking
* **Feature Type:** Optimization / Cost
* **Gap:** The current bi-encoder retrieval strategy relies on sending a large volume of chunks (`top_k=3`) to the LLM to guarantee high recall, resulting in severe "Context Window Bloat" and massive token costs during evaluation.
* **Planned Fix:** Integrate a local, CPU-based Cross-Encoder (via `sentence-transformers`) to act as a secondary precision filter. This will allow the pipeline to retrieve ~20 chunks via fast vector math, re-rank them locally for deep relevance, and pass only the top 1 chunk to the LLM.

### 5. Groq Compound Model Fallback Strategy
* **Feature Type:** Resilience / Cost
* **Gap:** The evaluation pipeline entirely halts if the primary heavy model (e.g., `llama-3.3-70b-versatile`) exhausts its daily token quota, creating a rigid single point of failure.
* **Planned Fix:** Implement a programmatic "Compound Model" fallback chain in the orchestrator/eval script. If the primary 70B model throws a `429 Too Many Requests` error, automatically catch the exception and immediately retry the request using a smaller, higher-limit model (e.g., `llama3-8b-8192` or `gemma2-9b-it`).
* **PM Value:** Drastically increases the robustness of the system. Allows the team to stretch the free-tier API limits significantly further by utilizing smaller models when the primary is exhausted, ensuring the CI/CD pipeline doesn't block developers.
