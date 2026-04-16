# Project Rules: Grounded Knowledge Engine
**Lead PM:** Technical Product Manager (Platform & AI)

## 1. Product Mission
To build a production-grade RAG system where accuracy and traceability are the core features. This is a high-caliber implementation focused on eliminating hallucinations in enterprise-grade knowledge retrieval.

## 2. Technical Bar (Definition of Done)
* **Full Traceability:** Every response must be verifiable. Each answer must map to a specific `doc_id` and character offset in the source.
* **Groundedness over "Vibes":** If the context is insufficient, the system must explicitly state what is missing rather than guessing. 
* **Evaluation-Driven:** No changes to retrieval or prompting logic are accepted without a "Before vs. After" evaluation report against the Golden Set.
* **Production Hygiene:** Use `uv` for dependency management and maintain a modular `src/` directory structure.

## 3. Agent Instructions
* **Implementation Memos:** Before writing code, provide a brief memo on technical trade-offs (e.g., Accuracy vs. Latency).
* **Failure Analysis:** Proactively identify where a design might fail (e.g., edge cases in chunking or ambiguous queries).
* **No-Credential-Inputs:** Prohibit the creation of UI elements that handle or display secrets (API keys, passwords, base URLs) if they can be managed by the environment or backend. Traceability must never compromise security.