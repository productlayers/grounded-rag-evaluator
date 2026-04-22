# Grounded RAG Engine

**A production-grade, metrics-driven Retrieval-Augmented Generation (RAG) system engineered to eliminate hallucinations.**

Most RAG prototypes depend entirely on brittle prompt engineering ("please don't hallucinate"). This engine takes a deterministic, systems-level approach: it enforces a strict mathematical minimum-confidence threshold on vector retrieval, and explicitly declines to answer (`INSUFFICIENT_EVIDENCE`) if the context isn't strong enough.

Built with an offline-first, provider-agnostic architecture and a built-in evaluation harness, this project represents the technical bar required for enterprise AI applications.

---

## 🎯 Product Highlights

1. **Deterministic Hallucination Gate:** Uses mathematical similarity thresholds (`min_score`) rather than LLM "vibes" to protect against off-domain queries preventing brand risk.
2. **Evaluation-Driven Iteration:** Includes a complete evaluation script (`run_eval.py`) that calculates quantitative metrics against a golden dataset (Hit Rate, OOD Decline Rate, Citation Rate).
3. **Enterprise Traceability:** Every LLM factual claim must be backed by a `[chunk_id]` citation, which maps directly back to the `char_start` / `char_end` offset of the original source document.
4. **Provider Agnostic & Secure:** Bridges the OpenAI SDK standard to work seamlessly across OpenAI, Groq (Llama-3 latency), or Local Offline Models (Ollama) without code changes. "Zero-Credential UI" policy ensures API keys never leak into the front end.

---

## 🛠 Status

- [x] Repo skeleton + project environment (`uv`)
- [x] Document ingestion + chunking → JSONL with character-offset traceability
- [x] Embeddings + retrieval (top-k, metadata)
- [x] Grounded answers with required citations + insufficient-evidence path
- [x] Eval set + automated metrics
- [x] Model-Agnostic bridging (OpenAI, Groq, Local Models)

---

## ⚙️ Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for fast dependency and Python version management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc  # or open a new terminal

# Install Python 3.12 (managed by uv, does not affect system Python)
uv python install 3.12

# Install project + dev dependencies
uv sync --dev
```

**Environment Variables**  
Create a `.env` file in the root directory. The application will pull credentials via the backend, keeping the UI secure.
```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1  # Optional: For using Groq/Llama
# OPENAI_MODEL=llama-3.3-70b-versatile  # Optional: The engine supports smart defaults
```

---

## 🚀 Run

### Launch Web UI

Spin up an interactive web interface using Streamlit to query the system and tweak hallucination thresholds in real-time.

```bash
uv run python -m streamlit run src/app/streamlit_app.py
```

### CLI Backend Usage

#### 1. Build retrieval index
Embed all chunks and build a similarity search index:
```bash
uv run python -m src.app.main retrieve-build
```

#### 2. Generate a grounded answer
Ask a question and get a citation-backed answer. The engine will default to your `.env` configuration.
```bash
uv run python -m src.app.main answer --q "How do I reset my password?" --mode llm
```

| Flag | Default | Description |
|------|---------|-------------|
| `--q` | *(required)* | The question to answer |
| `--mode` | `retrieval` | `retrieval` (verbatim from docs) or `llm` (Grounded GenAI) |
| `--top-k` | `3` | Chunks to retrieve |
| `--min-score` | `0.15` | Below this → insufficient evidence decline |

#### 3. Run evaluation harness
Grade the pipeline against the golden 55-question eval set:
```bash
uv run python -m src.app.main eval --mode llm
```

---

## 🧪 QA & Testing

The repo adheres to strict engineering standards. Run the master QA script to run formatting, linting, and 20+ unit/contract tests:

```bash
uv run qa
```

## 📂 Layout

| Path | Purpose |
|------|---------|
| `src/ingestion/` | Load docs, chunk with char offsets, write JSONL |
| `src/retrieval/` | Embeddings (sentence-transformers) + top-k retrieval |
| `src/generation/` | Grounded answers: extractive + OpenAPI logic with citations |
| `src/evals/` | Evaluation harness + metrics |
| `src/app/main.py` | CLI entrypoint |
| `data/eval/` | Fixed eval questions + labels |
| `tests/` | Unit and Contract tests to prevent mode-drift |
