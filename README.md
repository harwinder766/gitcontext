# GitContext

<p align="center">
  <strong>Zero-RAM Codebase Intelligence & Developer Copilot powered by Hybrid RAG</strong>
</p>

<p align="center">
  <a href="https://gitcontext-xw5w.onrender.com/docs"><img src="https://img.shields.io/badge/API_Docs-Swagger_UI-blue?style=flat-square&logo=fastapi" alt="Swagger UI"></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Agent-LangGraph-FF4B4B?style=flat-square" alt="LangGraph">
  <img src="https://img.shields.io/badge/Vector_DB-Supabase_PGVector-3ECF8E?style=flat-square&logo=supabase&logoColor=white" alt="Supabase PGVector">
  <img src="https://img.shields.io/badge/Embeddings_%26_Rerank-Jina_AI-121212?style=flat-square" alt="Jina AI">
  <img src="https://img.shields.io/badge/LLM_Gateway-LiteLLM-4B32C3?style=flat-square" alt="LiteLLM">
  <img src="https://img.shields.io/badge/Observability-Pydantic_Logfire-E92063?style=flat-square" alt="Logfire">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License: MIT">
</p>

---

## 🌐 Live Demos

- **Backend API Docs (Swagger UI):** [`https://gitcontext-xw5w.onrender.com/docs`](https://gitcontext-xw5w.onrender.com/docs)
- **Frontend Web UI (Streamlit):** [`https://<your-streamlit-app-url>.streamlit.app`](https://<your-streamlit-app-url>.streamlit.app) *(or run locally via `streamlit run ui/home.py`)*

---

## ⚡ Core Value Proposition: The "Zero-RAM" Architecture

Most open-source RAG architectures suffer from prohibitive memory requirements: loading local PyTorch dependencies (`torch`, `sentence-transformers`, `transformers`, `onnxruntime`, and local cross-encoder models) immediately consumes **2.5 GB to 3.5 GB of RAM** upon application boot. This makes hosting on budget or free-tier cloud platforms (Render, Koyeb, Railway) impossible due to strict **512 MB RAM limits**.

**GitContext is architected from the ground up for a Zero-RAM footprint (~120 MB to 180 MB runtime RAM at peak load):**

1. **Zero-RAM Dense Embeddings:** Replaces heavy local PyTorch embedding models with the remote **Jina AI Cloud Embeddings API** (`jina-embeddings-v3`, 384 dimensions) and optional **Google Gemini API** (`models/text-embedding-004`). Vectors are natively truncated to 384 dimensions to remain 100% compatible with existing Supabase pgvector tables.
2. **Zero-RAM Cross-Encoder Reranking:** Eliminates the ~650 MB local `ms-marco-MiniLM-L-6-v2` cross-encoder in favor of the **Jina AI Reranker API** (`jina-reranker-v2-base-multilingual`), providing state-of-the-art cross-attentive relevance scoring over HTTP without loading any weights into container memory.
3. **Zero-RAM Deterministic Guardrails:** Replaces 1 GB+ DeBERTa-based ONNX guardrail scanners with sub-millisecond deterministic regex and heuristic filters. It strips invisible zero-width Unicode characters and catches jailbreak injection patterns while intentionally preserving legitimate developer code terminology (`api_key`, `openai`, `token`, `secret`, `create_llm`).
4. **Resilient LLM Routing & In-Memory Caching:** Uses **LiteLLM Router** with automatic multi-tier fallback between **Groq** (`llama-3.3-70b-versatile`) and **Google Gemini** (`gemini-1.5-flash-latest`), backed by an in-memory TTL response cache (`LITELLM_CACHE_TYPE=local`, TTL 3600s) to eliminate redundant API calls and prevent 429 rate limit errors.

---

## 🔍 Hybrid Retrieval System

Codebase search requires both **conceptual understanding** (e.g., *"where is the database connection pool defined?"*) and **exact lexical precision** (e.g., *"find the `create_embedding_model` function definition"*). GitContext addresses this through a high-performance Hybrid Search pipeline:

```
                          ┌─────────────────────────────────────┐
                          │         Incoming User Query         │
                          └──────────────────┬──────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
        ┌───────────────────────────┐                 ┌───────────────────────────┐
        │   Dense Vector Search     │                 │   Sparse Lexical Search   │
        │   (Supabase pgvector)     │                 │   (In-Memory rank-bm25)   │
        │   Semantic Context Match  │                 │   Exact Symbol / File Match│
        └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Reciprocal Rank Fusion   │
                               │          (RRF k=60)       │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │    Jina Cross-Encoder     │
                               │         Reranker          │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   Top-K Reranked Chunks   │
                               │     (with Citations)      │
                               └───────────────────────────┘
```

### 1. Dense Vector Search (Supabase PGVector)
Connects directly to remote Postgres via `langchain-postgres` and `psycopg` session pooling. Chunks are queried using cosine similarity over 384-dimensional dense representations.

### 2. Sparse Lexical Search (rank-bm25)
Maintains an in-memory BM25 index over codebase tokens, ensuring instant exact-string, class name, error message, and identifier retrieval that dense search can miss.

### 3. Startup Index Caching (Zero Query Delay)
In standard implementations, loading documents from a remote database on every request introduces significant network delays. GitContext pre-fetches and indexes document chunks into BM25 **once during FastAPI lifespan startup**. The initialized retriever is cached in `app.state.retriever` and only refreshed incrementally when a new repository is indexed (`POST /index`), bringing query retrieval times down from **~89 seconds to milliseconds**.

### 4. Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking
Combines ranked candidates from BM25 and pgvector using reciprocal rank scoring ($RRF_{score} = \sum \frac{1}{k + rank}$, $k=60$). The merged candidate pool is passed to Jina Reranker v2 to re-score query-document pairs and return the most relevant documents.

---

## 📦 Multi-Source Git Ingestion Pipeline

GitContext indexes three layers of developer context:

1. **Repository Source Code Tree:** Discovers source code files, applies language-aware AST chunking (`ast_chunker.py` and `splitter.py`) preserving function/class boundaries, and assigns unique chunk hashes.
2. **Git Commit History & Diffs:** Uses **PyDriller** and **GitPython** to traverse commit logs, parsing commit messages, author metadata, and unified diff changes.
3. **Pull Requests & Reviews:** Integrates with **PyGithub** to extract PR titles, descriptions, and discussion threads.
4. **Rich Metadata & Inline Citations:** Every indexed chunk preserves structured metadata (`repo_name`, `file_path`, `commit_hash`, `pr_number`, `document_type`). When generating answers, the LLM provides grounded, bracketed inline citations `[1]`, `[2]` linking directly to source files and diffs.

---

## 🤖 System Architecture & LangGraph Workflow

The conversational flow is managed by **LangGraph** with PostgreSQL checkpointing for multi-turn conversational memory:

```
                   ┌───────────────────────────────┐
                   │       User Query / POST       │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │   Lightweight Guardrails      │
                   │ (Regex, Length, Injections)   │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │     Intent Classifier Node    │
                   │ (Query Rewriter & Decider)    │
                   └───────┬───────────────┬───────┘
                           │               │
        [Conversational]   │               │   [Repository Query]
                           ▼               ▼
         ┌───────────────────┐   ┌───────────────────────────┐
         │ Greeting Node     │   │ Repository RAG Node       │
         │ Direct Synthesis  │   │ - Hybrid Retrieval        │
         └─────────┬─────────┘   │ - Jina Cross-Encoder      │
                   │             │ - LiteLLM Router Synth    │
                   │             └─────────────┬─────────────┘
                   │                           │
                   └───────────────┬───────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │ Structured Response + Sources │
                   │  (Postgres Checkpointer Save) │
                   └───────────────────────────────┘
```

- **Observability:** End-to-end tracing and spans are recorded across all nodes, retrievers, and API calls using **Pydantic Logfire**.

---

## 📂 Repository Layout

```
rag_project/
├── gitcontext/                      # Core backend package
│   ├── api/                         # FastAPI routing and application layer
│   │   ├── guardrails.py            # Zero-RAM deterministic regex input/output sanitizers
│   │   ├── main.py                  # FastAPI lifespan, routes (/index, /query, /health, /status)
│   │   └── schemas.py               # Pydantic request and response contracts
│   ├── chunking/                    # Code-aware token splitters
│   │   ├── ast_chunker.py           # AST-based Python code boundary splitter
│   │   └── splitter.py              # Recursive character and code chunking logic
│   ├── citations/                   # Citation formatting and validation helpers
│   ├── embeddings/                  # Embeddings interface
│   │   └── embedder.py              # Jina Cloud v3 (384-dim) & Gemini embedding client
│   ├── eval/                        # DeepEval automated evaluation suite
│   │   ├── collect_responses.py     # Evaluation response collection
│   │   ├── deepeval_model.py        # Custom DeepEval LLM wrapper
│   │   ├── evaluate_dataset.py      # Faithfulness, Relevancy, Recall evaluator
│   │   └── generate_dataset.py      # Golden synthetic test dataset generator
│   ├── generation/                  # LLM inference & response generation
│   │   ├── classify_intent.py       # Intent router and conversational query rewriter
│   │   ├── generation.py            # RAG pipeline coordinator
│   │   ├── llm.py                   # LiteLLM router (Groq + Gemini) with in-memory caching
│   │   └── normal_message.py        # Conversational dialogue handler
│   ├── graph/                       # LangGraph state machine
│   │   ├── graph.py                 # Graph assembly and checkpointer injection
│   │   ├── nodes.py                 # Classification, Greeting, and Repository nodes
│   │   └── states.py                # TypedDict conversation state definition
│   ├── ingestion/                   # Multi-source repository ingestion
│   │   ├── commit_ingestion.py      # Commit traverse and unified diff parser
│   │   ├── discovery.py             # File tree filtering and text discovery
│   │   ├── pipeline.py              # Repository ingestion pipeline runner
│   │   ├── pr_ingestion.py          # GitHub Pull Request traversal
│   │   └── repository_cloner.py     # Shallow Git clone and workspace manager
│   ├── reranking/                   # Document reranking
│   │   └── reranker.py              # Jina Reranker v2 multilingual HTTP client
│   ├── retrieval/                   # Retrieval strategies
│   │   ├── hybrid_retriever.py      # BM25 + pgvector fusion via RRF
│   │   └── retriever.py             # Application retriever & startup index cache
│   └── vectorstore/                 # Database vector storage
│       └── supabase_store.py        # Supabase pgvector client (optimized load_documents)
├── ui/                              # Streamlit frontend user interface
│   ├── home.py                      # Multi-page landing and presentation home
│   ├── pages/
│   │   └── application.py           # Interactive chat interface with citation explorer
│   └── styles.css                   # Custom modern dark-mode stylesheet
├── tests/                           # Comprehensive automated test suite
├── pyproject.toml                   # Project metadata and dependencies
├── requirements.txt                 # Frozen production requirements
├── .env.example                     # Environment configuration template
└── README.md                        # Documentation
```

---

## 🚀 Local Setup & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git installed
- Supabase account with a PostgreSQL database (with `pgvector` enabled)

### 2. Clone the Repository
```bash
git clone https://github.com/harwinder766/gitcontext.git
cd gitcontext
```

### 3. Create and Activate a Virtual Environment
```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional) Install development dependencies for running evaluation scripts and linting:*
```bash
pip install -e ".[dev]"
```

### 5. Configure Environment Variables
Copy the template and provide your API keys:
```bash
cp .env.example .env
```
Edit `.env` with your preferred credentials (see the [Environment Variable Reference](#-environment-variable-reference-table) below).

### 6. Run the Test Suite
Ensure all unit and integration tests pass:
```bash
pytest -q
```

### 7. Run the Services

#### Start the FastAPI Backend
```bash
uvicorn gitcontext.api.main:app --reload --host 127.0.0.1 --port 8000
```
- Swagger API Documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

#### Start the Streamlit Frontend
In a separate terminal (with virtual environment activated):
```bash
streamlit run ui/home.py
```
- Access the Web UI at: `http://localhost:8501`

---

## 🔑 Environment Variable Reference Table

| Variable | Description | Required | Example / Default |
| :--- | :--- | :--- | :--- |
| `SUPABASE_DATABASE_URL` | PostgreSQL connection URI for Supabase pgvector and LangGraph checkpointer. | **Yes** | `postgresql://postgres.[ref]:[pass]@aws-0-[region].pooler.supabase.com:5432/postgres` |
| `JINA_API_KEY` | Jina AI API key for zero-RAM embeddings and cross-encoder reranking. | **Yes** | `jina_xxxxxxxxxxxx` |
| `EMBEDDING_PROVIDER` | Embedding provider to use (`jina` or `google`). | No | `jina` |
| `EMBEDDING_MODEL` | Embedding model identifier. | No | `jina-embeddings-v3` |
| `CROSSENCODER_MODEL` | Model for Jina Reranker API. | No | `jina-reranker-v2-base-multilingual` |
| `GROQ_API_KEY` | Groq Cloud API key for primary fast inference. | No* | `gsk_xxxxxxxxxxxx` |
| `GROQ_MODEL` | Primary model identifier on Groq. | No | `llama-3.3-70b-versatile` |
| `GEMINI_API_KEY` | Google AI Studio API key (used for synthesis fallback and Gemini embeddings). | No* | `AIzaxxxxxxxxxxxx` |
| `GEMINI_MODEL` | Fallback model identifier on Google Gemini. | No | `gemini-1.5-flash-latest` |
| `LITELLM_CACHE_TYPE` | Cache backend for LiteLLM (`local` or `redis`). | No | `local` |
| `LITELLM_CACHE_TTL` | Cache duration for duplicate queries (in seconds). | No | `3600` |
| `GITHUB_TOKEN` | Personal GitHub token for traversing private repositories and increasing PR rate limits. | No | `ghp_xxxxxxxxxxxx` |
| `LOGFIRE_TOKEN` | Pydantic Logfire token for distributed cloud tracing. | No | `logfire_token_xxxx` |
| `GITCONTEXT_API_URL` | Base URL used by the Streamlit frontend to reach the backend. | No | `http://127.0.0.1:8000` *(local)* or Render URL *(cloud)* |

*\* Note: At least one LLM key (`GROQ_API_KEY` or `GEMINI_API_KEY`) is required for response synthesis.*

---

## ☁️ Production Cloud Deployment Guide

### 1. Backend Deployment (Render / Koyeb / Railway Free Tier)
Deploy the FastAPI backend as a Docker container or Python Web Service:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:**
  ```bash
  uvicorn gitcontext.api.main:app --host 0.0.0.0 --port $PORT --workers 1 --no-access-log
  ```
- **Recommended Container Environment Variables:**
  - `PYTHONUNBUFFERED=1`
  - `MALLOC_ARENA_MAX=2` *(prevents memory fragmentation in small Linux containers)*
- Because heavy local ML models are eliminated, the entire backend service boots in seconds and comfortably operates under **180 MB RAM**, well within 512 MB free limits.

### 2. Frontend Deployment (Streamlit Community Cloud)
1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
3. Create a **New App**:
   - **Repository:** `your-username/gitcontext`
   - **Branch:** `main`
   - **Main file path:** `ui/home.py`
4. In **Advanced Settings > Secrets**, configure:
   ```toml
   GITCONTEXT_API_URL = "https://gitcontext-xw5w.onrender.com"
   ```
5. Click **Deploy**. Streamlit will automatically discover `ui/pages/application.py` for multipage navigation.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
