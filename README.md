# GitContext

<p align="center">
  <strong>AI-Powered Codebase Assistant & Developer Copilot built with Hybrid RAG</strong>
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
- **Frontend Web UI (Streamlit):** [Open GitContext](https://gitcontext-9w9gptwf8yyqtw5cwxrllc.streamlit.app) *(or run locally via `streamlit run ui/home.py`)*

---

## ⚙️ Lightweight, API-Based Architecture

GitContext uses hosted APIs for embedding generation and document reranking instead of loading those model weights into the backend process. The backend still keeps its Python runtime, application state, cached data, and BM25 index in memory, so actual consumption depends on repository size, workload, and deployment environment.

The main components of this design are:

1. **Hosted Embeddings:** Uses the **Jina AI Embeddings API** (`jina-embeddings-v3`) to generate 384-dimensional vectors, with an optional **Google Gemini** embedding provider. Embeddings are stored in Supabase PGVector. The selected embedding provider and dimensions must remain consistent with the existing vector collection.
2. **API-Based Reranking:** Uses the **Jina AI Reranker API** (`jina-reranker-v2-base-multilingual`) to score query-document pairs without loading a local cross-encoder model into the backend process. This reduces local model memory requirements, while introducing an external API dependency and network latency.
3. **Rule-Based Guardrails:** Uses lightweight checks for input validation and suspicious patterns, including invisible zero-width Unicode characters and prompt-injection indicators. The checks are designed to avoid unnecessarily flagging common developer terms such as `api_key`, `openai`, `token`, `secret`, and `create_llm`.
4. **LLM Routing & Response Caching:** Uses **LiteLLM Router** to route requests between configured providers such as **Groq** (`llama-3.3-70b-versatile`) and **Google Gemini** (`gemini-1.5-flash-latest`). An in-memory TTL cache (`LITELLM_CACHE_TYPE=local`, TTL 3600s) can reuse responses to repeated queries and reduce duplicate provider calls. Provider availability, quotas, and rate limits still apply.

---

## 📏 Deployment Notes & Large Repositories

GitContext is a personal portfolio project, so the currently deployed demo uses free-tier or quota-limited services to keep it publicly accessible:

- **Backend hosting:** Render
- **Vector database:** Supabase PostgreSQL with `pgvector`
- **Embedding and reranking:** Jina AI APIs
- **Frontend:** Streamlit Community Cloud

The ingestion pipeline is designed to collect context from repository files, commit history, and pull requests. However, the capacity of the **publicly deployed instance** is constrained by the resources and rate limits of its hosting and API providers. In particular, API quotas, request limits, and rate limiting can make ingestion of very large repositories slow or cause some ingestion operations to fail on the current free-tier deployment. This is a limitation of the demo environment and its service quotas, not a claim that the GitContext approach has a fixed maximum repository size. Actual capacity also depends on repository size, file count, commit and pull-request history, and the resources available to the backend.

### Running GitContext locally for larger repositories

If you need to ingest a large repository, you can clone this project and run it locally or on infrastructure with more generous resources. A possible configuration is to:

1. Replace the hosted **Jina embedding API** integration with a local embedding model, such as a compatible Sentence Transformers model.
2. Replace **Supabase PGVector** with a local vector store, such as **Chroma** or **FAISS**.
3. If you also want to avoid Jina's reranking API limits, use a local reranker or adjust the retrieval pipeline to run without the hosted reranking step.
4. Keep the embedding model, vector dimensions, and vector-store configuration consistent. Re-embed and re-index the repository when changing embedding models or vector-store backends.

This setup can reduce dependence on external API quotas and hosted database limits, but it requires the corresponding local integrations and enough CPU, RAM, and disk space on your machine. The current public deployment uses Jina and Supabase; local alternatives may require configuration changes or implementation work and are not automatically enabled by the deployed demo.

---

## 🔍 Hybrid Retrieval System

Codebase search benefits from both **semantic understanding** (e.g., *"where is the database connection pool defined?"*) and **lexical matching** (e.g., *"find the `create_embedding_model` function definition"*). GitContext combines these approaches in a hybrid retrieval pipeline:

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
Connects to the remote PostgreSQL database through `langchain-postgres` and `psycopg`. Chunks are queried using cosine similarity over 384-dimensional dense representations.

### 2. Sparse Lexical Search (rank-bm25)
Maintains an in-memory BM25 index over codebase tokens. This helps retrieve exact identifiers, class names, file names, and error messages that semantic search may rank less effectively.

### 3. Startup Index Caching
GitContext builds the in-memory BM25 index during FastAPI lifespan startup and stores the initialized retriever in `app.state.retriever`. The retriever is refreshed when a repository is indexed (`POST /index`) rather than rebuilding the BM25 index from the database for every question. This avoids repeated full-index loading during normal query handling. Actual latency depends on repository size, database performance, network conditions, and retrieval configuration.

### 4. Reciprocal Rank Fusion (RRF) & Cross-Encoder Reranking
Combines ranked candidates from BM25 and pgvector using reciprocal rank scoring ($RRF_{score} = \sum \frac{1}{k + rank}$, $k=60$). The merged candidate pool is then passed to Jina Reranker v2, which scores query-document pairs to help order the most relevant chunks.

---

## 📦 Multi-Source Git Ingestion Pipeline

GitContext builds context from three sources and enriches the resulting chunks with metadata and citations:

1. **Repository Source Code Tree:** The current AST-based source-code chunker is implemented for **Python**. It uses `ast_chunker.py` and `splitter.py` to identify Python function and class boundaries, then assigns unique chunk hashes. Python is therefore the primary language supported by the structured source-tree ingestion path.
2. **Git Commit History & Diffs:** Uses **PyDriller** and **GitPython** to traverse commit logs, parsing commit messages, author metadata, and unified diff changes. Commit messages and text diffs can provide context for other programming languages as well; this path does not apply language-specific AST parsing to those languages.
3. **Pull Requests & Reviews:** Integrates with **PyGithub** to extract PR titles, descriptions, and discussion threads. PR descriptions, comments, and code-change text can add context for repositories written in other languages.
4. **Rich Metadata & Inline Citations:** Indexed content preserves structured metadata such as `repo_name`, `file_path`, `commit_hash`, `pr_number`, and `document_type`. When generating answers, the LLM provides bracketed inline citations `[1]`, `[2]` linking to available source files and diffs.

### Language Support

**Python is the primary language supported by the current repository-file ingestion pipeline**, because the AST-based parser currently handles Python syntax only. GitContext can still collect information about repositories written in other languages through commit-history/diff ingestion and pull-request ingestion. However, those sources provide commit- and PR-level context; they do not provide the same AST-aware, function/class-level source-code indexing currently available for Python. Support for AST-based chunking of additional languages would require implementing language-specific parsers or chunkers.

---
## Project Demo

https://github.com/user-attachments/assets/9826f681-e9a2-482d-a0cb-90073493aa27


## 🧪 RAG Pipeline Evaluation

GitContext's RAG pipeline is evaluated with **[DeepEval](https://deepeval.com/)** on a test set of **18 cases**. The pass rate is the percentage of cases that met the configured threshold of **0.70** for each metric.

| Metric | Average score | Cases passed | Pass rate | Cases evaluated |
| :--- | ---: | ---: | ---: | ---: |
| [Faithfulness](https://deepeval.com/docs/metrics-faithfulness) | **0.9323** | 17/18 | **94.44%** | 18 |
| [Answer relevancy](https://deepeval.com/docs/metrics-answer-relevancy) | **0.9232** | 17/18 | **94.44%** | 18 |
| [Contextual relevancy](https://deepeval.com/docs/metrics-contextual-relevancy) | **0.5381** | 10/18 | **55.56%** | 18 |
| [Contextual precision](https://deepeval.com/docs/metrics-contextual-precision) | **0.8611** | 14/18 | **77.78%** | 18 |
| [Contextual recall](https://deepeval.com/docs/metrics-contextual-recall) | **0.8296** | 15/18 | **83.33%** | 18 |

### What these results indicate

- **Answer generation is a relative strength in this evaluation.** Faithfulness (0.9323) and answer relevancy (0.9232) both score above 0.92, and 17 of 18 cases pass the threshold for each metric. This suggests that the generated answers are generally grounded in the retrieved context and address the questions in this test set.
- **Retrieval performance is mixed.** Contextual precision (0.8611; 14/18 passing) and contextual recall (0.8296; 15/18 passing) are encouraging. However, contextual relevancy is weak: its average is 0.5381, and only 3 of 18 cases pass. This indicates that the retrieved context often includes material that is not sufficiently relevant to the query, even when useful evidence may also be present.
- **The main improvement area is retrieval relevance and noise reduction.** Areas to investigate include chunk boundaries and size, the number of candidates retrieved (top-K), metadata or file filtering, query rewriting, and the balance between BM25 and dense vector retrieval.

**Overall assessment:** these are promising early results for answer faithfulness and answer relevancy, but they do **not** show that retrieval quality is consistently strong. Contextual relevancy is a clear weakness that should be addressed before describing the RAG pipeline as highly reliable. The evaluation contains only 18 cases, so it is a small sample rather than a production-wide performance guarantee. Results can vary with the dataset, metric configuration, and evaluator model; expand the test set and rerun DeepEval after retrieval or generation changes.

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
│   │   ├── guardrails.py            # Rule-based input/output checks and sanitizers
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
| `JINA_API_KEY` | Jina AI API key for hosted embeddings and reranking (when those providers are enabled). | **Yes** | `jina_xxxxxxxxxxxx` |
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

### 1. Backend Deployment (Render / Koyeb / Railway)
Deploy the FastAPI backend as a Docker container or Python Web Service:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:**
  ```bash
  uvicorn gitcontext.api.main:app --host 0.0.0.0 --port $PORT --workers 1 --no-access-log
  ```
- **Useful runtime setting:** `PYTHONUNBUFFERED=1` (makes application logs appear promptly in the hosting provider's log viewer).
- Using hosted embedding and reranking APIs avoids loading those model weights into the backend process. Measure memory under your expected repository size and request load, and select a hosting plan with enough headroom for startup and peak usage.

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
