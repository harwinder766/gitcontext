# GitContext

A codebase intelligence and developer copilot powered by RAG (Retrieval-Augmented Generation).

## What is GitContext?

GitContext lets you connect a Python repository and ask natural language questions about:
- Source code (functions, classes, modules)
- Documentation (README, docstrings)
- Git history (commits, changes)
- Pull requests (planned)

It retrieves relevant context using hybrid search (semantic + keyword) and generates answers with source citations using an LLM.

## Project Status

🚧 **Phase 0: Project Setup & Architecture** — In progress

See [PHASES.md](docs/PHASES.md) for the full development roadmap.

## Architecture

```
Repository → Ingestion → Parsing → Chunking → Embeddings → Vector DB
                                                                  ↓
User Query → Hybrid Retrieval (Semantic + BM25) → Reranking → LLM → Answer + Citations
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd rag_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Copy environment template
cp .env.example .env
# Edit .env with your API keys

# 5. Run tests
pytest
```

## Project Structure

```
gitcontext/           # Main package
├── config/           # Configuration and settings
├── ingestion/        # Repository cloning and file discovery
├── parsing/          # Code and document parsing
├── chunking/         # Code-aware text chunking
├── embeddings/       # Text → vector embedding
├── vectorstore/      # Vector database operations
├── retrieval/        # Semantic, keyword, and hybrid retrieval
├── reranking/        # Result reranking
├── generation/       # LLM prompt construction and answer generation
├── citations/        # Source citation extraction
└── models/           # Data models (documents, chunks, queries)
tests/                # Test suite mirroring package structure
docs/                 # Architecture docs and decision records
data/                 # Local data (git-ignored)
```

## Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Language | Python 3.10+ | Primary ecosystem for ML/NLP tooling |
| Embeddings | OpenAI / Sentence-Transformers | TBD based on Phase 3 evaluation |
| Vector DB | ChromaDB | Simple, embedded, no server needed |
| Keyword Search | rank-bm25 | Lightweight BM25 implementation |
| LLM | OpenAI GPT / Anthropic Claude | TBD based on Phase 5 evaluation |
| API | FastAPI | Async, typed, auto-docs (Phase 13) |
| Testing | pytest | Standard Python testing |

## Decision Log

See [docs/DECISIONS.md](docs/DECISIONS.md) for a record of major technical decisions and their rationale.

## License

MIT
