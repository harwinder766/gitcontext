# Development Phases

## Phase 0: Project Setup & Architecture ← CURRENT
- [x] Initialize Git repository
- [x] Create project structure
- [x] Write architecture documentation
- [x] Set up pyproject.toml with dev dependencies
- [ ] Set up virtual environment
- [ ] Verify installation works
- [ ] Initial commit

## Phase 1: Repository Ingestion
- Clone a GitHub repo to local disk (or accept a local path)
- Walk the directory tree
- Filter for Python files and documentation files
- Return a list of file paths with basic metadata (size, extension)
- **Test**: Ingest a small open-source Python repo and verify file list

## Phase 2: Python Code Parsing & Code-Aware Chunking
- Parse Python files using the `ast` module
- Extract functions, classes, methods with their docstrings
- Extract module-level docstrings
- Parse Markdown files into sections
- Split large code blocks into chunks respecting boundaries
- **Test**: Parse a Python file and verify extracted functions match

## Phase 3: Embeddings & Vector Storage
- Generate embeddings for chunks (OpenAI or Sentence-Transformers)
- Store embeddings + metadata in ChromaDB
- Build a simple indexing pipeline: file → parse → chunk → embed → store
- **Test**: Index a small repo and verify vectors are stored correctly

## Phase 4: Basic Semantic Retrieval
- Accept a natural language query
- Embed the query
- Search ChromaDB for nearest neighbors
- Return ranked chunks with metadata
- **Test**: Ask "what does function X do?" and verify relevant chunk is returned

## Phase 5: RAG Generation
- Build a prompt template with retrieved context
- Send to LLM (OpenAI/Anthropic)
- Return the generated answer
- **Test**: End-to-end query → retrieval → generation on a test repo

## Phase 6: Source Citations
- Instruct the LLM to cite sources in its response
- Extract citations and map to file paths + line numbers
- Format citations in the response
- **Test**: Verify citations point to actual source locations

## Phase 7: BM25 Retrieval
- Build a BM25 index from chunk text
- Implement keyword search
- **Test**: Query exact function names and verify BM25 finds them

## Phase 8: Hybrid Retrieval
- Combine semantic and BM25 scores
- Implement Reciprocal Rank Fusion (RRF) or weighted combination
- **Test**: Verify hybrid beats either method alone on test queries

## Phase 9: Reranking
- Add a cross-encoder reranker
- Re-score top-N hybrid results
- **Test**: Verify reranking improves result quality

## Phase 10: Git Commit Integration
- Parse git log for commit history
- Chunk commit messages with diffs
- Index commits alongside code
- **Test**: Ask "what changed in the last commit?" and get accurate answer

## Phase 11: Pull Request Integration
- Fetch PR data via GitHub API
- Index PR descriptions, comments, review comments
- **Test**: Ask about a specific PR and get relevant context

## Phase 12: Evaluation
- Build a small evaluation dataset (question + expected answer)
- Measure retrieval quality (recall@k, MRR)
- Measure generation quality (manual review or LLM-as-judge)
- Compare retrieval strategies

## Phase 13: FastAPI API
- REST API for indexing and querying
- Endpoints: POST /index, POST /query, GET /status
- API documentation via OpenAPI/Swagger

## Phase 14: Simple UI
- Minimal web interface for asking questions
- Display answers with clickable source citations
- Show retrieval confidence scores

## Phase 15: Advanced Features (Optional)
- Incremental re-indexing (only re-process changed files)
- Stale code detection
- Multi-language support
- Conversation memory
- Repository comparison
