# GitContext Architecture

## System Overview

GitContext is a RAG (Retrieval-Augmented Generation) system designed for codebase intelligence.
The system ingests a Python repository, processes its contents into searchable chunks,
and answers developer questions with cited sources.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
│                                                              │
│  Repository ─→ Clone/Load ─→ Parse ─→ Chunk ─→ Embed ─→ Store│
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            ↕ Vector DB + BM25 Index
┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                                                              │
│  Question ─→ Embed ─→ Retrieve ─→ Rerank ─→ Build Context   │
│                                          ─→ LLM ─→ Answer   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `config.py`
Central configuration management. Loads environment variables from `.env`
using python-dotenv. No business logic lives here — only knobs and dials.

**Why separate?** Configuration changes frequently (API keys, model names,
chunk sizes). Keeping it isolated means you never grep through business
logic to find a setting.

### `ingestion/`
Responsible for getting repository contents onto disk and discovering
which files to process.

- Clone a repo from a URL (or use a local path)
- Walk the file tree and filter by extension/pattern
- Produce a list of file paths to process

**Does NOT** parse file contents — that's the parser's job.

### `parsing/`
Reads files and extracts structured information using AST (Abstract Syntax Tree)
parsing for Python files and plain text extraction for documentation.

- Python files → extract functions, classes, methods with docstrings
- Markdown/RST files → extract sections
- Other text files → plain text extraction

**Why AST parsing?** Naive text splitting breaks code mid-function.
AST parsing gives us semantically meaningful units (a complete function,
a complete class) which produce much better retrieval results.

### `chunking/`
Splits parsed content into chunks suitable for embedding. This is where
we make chunks that are:

- Small enough to embed meaningfully (typically 256-512 tokens)
- Large enough to contain useful context
- Aligned to code boundaries (don't split mid-function)

**Why code-aware chunking?** Generic text splitters (like LangChain's
RecursiveCharacterTextSplitter) don't understand code structure.
They'll happily split a function in half. Our chunker respects
function/class boundaries.

### `embeddings/`
Converts text chunks into vector representations. Abstracts the
embedding provider so we can swap between:

- OpenAI embeddings (easy, good quality, costs money)
- Sentence-Transformers (free, local, needs GPU for speed)

### `vectorstore/`
Manages the vector database for storing and querying embeddings.

Starting with ChromaDB because:
- Embedded (no separate server process)
- Python-native
- Good enough for single-repo scale
- Easy to replace later if needed

### `retrieval/`
Implements search strategies:

1. **Semantic retrieval** — Find chunks whose embeddings are close
   to the query embedding (cosine similarity)
2. **BM25 retrieval** — Classic keyword matching (handles exact
   function names, variable names)
3. **Hybrid retrieval** — Combine semantic + BM25 scores with
   configurable weighting

**Why hybrid?** Semantic search is great for "how does authentication work?"
but bad for "find the function `parse_config`". BM25 is the opposite.
Combining them covers both cases.

### `reranking/`
Takes the top-N results from hybrid retrieval and re-scores them
using a more expensive but more accurate model (e.g., a cross-encoder).

**Why rerank?** The initial retrieval casts a wide net (fast but imprecise).
Reranking narrows it down (slow but accurate). This two-stage approach
is standard in information retrieval.

### `generation/`
Constructs prompts from retrieved context and sends them to the LLM.

- Builds a prompt with the question + retrieved chunks
- Calls the LLM API
- Returns the raw response

### `citations/`
Extracts source references from the LLM response and maps them back
to the original files/line numbers.

### `models/`
Data classes shared across modules:

- `Document` — A parsed file with metadata
- `Chunk` — A piece of a document with embedding
- `QueryResult` — A retrieval result with score
- `Answer` — LLM response with citations

**Why a separate models module?** Avoids circular imports. Every module
can import from `models/` without importing each other.

## Data Flow

```
1. User provides a repo URL or local path
2. ingestion/ clones it and lists Python + doc files
3. parsing/ extracts functions, classes, docstrings from each file
4. chunking/ splits large items into embedding-sized chunks
5. embeddings/ converts each chunk to a vector
6. vectorstore/ stores vectors + metadata in ChromaDB
7. (BM25 index is also built from chunk text)

--- Query time ---

8.  User asks a question
9.  embeddings/ converts the question to a vector
10. retrieval/ searches both vector DB and BM25 index
11. retrieval/ merges results (hybrid fusion)
12. reranking/ re-scores top results with cross-encoder
13. generation/ builds a prompt with top chunks as context
14. generation/ calls the LLM
15. citations/ extracts file/line references from the answer
16. Return answer + citations to the user
```

## Key Design Decisions

1. **Monolithic package, not microservices** — This is a single-user
   tool. No need for service boundaries, message queues, or containers
   until we have a real scaling need.

2. **Embedded vector DB** — ChromaDB runs in-process. No Docker, no
   server management. We can migrate to a client-server DB later if needed.

3. **Python-only initially** — Constraining to Python lets us build
   a high-quality AST parser before generalizing to other languages.

4. **No LangChain** — We're building the RAG pipeline from components
   to understand how it works. LangChain adds abstraction without
   adding understanding for a learning project.

5. **Configuration via environment variables** — Simple, secure,
   12-factor app compatible. No config files to accidentally commit.
