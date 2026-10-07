# Technical Decisions

This document records major technical decisions, their context, and rationale.

---

## DEC-001: No LangChain / LlamaIndex

**Date**: 2026-10-02
**Status**: Accepted

**Context**: Frameworks like LangChain and LlamaIndex provide pre-built RAG
pipelines that could accelerate development.

**Decision**: Build the RAG pipeline from individual components without using
a RAG framework.

**Rationale**:
- This is a learning project. The goal is to understand how RAG works, not
  to ship a product as fast as possible.
- Frameworks hide the details that matter most: how chunking affects retrieval,
  how prompt construction affects generation quality, how hybrid search
  actually works.
- Debugging framework abstractions is harder than debugging your own code.
- We can always adopt a framework later if we understand what it does.

---

## DEC-002: ChromaDB as Initial Vector Store

**Date**: 2026-10-02
**Status**: Accepted

**Context**: Need a vector database to store and query embeddings. Options
include ChromaDB, FAISS, Qdrant, Pinecone, Weaviate, Milvus.

**Decision**: Use ChromaDB.

**Rationale**:
- Embedded mode (no separate server) simplifies development setup.
- Python-native API.
- Supports metadata filtering.
- Handles persistence to disk.
- Good enough for single-repo scale (thousands to tens of thousands of chunks).
- If we outgrow it, migrating to Qdrant or FAISS is straightforward since
  our vectorstore module abstracts the implementation.

**Alternatives considered**:
- FAISS: More performant but no built-in metadata support, no persistence
  out of the box.
- Qdrant: Excellent but requires running a server (Docker).
- Pinecone: Cloud-only, costs money, adds external dependency.

---

## DEC-003: Python-Only Scope Initially

**Date**: 2026-10-02
**Status**: Accepted

**Context**: The system could support multiple programming languages from
the start, but this adds significant complexity.

**Decision**: Support only Python repositories initially.

**Rationale**:
- Python's `ast` module provides excellent built-in parsing — no external
  parser needed.
- Constraining scope lets us build a high-quality parser before generalizing.
- Multi-language support (tree-sitter, etc.) is a natural Phase 15 extension.

---

## DEC-004: Monolithic Package Structure

**Date**: 2026-10-02
**Status**: Accepted

**Context**: Could structure as microservices, separate packages, or a
single package with clear module boundaries.

**Decision**: Single Python package (`gitcontext/`) with submodules.

**Rationale**:
- Single-user development tool doesn't need service boundaries.
- Simpler development, testing, and deployment.
- Module boundaries within the package provide enough separation.
- Can extract services later if a real scaling need arises.

---

## DEC-005: Configuration via Environment Variables

**Date**: 2026-10-02
**Status**: Accepted

**Context**: Need to manage API keys, model names, and tunable parameters.

**Decision**: Use environment variables loaded via python-dotenv, with a
`Settings` class using pydantic-settings.

**Rationale**:
- Follows 12-factor app principles.
- `.env` file is git-ignored, preventing accidental key commits.
- pydantic-settings provides validation and type coercion.
- Single source of truth for all configuration.

---

## DEC-006: LangChain Document as Universal Data Type

**Date**: 2026-10-08
**Status**: Accepted
**Supersedes**: DEC-001 (partially — we still build pipeline logic ourselves)

**Context**: The project initially planned custom `Document` and `Chunk` dataclasses
(DEC-001) and no LangChain usage. However, the project will use LangChain components
for embeddings, text splitting, and vector store integration (ChromaDB). Using custom
types would require converter functions at every boundary between our code and
LangChain components.

**Decision**: Use `langchain_core.documents.Document` as the universal data container
across the entire pipeline. Store file and chunk-specific information in the metadata
dictionary.

**Rationale**:
- `langchain_core` is the lightweight base package, not the full LangChain framework.
  `Document` is just a Pydantic model with `page_content` and `metadata`.
- One type flows through the entire pipeline (ingestion → chunking → embedding →
  retrieval) without conversion.
- Plugs directly into LangChain ecosystem: text splitters, embedding models, ChromaDB,
  retrievers all accept and return `Document` objects.
- Eliminates converter code between custom types and LangChain types.
- The custom `Chunk` dataclass becomes unnecessary — a `Document` with chunk-specific
  metadata (chunk_type, name, start_line, end_line) serves the same purpose.
- DEC-001's core principle still applies: we build our own pipeline logic (ingestion,
  AST parsing, prompt construction) rather than using LangChain chains or agents.

**Trade-offs accepted**:
- Metadata keys are strings, not typed fields. Typos in metadata key names
  (e.g., `metadata["surce"]`) are not caught by the editor. Mitigated by
  documenting standard keys in `gitcontext/models/__init__.py`.
- Coupling to `langchain_core`. Acceptable because we already depend on LangChain
  components for embeddings and vector store integration.

**Standard metadata keys** (documented in `gitcontext/models/__init__.py`):
- `source` — relative path to the file
- `absolute_path` — full filesystem path as a string
- `extension` — file extension including the dot
- `repo_name` — name of the repository directory
- `size_bytes` — file size in bytes
