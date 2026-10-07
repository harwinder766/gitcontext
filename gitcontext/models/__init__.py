"""Data models shared across GitContext modules.

All pipeline stages use langchain_core.documents.Document as the universal
data container. File and chunk-specific information is stored in the
metadata dictionary.

Standard metadata keys used across the pipeline:

    Ingestion stage:
        source        - Relative path to the file (e.g., "src/utils.py")
        absolute_path - Full filesystem path as a string
        extension     - File extension including the dot (e.g., ".py")
        repo_name     - Name of the repository directory
        size_bytes    - File size in bytes

    Chunking stage (to be added later):
        chunk_type    - "function", "class", "markdown_section", etc.
        name          - Name of the code element or section
        start_line    - Starting line number in the source file (1-indexed)
        end_line      - Ending line number in the source file (1-indexed)
"""
