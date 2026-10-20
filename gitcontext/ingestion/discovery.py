from pathlib import Path

import logfire
from langchain_core.documents import Document

# Directories to skip during file discovery.
# These never contain source code or documentation we want to index.
EXCLUDED_DIRS: set[str] = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    ".tox",
    ".eggs",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".nox",
    "dist",
    "build",
}

# File extensions we want to process.
# Python source files and common documentation formats.
SUPPORTED_EXTENSIONS: set[str] = {
    ".py",
    ".md",
    ".rst",
    ".txt",
}


def _is_excluded(file_path: Path, repo_path: Path) -> bool:
    """Check if a file path is inside an excluded directory."""
    relative = file_path.relative_to(repo_path)
    for part in relative.parts:
        if part in EXCLUDED_DIRS:
            return True
        if part.endswith(".egg-info"):
            return True
    return False


@logfire.instrument("Discover repository files")
def discover_files(
    repo_path: Path,
    repo_name: str | None = None,
) -> list[Document]:
    """Discover relevant source and documentation files in a repository."""
    repo_path = repo_path.resolve()
    canonical_repo_name = repo_name or repo_path.name

    # Keep validation inside the instrumented function so failures are visible.
    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo_path}")

    documents: list[Document] = []
    files_scanned = 0
    files_skipped = 0

    with logfire.span(
        "Walk repository tree",
        repo_name=canonical_repo_name,
    ):
        for file_path in sorted(repo_path.rglob("*")):
            # Skip directories — we only want files.
            if file_path.is_dir():
                continue

            files_scanned += 1

            # Skip files inside excluded directories.
            if _is_excluded(file_path, repo_path):
                files_skipped += 1
                continue

            # Skip files with unsupported extensions.
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                files_skipped += 1
                continue

            # Read the file content, skipping files that cannot be decoded.
            try:
                content = file_path.read_text(encoding="utf-8")
                size_bytes = file_path.stat().st_size
            except (UnicodeDecodeError, OSError):
                # The file may be binary, inaccessible, or locked.
                files_skipped += 1
                logfire.debug(
                    "Skipping unreadable file {source}",
                    source=file_path.relative_to(repo_path).as_posix(),
                )
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path.relative_to(repo_path).as_posix(),
                        "absolute_path": str(file_path),
                        "extension": file_path.suffix.lower(),
                        "repo_name": canonical_repo_name,
                        "size_bytes": size_bytes,
                    },
                )
            )

    logfire.info(
        "Repository file discovery completed",
        repo_name=canonical_repo_name,
        files_scanned=files_scanned,
        documents_created=len(documents),
        files_skipped=files_skipped,
    )

    return documents
