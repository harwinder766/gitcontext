import argparse
from pathlib import Path

from gitcontext.ingestion.pipeline import ingest_repository

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a repository into GitContext."
    )
    parser.add_argument(
        "repo_path",
        type=Path,
        help="Path to the repository.",
    )

    args = parser.parse_args()

    if not args.repo_path.is_dir():
        parser.error(f"Directory does not exist: {args.repo_path}")

    ingest_repository(args.repo_path)


if __name__ == "__main__":
    main()