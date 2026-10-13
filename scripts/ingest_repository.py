import argparse

from gitcontext.ingestion.pipeline import ingest_repository

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a repository into GitContext."
    )
    parser.add_argument(
        "repo_url",
        help="HTTPS GitHub repository URL.",
    )

    args = parser.parse_args()

    ingest_repository(args.repo_url)


if __name__ == "__main__":
    main()