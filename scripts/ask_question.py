import argparse

from gitcontext.generation.generation import RAG

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question about an indexed code repository."
    )
    parser.add_argument(
        "question",
        help="Question to ask about the repository.",
    )
    parser.add_argument(
        "--repo-name",
        required=True,
        help="Repository name stored in Supabase.",
    )

    args = parser.parse_args()

    rag = RAG(
        query=args.question,
        filter={"repo_name": args.repo_name},
    )
    response = rag.generate_response()

    # Display the answer
    print("\n" + "=" * 60)
    print("GitContext Answer")
    print("=" * 60)
    print(response.answer)

    # Display source references
    print("\n" + "-" * 60)
    print("Sources")
    print("-" * 60)

    if not response.sources:
        print("No sources available.")
    else:
        for i, source in enumerate(response.sources, start=1):
            print(f"\n[{i}] {source.file_path}")
            print(f"    Repository : {source.repo_name}")
            print(f"    Type       : {source.document_type}")
            if source.commit_hash:
                print(f"    Commit     : {source.commit_hash}")
                print(f"    Message    : {source.commit_msg}")
                print(f"    Chunk      : {source.chunk_index}")
                print(f"    Extension  : {source.extension}")
                print("    Size       : not available for commit diff")
                print(f"    File path  : {source.file_path}")
            else:
                print(f"    Extension  : {source.extension}")
                print(f"    Size       : {source.size_bytes} bytes")
                print(f"    Full path  : {source.absolute_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
