# import argparse

# from gitcontext.generation.generation import RAG


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Ask a question about an indexed code repository."
#     )
#     parser.add_argument(
#         "question",
#         help="Question to ask about the repository.",
#     )

#     args = parser.parse_args()

#     rag = RAG(query=args.question, filter = {'repo_name': 'test_repo'})
#     answer = rag.generate_response()

#     print("\nGitContext Answer:\n")
#     print(answer)


# if __name__ == "__main__":
#     main()

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

    args = parser.parse_args()

    rag = RAG(query=args.question, filter = {'repo_name': 'test_repo'})
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
            print(f"    Extension  : {source.extension}")
            print(f"    Size       : {source.size_bytes} bytes")
            print(f"    Full path  : {source.absolute_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

