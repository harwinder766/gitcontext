from pydriller import Repository
from langchain_core.documents import Document
from datetime import datetime
from pathlib import Path

class CommitIngestion:
    def __init__(
            self,
            repo_path: str,
            since: datetime | None = None,
            to: datetime | None = None,
            from_commit: str | None = None,
            to_commit: str | None = None,
            from_tag: str | None = None,
            to_tag: str | None = None,
            only_commits: list[str] | None = None,
            only_in_branch: list[str] | None = None,
            only_authors : list[str] | None = None,
            only_modifications_with_file_types : list[str] | None = None,
            repo_name: str | None = None,
    ):
        if not repo_path:
            raise ValueError("Repository path cannot be empty.")
        self.repo_path = repo_path
        self.repo_name = repo_name or Path(repo_path).name
        self.since = since
        self.to = to
        self.from_commit = from_commit
        self.to_commit = to_commit  
        self.from_tag = from_tag
        self.to_tag = to_tag
        self.only_commits = only_commits
        self.only_in_branch = only_in_branch   
        self.only_authors = only_authors
        self.only_modifications_with_file_types = only_modifications_with_file_types

    
    def commits_traverse(self) -> list[Document]:
        """Traverse commits and create one Document per modified file."""
        documents: list[Document] = []
        for commit in Repository(
            self.repo_path,
            order="reverse",
            since=self.since,
            to=self.to,
            from_commit=self.from_commit,
            to_commit=self.to_commit,
            from_tag=self.from_tag,
            to_tag=self.to_tag,
            only_commits=self.only_commits,
            only_in_branch=self.only_in_branch,
            only_authors=self.only_authors,
            only_modifications_with_file_types=(
                self.only_modifications_with_file_types
            ),
        ).traverse_commits():
    
            for file in commit.modified_files:
                file_path = file.new_path or file.old_path
                diff = file.diff or ""
    
                # Skip changes without a usable path or textual diff.
                if not file_path or not diff.strip():
                    continue
    
                source = Path(file_path).as_posix()
    
                page_content = (
                    f"Commit message: {commit.msg}\n"
                    f"Commit hash: {commit.hash}\n"
                    f"File path: {file_path}\n"
                    f"Change type: {file.change_type.name}\n\n"
                    f"Diff:\n{diff}"
                )
    
                document = Document(
                    page_content=page_content,
                    metadata={
                        "document_type": "commit",
                        "repo_name": self.repo_name,
                        "source": source,
                        "commit_hash": commit.hash,
                        "commit_msg": commit.msg,
                        "author_name": commit.author.name,
                        "author_email": commit.author.email,
                        "commit_date": commit.committer_date.isoformat(),
                        "branch": list(commit.branches),
                        "file_path": file_path,
                        "change_type": file.change_type.name,
                        "no_of_lines_added": file.added_lines,
                        "no_of_lines_deleted": file.deleted_lines,
                    },
                )
    
                documents.append(document)
    
        return documents