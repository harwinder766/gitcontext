from github import Auth, Github
from langchain_core.documents import Document
from itertools import islice

import os
from dotenv import load_dotenv
load_dotenv()

class PullRequestIngestion:
    """Fetch pull requests of one GitHub repository as Documents."""
 
    def __init__(
        self,
        repo_full_name: str,           
        token: str | None = None,     
        state: str = "all",        
        max_prs: int = 50,             
    ):
        if not repo_full_name or "/" not in repo_full_name:
            raise ValueError("repo_full_name must look like 'owner/repo'.")
 
        token = token or os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GitHub token is missing. Set GITHUB_TOKEN in your .env file.")
 
        self.repo_full_name = repo_full_name
        self.state = state
        self.max_prs = max_prs
        self.repo = Github(auth=Auth.Token(token)).get_repo(repo_full_name)
 
    def pulls_traverse(self) -> list[Document]:
        """Go through the latest pull requests and create Documents."""
        documents: list[Document] = []
 
        pulls = self.repo.get_pulls(state=self.state, sort="updated", direction="desc")
 
        for pr in islice(pulls, self.max_prs):
            # 1. PR description
            description = (pr.body or "").strip() or "(no description)"
            documents.append(
                self._make_document(
                    pr=pr,
                    item_type="description",
                    item_label="PR description",
                    text=description,
                    author=self._author(pr),
                    url=pr.html_url,
                    source=f"PR #{pr.number}",
                )
            )
 
            # 2. Discussion comments (the general conversation)
            for comment in pr.get_issue_comments():
                if self._skip_comment(comment):
                    continue
                author = self._author(comment)
                documents.append(
                    self._make_document(
                        pr=pr,
                        item_type="comment",
                        item_label=f"Comment by {author}",
                        text=comment.body.strip(),
                        author=author,
                        url=comment.html_url,
                        source=f"PR #{pr.number} comment {comment.id}",
                    )
                )
 
            # 3. Review comments (comments on specific lines of code)
            for comment in pr.get_review_comments():
                if self._skip_comment(comment):
                    continue
                author = self._author(comment)
                documents.append(
                    self._make_document(
                        pr=pr,
                        item_type="review_comment",
                        item_label=f"Review comment by {author} on {comment.path}",
                        text=comment.body.strip(),
                        author=author,
                        url=comment.html_url,
                        source=f"PR #{pr.number} review comment {comment.id}",
                        file_path=comment.path,
                    )
                )

            # 4. Submitted reviews
            for review in pr.get_reviews():
                if not (review.body or "").strip():
                    continue
            
                author = self._author(review)
            
                documents.append(
                    self._make_document(
                        pr=pr,
                        item_type="review",
                        item_label=f"Review by {author} ({review.state})",
                        text=review.body.strip(),
                        author=author,
                        url=review.html_url,
                        source=f"PR #{pr.number} review {review.id}",
                    )
                )
             
        return documents
 
    def _make_document(
        self,
        pr,
        item_type: str,
        item_label: str,
        text: str,
        author: str,
        url: str,
        source: str,
        file_path: str | None = None,
    ) -> Document:
        """Build one Document. The header at the top tells which PR the text belongs to."""
        pr_state = "merged" if pr.merged else pr.state
 
        page_content = (
            f"PR #{pr.number}: {pr.title}\n"
            f"PR author: {self._author(pr)}\n"
            f"PR state: {pr_state}\n"
            f"Item: {item_label}\n\n"
            f"Body:\n{text}"
        )
 
        return Document(
            page_content=page_content,
            metadata={
                "document_type": "pull_request",
                "pr_item_type": item_type,
                "repo_name": self.repo.name,
                "source": source,
                "pr_number": pr.number,
                "pr_title": pr.title,
                "pr_state": pr_state,
                "author_name": author,
                "url": url,
                "commit_hash": pr.merge_commit_sha,   # links this PR to the commits from Phase 10
                "created_at": pr.created_at.isoformat(),
                "file_path": file_path,
            },
        )
 
    @staticmethod
    def _author(item) -> str:
        """Return the username, or 'unknown' for deleted accounts."""
        return item.user.login if item.user else "unknown"
 
    @staticmethod
    def _skip_comment(comment) -> bool:
        """Skip empty comments and comments written by bots."""
        if not (comment.body or "").strip():
            return True
        if comment.user is None or comment.user.type == "Bot":
            return True
        return False