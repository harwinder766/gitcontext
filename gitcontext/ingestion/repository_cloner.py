# from pathlib import Path
# from urllib.parse import urlparse

# from git import Repo, GitCommandError, InvalidGitRepositoryError, NoSuchPathError


# def clone_repository(
#     repo_url: str,
#     clone_dir: str | Path = "repositories",
# ) -> Path:
#     """Clone a GitHub repository and return its local path."""

#     # Step 1: Validate the URL
#     if not isinstance(repo_url, str) or not repo_url.strip():
#         raise ValueError("Repository URL cannot be empty.")

#     repo_url = repo_url.strip()
#     parsed_url = urlparse(repo_url)

#     if (
#         parsed_url.scheme != "https"
#         or parsed_url.hostname != "github.com"
#         or parsed_url.port is not None
#         or parsed_url.username is not None
#         or parsed_url.password is not None
#         or parsed_url.query
#         or parsed_url.fragment
#     ):
#         raise ValueError(
#             "Provide a valid HTTPS GitHub repository URL, "
#             "such as https://github.com/owner/repository."
#         )

#     # Step 2: Extract owner and repository name
#     parts = parsed_url.path.strip("/").split("/")

#     if (
#         len(parts) != 2
#         or not all(parts)
#         or parts[0] in {".", ".."}
#         or parts[1] in {".", ".."}
#     ):
#         raise ValueError(
#             "URL must have the format "
#             "https://github.com/owner/repository."
#         )

#     owner, repo_name = parts

#     if repo_name.endswith(".git"):
#         repo_name = repo_name[:-4]

#     if not repo_name:
#         raise ValueError("Repository name cannot be empty.")

#     # Step 3: Build the destination path
#     destination = (
#         Path(clone_dir).expanduser().resolve() / owner / repo_name
#     )
#     destination.parent.mkdir(parents=True, exist_ok=True)

#     # Step 4: Handle an existing destination
#     if destination.exists():
#         try:
#             existing_repo = Repo(destination, search_parent_directories=False)

#             if existing_repo.bare:
#                 raise ValueError(
#                     f"Existing repository at {destination} is bare."
#                 )

#             # Verify that the existing clone points to the requested repo.
#             expected_url = f"https://github.com/{owner}/{repo_name}.git"
#             remotes = [
#                 remote.url for remote in existing_repo.remotes
#             ]

#             if not any(
#                 url.rstrip("/").removesuffix(".git").lower()
#                 == expected_url.removesuffix(".git").lower()
#                 for url in remotes
#             ):
#                 raise FileExistsError(
#                     f"Destination exists but is not a clone of "
#                     f"{owner}/{repo_name}: {destination}"
#                 )

#             # Reuse the existing clone; do not clone again.
#             return destination

#         except (InvalidGitRepositoryError, NoSuchPathError):
#             raise FileExistsError(
#                 f"Destination already exists and is not a valid Git "
#                 f"repository: {destination}"
#             ) from None

#     # Step 5: Clone the repository
#     try:
#         Repo.clone_from(repo_url, destination)
#     except GitCommandError as exc:
#         # Remove a partial clone created by a failed Git operation.
#         if destination.exists():
#             import shutil
#             shutil.rmtree(destination, ignore_errors=True)

#         raise RuntimeError(
#             "Failed to clone the GitHub repository. Check the URL, "
#             "repository access, Git installation, and authentication."
#         ) from exc

#     return destination


from pathlib import Path
from urllib.parse import urlparse
import shutil

import logfire
from git import (
    Repo,
    GitCommandError,
    InvalidGitRepositoryError,
    NoSuchPathError,
)


# Keep cloned repositories outside the application's source/current working
# directory so that development reloaders (for example, Uvicorn --reload)
# do not restart the server when repository files are created or changed.
#
# Override this location with the REPOSITORY_CLONE_ROOT environment variable
# if needed. The default is a sibling directory next to the current working
# directory, rather than a folder inside it.
import os

_DEFAULT_CLONE_ROOT = Path.cwd().resolve().parent / "gitcontext_repositories"
CLONE_ROOT = Path(
    os.getenv("REPOSITORY_CLONE_ROOT", str(_DEFAULT_CLONE_ROOT))
).expanduser().resolve()


@logfire.instrument("Clone GitHub repository")
def clone_repository(
    repo_url: str,
    clone_dir: str | Path | None = None,
) -> Path:
    """Clone a GitHub repository and return its local path.

    Repositories are stored outside the application's working directory by
    default to avoid triggering file-watch reloads during development.
    Set REPOSITORY_CLONE_ROOT or pass clone_dir to choose another root.
    """
    # Step 1: Validate the URL.
    if not isinstance(repo_url, str) or not repo_url.strip():
        raise ValueError("Repository URL cannot be empty.")

    repo_url = repo_url.strip()
    parsed_url = urlparse(repo_url)

    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname != "github.com"
        or parsed_url.port is not None
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError(
            "Provide a valid HTTPS GitHub repository URL, "
            "such as https://github.com/owner/repository."
        )

    # Step 2: Extract owner and repository name.
    parts = parsed_url.path.strip("/").split("/")
    if (
        len(parts) != 2
        or not all(parts)
        or parts[0] in {".", ".."}
        or parts[1] in {".", ".."}
    ):
        raise ValueError(
            "URL must have the format "
            "https://github.com/owner/repository."
        )

    owner, repo_name = parts
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    if not repo_name:
        raise ValueError("Repository name cannot be empty.")

    # Step 3: Build the destination path.
    # Use an explicit clone_dir when provided; otherwise use the external
    # clone root so repository changes do not occur inside the app directory.
    clone_root = (
        Path(clone_dir).expanduser().resolve()
        if clone_dir is not None
        else CLONE_ROOT
    )
    destination = clone_root / owner / repo_name

    logfire.info(
        "Preparing repository clone destination",
        owner=owner,
        repo_name=repo_name,
        clone_root=str(clone_root),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Step 4: Handle an existing destination.
    if destination.exists():
        try:
            existing_repo = Repo(destination, search_parent_directories=False)

            if existing_repo.bare:
                raise ValueError(
                    f"Existing repository at {destination} is bare."
                )

            expected_url = f"https://github.com/{owner}/{repo_name}.git"
            remotes = [remote.url for remote in existing_repo.remotes]

            if not any(
                url.rstrip("/").removesuffix(".git").lower()
                == expected_url.removesuffix(".git").lower()
                for url in remotes
            ):
                raise FileExistsError(
                    f"Destination exists but is not a clone of "
                    f"{owner}/{repo_name}: {destination}"
                )

            logfire.info(
                "Reusing existing repository clone",
                repo_name=repo_name,
                destination=str(destination),
            )
            return destination

        except (InvalidGitRepositoryError, NoSuchPathError):
            raise FileExistsError(
                f"Destination already exists and is not a valid Git "
                f"repository: {destination}"
            ) from None

    # Step 5: Clone the repository.
    try:
        with logfire.span(
            "Git clone",
            repo_name=repo_name,
            owner=owner,
            destination=str(destination),
        ):
            Repo.clone_from(repo_url, destination)
    except GitCommandError as exc:
        # Remove only the partial destination created by this failed clone.
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)

        logfire.exception(
            "GitHub repository clone failed",
            repo_name=repo_name,
            owner=owner,
        )
        raise RuntimeError(
            "Failed to clone the GitHub repository. Check the URL, "
            "repository access, Git installation, and authentication."
        ) from exc

    logfire.info(
        "Repository cloned successfully",
        repo_name=repo_name,
        owner=owner,
        destination=str(destination),
    )
    return destination
