import pytest

from gitcontext.ingestion.pipeline import extract_github_repo


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/owner/project", "owner/project"),
        ("https://github.com/owner/project.git", "owner/project"),
        ("https://github.com/owner/project/", "owner/project"),
    ],
)
def test_extract_github_repo(url, expected):
    assert extract_github_repo(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/owner/project",
        "https://github.com/owner/project?tab=readme",
        "https://gitlab.com/owner/project",
        "https://github.com/owner",
    ],
)
def test_extract_github_repo_rejects_invalid_urls(url):
    with pytest.raises(ValueError):
        extract_github_repo(url)
