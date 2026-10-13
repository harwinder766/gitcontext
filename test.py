# from pydriller import Repository

# url = "C:/Users/harwi/rag_project"

# for commit in Repository(url, only_in_branch=['feat-vector-database']).traverse_commits():
#     print(commit.hash)
#     print(commit.msg)
#     print(commit.author.name)
#     print(commit.author.email)
#     print(commit.branches)

#     for file in commit.modified_files:
#         print(file.new_path or file.old_path)
#         print(file.diff)

from github import Github, Auth
import os
from dotenv import load_dotenv
load_dotenv()

GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")
if not GITHUB_API_TOKEN:
    raise ValueError("GITHUB_API_TOKEN is not set in the environment.")

g = Github(auth=Auth.Token(GITHUB_API_TOKEN))
g.get_user().login

repo = g.get_repo("harwinder766/gitcontext")
print(f"Repository: {repo.name}")

repositories = g.search_repositories(query='language:python')
for repo in repositories:
   print(repo)
