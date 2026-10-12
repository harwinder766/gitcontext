from pydriller import Repository

url = "C:/Users/harwi/rag_project"

for commit in Repository(url, only_in_branch=['feat-vector-database']).traverse_commits():
    print(commit.hash)
    print(commit.msg)
    print(commit.author.name)
    print(commit.author.email)
    print(commit.branches)

    for file in commit.modified_files:
        print(file.new_path or file.old_path)
        print(file.diff)

