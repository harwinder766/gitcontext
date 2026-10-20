# # from pydriller import Repository

# # url = "C:/Users/harwi/rag_project"

# # for commit in Repository(url, only_in_branch=['feat-vector-database']).traverse_commits():
# #     print(commit.hash)
# #     print(commit.msg)
# #     print(commit.author.name)
# #     print(commit.author.email)
# #     print(commit.branches)

# #     for file in commit.modified_files:
# #         print(file.new_path or file.old_path)
# #         print(file.diff)

# from github import Github, Auth
# import os
# from dotenv import load_dotenv
# load_dotenv()

# GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")
# if not GITHUB_API_TOKEN:
#     raise ValueError("GITHUB_API_TOKEN is not set in the environment.")

# g = Github(auth=Auth.Token(GITHUB_API_TOKEN))
# g.get_user().login

# repo = g.get_repo("harwinder766/gitcontext")
# print(f"Repository: {repo.name}")

# repositories = g.search_repositories(query='language:python')
# for repo in repositories:
#    print(repo)

# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def status():
#     return {"status": "ok"}

# @app.get('/about')
# def about():
#     return {"message": "This is a FastAPI application."}

import ast

source = """
class Name:
    def get_name(self):
        return "Alex"

class Solution:
    '''Given a binary tree, find the maximum path sum.'''
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        
        m_sum = [float('-inf')]
        self.calc_sum(root, m_sum)
        return m_sum[0]

    def calc_sum(self, node, m_sum):
        if node is None:
            return 0
        l_sum = max(0, self.calc_sum(node.left,m_sum))
        r_sum = max(0, self.calc_sum(node.right,m_sum))
    
        m_sum[0] = max(m_sum[0], l_sum + r_sum + node.val)
        return max(l_sum, r_sum) + node.val
    """

tree = ast.parse(source)
lines = source.splitlines()

for node in tree.body:
    if isinstance(node, ast.ClassDef):
        # print(f"Class: {node.name}")
        for sub_node in node.body:
            if isinstance(sub_node, ast.FunctionDef):
                start = sub_node.lineno - 1
                end = sub_node.end_lineno

                chunk_text = "\n".join(lines[start:end])

                print(f"Class: {node.name}, Function: {sub_node.name}")
                print(chunk_text)
                print("-" * 30)
        # start = node.lineno - 1
        # end = node.end_lineno

        # chunk_text = "\n".join(lines[start:end])

        # print(f"Class: {node.name}")
        # print(chunk_text)
        # print("-" * 30)

# tree = ast.parse(code)
# print(ast.dump(tree, indent=4))

# for node in ast.walk(tree):
#     if isinstance(node, ast.ClassDef):
#         print(f"Class name: {node.name}")
#         # print(f"Arguments: {[arg.arg for arg in node.args.args]}")
#         # print(f"Return type: {ast.dump(node.returns) if node.returns else 'None'}")
#         print(f"Base classes: {[ast.dump(base) for base in node.bases]}")
#         print(f"Is async: {isinstance(node, ast.AsyncFunctionDef)}")
#         print(f"Docstring: {ast.get_docstring(node)}")
#         print(f"Start line: {node.lineno}, End line: {node.end_lineno}")
#         print(f"Decorators: {[ast.dump(decorator) for decorator in node.decorator_list]}")

# for node in tree.body:
#     print(f"Node type: {type(node).__name__}"

# source = """
# def add(a, b):
#     result = a + b
#     return result

# def multiply(a, b):
#     return a * b
# """
# tree = ast.parse(source)

# for node in tree.body:
#     if isinstance(node, ast.FunctionDef):
#         chunk_text = ast.get_source_segment(source, node)

#         print(f"Function: {node.name}")
#         print(chunk_text)
#         print("-" * 30)

# import ast

# source = """
# def add(a, b):
#     result = a + b
#     return result

# def multiply(a, b):
#     return a * b
# """

# tree = ast.parse(source)
# lines = source.splitlines()

# for node in tree.body:
#     if isinstance(node, ast.FunctionDef):
#         start = node.lineno - 1
#         end = node.end_lineno

#         chunk_text = "\n".join(lines[start:end])

#         print(f"Function: {node.name}")
#         print(chunk_text)
#         print("-" * 30)