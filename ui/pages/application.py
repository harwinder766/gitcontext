# """GitContext Streamlit page: index repositories and ask questions about them."""

# import os
# import re
# from uuid import uuid4
# from pathlib import Path

# import requests
# import streamlit as st

# st.set_page_config(page_title="GitContext | App", page_icon="🧠", layout="wide")

# API_URL = os.getenv("GITCONTEXT_API_URL", "http://127.0.0.1:8000")
# GITHUB_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+?(?:\.git)?/?$")
# CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# SUGGESTIONS = [
#     "Give me a brief overview of this repository",
#     "What are the main components and how do they fit together?",
#     "What changed in the most recent commits?",
# ]

# # Reuse the landing-page stylesheet if it exists.
# css_file = Path(__file__).parent.parent / "styles.css"
# if css_file.exists():
#     st.markdown(
#         f"<style>{css_file.read_text(encoding='utf-8')}</style>",
#         unsafe_allow_html=True,
#     )


# # -------------------- API HELPERS --------------------

# def call_api(method, path, timeout=60, **kwargs):
#     """Return (response_data, error_message)."""
#     try:
#         response = requests.request(
#             method, f"{API_URL}{path}", timeout=timeout, **kwargs
#         )
#     except requests.exceptions.ConnectionError:
#         return None, f"Cannot reach the API at {API_URL}. Is the backend running?"
#     except requests.exceptions.Timeout:
#         return None, "The request timed out."

#     if response.status_code >= 400:
#         try:
#             message = response.json().get("detail", response.text)
#         except ValueError:
#             message = response.text
#         return None, str(message)

#     try:
#         return response.json(), None
#     except ValueError:
#         return response.text, None


# def load_repositories():
#     data, error = call_api("GET", "/repositories", timeout=120)
#     if error:
#         st.sidebar.error(error)
#         return st.session_state.get("repos", [])
#     return data.get("repositories", [])


# # -------------------- SOURCE DISPLAY --------------------

# def first_value(info, *keys):
#     """Return the first non-empty field."""
#     return next((info[key] for key in keys if info.get(key) not in (None, "")), None)


# def source_info(source):
#     """Flatten metadata when source fields are nested."""
#     if not isinstance(source, dict):
#         return {"title": str(source)}
#     metadata = source.get("metadata")
#     return {**source, **metadata} if isinstance(metadata, dict) else source


# def source_kind(info):
#     kind = str(first_value(info, "document_type", "type") or "").lower()
#     return kind if kind in ("commit", "pull_request") else "code"


# def source_title(info):
#     if source_kind(info) == "pull_request":
#         return f"PR #{info.get('pr_number', '?')}: {info.get('pr_title', '')}".strip()
#     return str(first_value(info, "file_path", "source", "title") or "unknown source")


# def source_link(repo, info):
#     base = f"https://github.com/{repo}"
#     kind = source_kind(info)
#     commit = first_value(info, "commit_hash", "commit")
#     path = first_value(info, "file_path", "source")

#     if info.get("url"):
#         return info["url"]
#     if kind == "pull_request" and info.get("pr_number"):
#         return f"{base}/pull/{info['pr_number']}"
#     if kind == "commit" and commit:
#         return f"{base}/commit/{commit}"
#     if kind == "code" and path:
#         return f"{base}/blob/HEAD/{path}"
#     return None


# def source_details(info):
#     details = []
#     kind = source_kind(info)
#     commit = first_value(info, "commit_hash", "commit")
#     message = first_value(info, "commit_msg", "message")
#     chunk = first_value(info, "chunk_index", "chunk")

#     if kind == "commit" and commit:
#         details.append(f"commit {str(commit)[:7]}")
#     if kind == "commit" and message:
#         details.append(str(message).strip().splitlines()[0])
#     if kind == "pull_request" and info.get("pr_state"):
#         details.append(str(info["pr_state"]))
#     if chunk is not None:
#         details.append(f"chunk {chunk}")
#     return " · ".join(details)


# def render_sources(repo, answer, sources):
#     """Display answer sources, placing cited sources first."""
#     if not sources:
#         return

#     cited = {
#         int(number)
#         for group in CITATION.findall(answer)
#         for number in group.split(",")
#     }
#     rows = []
#     for position, source in enumerate(sources, start=1):
#         info = source_info(source)
#         rows.append((info.get("label", position), info))

#     rows.sort(key=lambda item: (item[0] not in cited, item[0]))

#     with st.expander(f"Sources ({len(rows)})"):
#         for label, info in rows:
#             kind = source_kind(info)
#             icon = {"commit": "🕰️", "pull_request": "🔀"}.get(kind, "📄")
#             title = source_title(info)
#             link = source_link(repo, info)
#             heading = f"[`{title}`]({link})" if link else f"`{title}`"
#             cited_text = " ✅ cited" if label in cited else ""

#             st.markdown(f"**[{label}]** {icon} {heading}{cited_text}")
#             details = source_details(info)
#             if details:
#                 st.caption(details)


# def parse_response(data):
#     """Return answer text and source list from the API response."""
#     if isinstance(data, dict):
#         return data.get("answer") or data.get("response") or "", data.get("sources") or []
#     return str(data), []


# # -------------------- SESSION STATE --------------------

# if "repos" not in st.session_state:
#     with st.spinner("Loading repositories..."):
#         st.session_state.repos = load_repositories()

# if "chats" not in st.session_state:
#     st.session_state.chats = {}
# if "thread_ids" not in st.session_state:
#     st.session_state.thread_ids = {}


# # -------------------- SIDEBAR --------------------

# with st.sidebar:
#     st.header("🧠 GitContext")

#     _, health_error = call_api("GET", "/health", timeout=3)
#     if health_error:
#         st.error(health_error)
#         st.stop()

#     notice = st.session_state.pop("notice", None)
#     if notice:
#         st.success(notice)

#     st.subheader("Index a repository")
#     repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")

#     with st.expander("Options"):
#         max_commits = st.number_input("Max commits", 0, 5000, 50)
#         max_prs = st.number_input("Max pull requests", 0, 200, 50)

#     if st.button("Index repository", type="primary", width="stretch"):
#         repo_url = repo_url.strip()
#         if not GITHUB_URL.match(repo_url):
#             st.error("Enter a URL like https://github.com/owner/repo")
#         else:
#             with st.spinner("Indexing... this can take several minutes."):
#                 data, error = call_api(
#                     "POST",
#                     "/index",
#                     timeout=1800,
#                     json={
#                         "repo_url": repo_url,
#                         "max_commits": int(max_commits),
#                         "max_prs": int(max_prs),
#                     },
#                 )

#             if error:
#                 st.error(error)
#             else:
#                 result = data.get("result")
#                 chunks = result.get("chunks") if isinstance(result, dict) else None
#                 st.session_state.repos = load_repositories()
#                 st.session_state.selected_repo = data["repo_name"]
#                 st.session_state.notice = (
#                     f"Indexed {data['repo_name']}"
#                     + (f" ({chunks} chunks)." if chunks else ".")
#                 )
#                 st.rerun()

#     st.divider()
#     st.subheader("Repository")

#     repos = st.session_state.repos or []
#     repo = None
#     if repos:
#         if st.session_state.get("selected_repo") not in repos:
#             st.session_state.selected_repo = repos[0]
#         repo = st.selectbox("Ask about", repos, key="selected_repo")

#     refresh_col, clear_col = st.columns(2)
#     if refresh_col.button("Refresh", width="stretch"):
#         st.session_state.repos = load_repositories()
#         st.rerun()

#     if clear_col.button("Clear chat", width="stretch") and repo:
#         st.session_state.chats.pop(repo, None)
#         st.session_state.thread_ids.pop(repo, None)
#         st.session_state.thread_ids.pop(repo, None)
#         st.rerun()

#     if repo:
#         with st.popover("Delete this repository", width="stretch"):
#             st.warning(f"Delete all indexed data for {repo}?")
#             if st.button("Yes, delete", type="primary"):
#                 data, error = call_api("DELETE", f"/repositories/{repo}", timeout=120)
#                 if error:
#                     st.error(error)
#                 else:
#                     st.session_state.chats.pop(repo, None)
#                     st.session_state.repos = load_repositories()
#                     st.session_state.notice = (
#                         f"Deleted {repo} ({data.get('deleted_chunks', 0)} chunks)."
#                     )
#                     st.rerun()

#     st.divider()
#     debug = st.toggle("Show raw API response")


# # -------------------- MAIN CHAT AREA --------------------

# if not repo:
#     st.title("Ask your codebase")
#     st.info("Index a GitHub repository from the sidebar to get started.")
#     st.stop()

# st.title("Ask your codebase")
# st.caption(f"Repository: [{repo}](https://github.com/{repo})")

# messages = st.session_state.chats.setdefault(repo, [])
# thread_id = st.session_state.thread_ids.setdefault(repo, str(uuid4()))

# for message in messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if message["role"] == "assistant":
#             render_sources(repo, message["content"], message.get("sources", []))

# suggestion = None
# if not messages:
#     st.write("Try asking:")
#     for column, text in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
#         if column.button(text, key=text, width="stretch"):
#             suggestion = text

# question = st.chat_input("Ask a question about this repository...") or suggestion

# if question:
#     messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             data, error = call_api(
#                 "POST",
#                 "/query",
#                 timeout=180,
#                 json={
#                     "question": question,
#                     "repo_name": repo,
#                     "thread_id": thread_id,
#                 },
#             )

#         if error:
#             st.error(error)
#         else:
#             answer, sources = parse_response(data)
#             st.markdown(answer or "_The API returned no answer._")
#             render_sources(repo, answer, sources)

#             if debug:
#                 with st.expander("Raw API response"):
#                     st.json(data)

#             messages.append(
#                 {"role": "assistant", "content": answer, "sources": sources}
#             )


"""GitContext Streamlit page: index repositories and ask questions about them."""

import os
from pathlib import Path
import re
from uuid import uuid4

import requests
import streamlit as st

st.set_page_config(page_title="GitContext | App", page_icon="🧠", layout="wide")

API_URL = os.getenv("GITCONTEXT_API_URL", "http://127.0.0.1:8000").rstrip("/")
GITHUB_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+?(?:\.git)?/?$")
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SUGGESTIONS = [
    "Give me a brief overview of this repository",
    "What are the main components and how do they fit together?",
    "What changed in the most recent commits?",
]

# Reuse the landing-page stylesheet if it exists.
css_file = Path(__file__).parent.parent / "styles.css"
if css_file.exists():
    st.markdown(
        f"<style>{css_file.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


# -------------------- API HELPERS --------------------

def call_api(method: str, path: str, timeout: int = 60, **kwargs):
    """Return (response_data, error_message)."""
    try:
        response = requests.request(
            method, f"{API_URL}{path}", timeout=timeout, **kwargs
        )
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the API at {API_URL}. Is the backend running?"
    except requests.exceptions.Timeout:
        return None, "The request timed out."
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed: {exc}"

    if response.status_code >= 400:
        try:
            payload = response.json()
            message = payload.get("detail", response.text) if isinstance(payload, dict) else response.text
        except ValueError:
            message = response.text
        return None, str(message)

    try:
        return response.json(), None
    except ValueError:
        return response.text, None


def load_repositories() -> list[str]:
    data, error = call_api("GET", "/repositories", timeout=120)
    if error:
        st.sidebar.error(error)
        return st.session_state.get("repos", [])
    if isinstance(data, dict):
        return data.get("repositories", [])
    return []


# -------------------- SOURCE DISPLAY --------------------

def first_value(info: dict, *keys: str):
    """Return the first non-empty field."""
    return next((info[key] for key in keys if info.get(key) not in (None, "")), None)


def source_info(source) -> dict:
    """Flatten metadata when source fields are nested."""
    if not isinstance(source, dict):
        return {"title": str(source)}
    metadata = source.get("metadata")
    return {**source, **metadata} if isinstance(metadata, dict) else source


def source_kind(info: dict) -> str:
    kind = str(first_value(info, "document_type", "type") or "").lower()
    return kind if kind in ("commit", "pull_request") else "code"


def source_title(info: dict) -> str:
    if source_kind(info) == "pull_request":
        return f"PR #{info.get('pr_number', '?')}: {info.get('pr_title', '')}".strip()
    return str(first_value(info, "file_path", "source", "title") or "unknown source")


def source_link(repo: str, info: dict) -> str | None:
    base = f"https://github.com/{repo}"
    kind = source_kind(info)
    commit = first_value(info, "commit_hash", "commit")
    path = first_value(info, "file_path", "source")

    if info.get("url"):
        return str(info["url"])
    if kind == "pull_request" and info.get("pr_number"):
        return f"{base}/pull/{info['pr_number']}"
    if kind == "commit" and commit:
        return f"{base}/commit/{commit}"
    if kind == "code" and path:
        return f"{base}/blob/HEAD/{path}"
    return None


def source_details(info: dict) -> str:
    details = []
    kind = source_kind(info)
    commit = first_value(info, "commit_hash", "commit")
    message = first_value(info, "commit_msg", "message")
    chunk = first_value(info, "chunk_index", "chunk")

    if kind == "commit" and commit:
        details.append(f"commit {str(commit)[:7]}")
    if kind == "commit" and message:
        details.append(str(message).strip().splitlines()[0])
    if kind == "pull_request" and info.get("pr_state"):
        details.append(str(info["pr_state"]))
    if chunk is not None:
        details.append(f"chunk {chunk}")
    return " · ".join(details)


def render_sources(repo: str, answer: str, sources: list) -> None:
    """Display answer sources, placing cited sources first."""
    if not sources:
        return

    cited = {
        int(number.strip())
        for group in CITATION.findall(answer or "")
        for number in group.split(",")
        if number.strip().isdigit()
    }
    rows = []
    for position, source in enumerate(sources, start=1):
        info = source_info(source)
        rows.append((info.get("label", position), info))

    rows.sort(key=lambda item: (item[0] not in cited, item[0]))

    with st.expander(f"Sources ({len(rows)})"):
        for label, info in rows:
            kind = source_kind(info)
            icon = {"commit": "🕰️", "pull_request": "🔀"}.get(kind, "📄")
            title = source_title(info)
            link = source_link(repo, info)
            heading = f"[`{title}`]({link})" if link else f"`{title}`"
            cited_text = " ✅ cited" if label in cited else ""

            st.markdown(f"**[{label}]** {icon} {heading}{cited_text}")
            details = source_details(info)
            if details:
                st.caption(details)


def parse_response(data) -> tuple[str, list]:
    """Return answer text and source list from the API response."""
    if isinstance(data, dict):
        return data.get("answer") or data.get("response") or "", data.get("sources") or []
    return str(data), []


# -------------------- SESSION STATE --------------------

if "repos" not in st.session_state:
    with st.spinner("Loading repositories..."):
        st.session_state.repos = load_repositories()

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "thread_ids" not in st.session_state:
    st.session_state.thread_ids = {}


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("🧠 GitContext")

    _, health_error = call_api("GET", "/health", timeout=3)
    if health_error:
        st.error(health_error)
        st.stop()

    notice = st.session_state.pop("notice", None)
    if notice:
        st.success(notice)

    st.subheader("Index a repository")
    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")

    with st.expander("Options"):
        max_commits = st.number_input("Max commits", 0, 5000, 50)
        max_prs = st.number_input("Max pull requests", 0, 200, 50)

    if st.button("Index repository", type="primary", width="stretch"):
        repo_url = repo_url.strip()
        if not GITHUB_URL.match(repo_url):
            st.error("Enter a URL like https://github.com/owner/repo")
        else:
            with st.spinner("Indexing... this can take several minutes."):
                data, error = call_api(
                    "POST",
                    "/index",
                    timeout=1800,
                    json={
                        "repo_url": repo_url,
                        "max_commits": int(max_commits),
                        "max_prs": int(max_prs),
                    },
                )

            if error:
                st.error(error)
            else:
                result = data.get("result")
                chunks = result.get("chunks") if isinstance(result, dict) else None
                st.session_state.repos = load_repositories()
                st.session_state.selected_repo = data["repo_name"]
                st.session_state.notice = (
                    f"Indexed {data['repo_name']}"
                    + (f" ({chunks} chunks)." if chunks else ".")
                )
                st.rerun()

    st.divider()
    st.subheader("Repository")

    repos = st.session_state.repos or []
    repo = None
    if repos:
        if st.session_state.get("selected_repo") not in repos:
            st.session_state.selected_repo = repos[0]
        repo = st.selectbox("Ask about", repos, key="selected_repo")

    refresh_col, clear_col = st.columns(2)
    if refresh_col.button("Refresh", width="stretch"):
        st.session_state.repos = load_repositories()
        st.rerun()

    if clear_col.button("Clear chat", width="stretch") and repo:
        st.session_state.chats.pop(repo, None)
        st.session_state.thread_ids.pop(repo, None)
        st.rerun()

    if repo:
        with st.popover("Delete this repository", width="stretch"):
            st.warning(f"Delete all indexed data for {repo}?")
            if st.button("Yes, delete", type="primary"):
                data, error = call_api("DELETE", f"/repositories/{repo}", timeout=120)
                if error:
                    st.error(error)
                else:
                    st.session_state.chats.pop(repo, None)
                    st.session_state.thread_ids.pop(repo, None)
                    st.session_state.repos = load_repositories()
                    st.session_state.notice = (
                        f"Deleted {repo} ({data.get('deleted_chunks', 0)} chunks)."
                    )
                    st.rerun()

    st.divider()
    debug = st.toggle("Show raw API response")


# -------------------- MAIN CHAT AREA --------------------

if not repo:
    st.title("Ask your codebase")
    st.info("Index a GitHub repository from the sidebar to get started.")
    st.stop()

st.title("Ask your codebase")
st.caption(f"Repository: [{repo}](https://github.com/{repo})")

messages = st.session_state.chats.setdefault(repo, [])
thread_id = st.session_state.thread_ids.setdefault(repo, str(uuid4()))

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(repo, message["content"], message.get("sources", []))

suggestion = None
if not messages:
    st.write("Try asking:")
    for column, text in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
        if column.button(text, key=text, width="stretch"):
            suggestion = text

question = st.chat_input("Ask a question about this repository...") or suggestion

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data, error = call_api(
                "POST",
                "/query",
                timeout=180,
                json={
                    "question": question,
                    "repo_name": repo,
                    "thread_id": thread_id,
                },
            )

        if error:
            st.error(error)
        else:
            answer, sources = parse_response(data)
            st.markdown(answer or "_The API returned no answer._")
            render_sources(repo, answer, sources)

            if debug:
                with st.expander("Raw API response"):
                    st.json(data)

            # Only persist the turn to chat history once the request succeeds
            messages.append({"role": "user", "content": question})
            messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )