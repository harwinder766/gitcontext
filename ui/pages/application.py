"""GitContext app page: index a GitHub repository, then ask questions about it.
 
Put this file at pages/application.py (the page that home.py links to).
It talks to the FastAPI backend: /health, /index, /repositories, /query, DELETE /repositories/{repo}.
Needs: pip install streamlit requests
"""
 
import os
import re
from pathlib import Path
 
import requests
import streamlit as st
 
st.set_page_config(page_title="GitContext | App", page_icon="🧠", layout="wide")
 
API_URL = os.getenv("GITCONTEXT_API_URL", "http://127.0.0.1:8000")
GITHUB_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+?(?:\.git)?/?$")
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")   # matches [3] and [3,4] in the answer text
 
SUGGESTIONS = [
    "Give me a brief overview of this repository",
    "What are the main components and how do they fit together?",
    "What changed in the most recent commits?",
]
 
# reuse the landing page styles if the file is there
css_file = Path(__file__).parent.parent / "styles.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
 
 
# -------------------- API HELPERS --------------------
 
def call_api(method: str, path: str, timeout: int = 60, **kwargs):
    """Call the GitContext API. Returns (data, error_message)."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the API at {API_URL}. Is the backend running?"
    except requests.exceptions.Timeout:
        return None, "The request timed out."
 
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, str(detail)
 
    return response.json(), None
 
 
def load_repositories() -> list[str]:
    data, error = call_api("GET", "/repositories", timeout=120)
    if error:
        st.sidebar.error(error)
        return st.session_state.get("repos") or []
    return data["repositories"]
 
 
# -------------------- SOURCE HELPERS --------------------
 
def pick(info: dict, *keys):
    """Return the first value that exists under any of the keys."""
    for key in keys:
        if info.get(key) not in (None, ""):
            return info[key]
    return None
 
 
def as_info(source) -> dict:
    """Flatten a source into one dict (some APIs nest the fields under 'metadata')."""
    if not isinstance(source, dict):
        return {"title": str(source)}
    extra = source.get("metadata")
    return {**source, **extra} if isinstance(extra, dict) else source
 
 
def source_kind(info: dict) -> str:
    kind = str(pick(info, "document_type", "type") or "").lower()
    if kind == 'pull_request':
        return "pull_request"
    if kind == "commit":
        return "commit"
    return "code"
 
 
def source_title(info: dict) -> str:
    if source_kind(info) == "pull_request":
        return f"PR #{info.get('pr_number', '?')}: {info.get('pr_title', '')}".strip()
    return str(pick(info, "file_path", "source", "title") or "unknown source")
 
 
def source_link(repo: str, info: dict) -> str | None:
    base = f"https://github.com/{repo}"
    kind = source_kind(info)
    commit = pick(info, "commit_hash", "commit")
    path = pick(info, "file_path", "source")
 
    if info.get("url"):
        return info["url"]
    if kind == "pull_request" and info.get("pr_number"):
        return f"{base}/pull/{info['pr_number']}"
    if kind == "commit" and commit:
        return f"{base}/commit/{commit}"
    if kind == "code" and path:
        return f"{base}/blob/HEAD/{path}"
    return None
 
 
def source_details(info: dict) -> str:
    parts = []
    commit = pick(info, "commit_hash", "commit")
    message = pick(info, "commit_msg", "message")
 
    if source_kind(info) == "commit" and commit:
        parts.append(f"commit {str(commit)[:7]}")
    if source_kind(info) == "commit" and message:
        parts.append(str(message).strip().splitlines()[0])
    if source_kind(info) == "pull_request" and info.get("pr_state"):
        parts.append(str(info["pr_state"]))
    chunk = pick(info, "chunk_index", "chunk")
    if chunk is not None:
        parts.append(f"chunk {chunk}")
    return " · ".join(parts)
 
 
def render_sources(repo: str, answer: str, sources: list) -> None:
    """Show the sources, with the ones cited in the answer first."""
    if not sources:
        return
 
    cited = {int(n) for group in CITATION.findall(answer) for n in group.split(",")}
 
    rows = []
    for position, source in enumerate(sources, start=1):
        info = as_info(source)
        rows.append((info.get("label", position), info))
    rows.sort(key=lambda row: (row[0] not in cited, row[0]))   # cited sources first
 
    with st.expander(f"Sources ({len(rows)})"):
        for label, info in rows:
            icon = {"commit": "🕰️", "pull_request": "🔀"}.get(source_kind(info), "📄")
            title = source_title(info)
            link = source_link(repo, info)
            heading = f"[`{title}`]({link})" if link else f"`{title}`"
            used = " ✅ cited" if label in cited else ""
 
            st.markdown(f"**[{label}]** {icon} {heading}{used}")
            details = source_details(info)
            if details:
                st.caption(details)
 
 
def parse_response(data) -> tuple[str, list]:
    """Return (answer, sources). If your RAG response uses other keys, change them here."""
    if isinstance(data, dict):
        answer = data.get("answer") or data.get("response") or ""
        return answer, data.get("sources") or []
    return str(data), []
 
 
# -------------------- SESSION STATE --------------------
 
if "repos" not in st.session_state:
    with st.spinner("Loading repositories..."):
        st.session_state.repos = load_repositories()
if "chats" not in st.session_state:
    st.session_state.chats = {}      # repo name -> list of chat messages
 
 
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
 
    # ---- index a repository ----
    st.subheader("Index a repository")
    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")
 
    with st.expander("Options"):
        max_commits = st.number_input("Max commits", min_value=0, max_value=5000, value=50)
        max_prs = st.number_input("Max pull requests", min_value=0, max_value=200, value=50)
 
    if st.button("Index repository", type="primary", width="stretch"):
        if not GITHUB_URL.match(repo_url.strip()):
            st.error("Enter a URL like https://github.com/owner/repo")
        else:
            with st.spinner("Indexing... this can take several minutes."):
                data, error = call_api(
                    "POST",
                    "/index",
                    timeout=1800,
                    json={
                        "repo_url": repo_url.strip(),
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
                    f"Indexed {data['repo_name']}" + (f" ({chunks} chunks)." if chunks else ".")
                )
                st.rerun()
 
    st.divider()
 
    # ---- choose a repository ----
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
                    st.session_state.repos = load_repositories()
                    st.session_state.notice = f"Deleted {repo} ({data.get('deleted_chunks', 0)} chunks)."
                    st.rerun()
 
    st.divider()
    debug = st.toggle("Show raw API response")
 
 
# -------------------- MAIN AREA --------------------
 
if not repo:
    st.title("Ask your codebase")
    st.info("Index a GitHub repository from the sidebar to get started.")
    st.stop()
 
st.title("Ask your codebase")
st.caption(f"Repository: [{repo}](https://github.com/{repo})")
 
messages = st.session_state.chats.setdefault(repo, [])
 
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(repo, message["content"], message.get("sources", []))
 
# suggestion buttons, only while the chat is empty
suggestion = None
if not messages:
    st.write("Try asking:")
    for column, text in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
        if column.button(text, key=text, width="stretch"):
            suggestion = text
 
question = st.chat_input("Ask a question about this repository...") or suggestion
 
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    with st.chat_message("assistant"):
        with st.spinner("Searching the repository..."):
            data, error = call_api(
                "POST", "/query", timeout=180, json={"question": question, "repo_name": repo}
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
            messages.append({"role": "assistant", "content": answer, "sources": sources})
