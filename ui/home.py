from pathlib import Path
 
import streamlit as st
 
st.set_page_config(
    page_title="GitContext | Understand Any Codebase",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
APP_PAGE = "pages/application.py"   # path is relative to the file you run with `streamlit run`
 
 
# -------------------- HELPERS --------------------
 
def show_html(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)
 
 
def section(title: str, subtitle: str = "") -> None:
    html = f'<div class="section-title">{title}</div>'
    if subtitle:
        html += f'<div class="section-subtitle">{subtitle}</div>'
    show_html(html)
 
 
def card(icon: str, title: str, description: str) -> None:
    show_html(
        '<div class="card feature-card">'
        f'<div class="feature-icon">{icon}</div>'
        f'<div class="feature-title">{title}</div>'
        f'<div class="feature-description">{description}</div>'
        "</div>"
    )
 
 
# -------------------- STYLING (see styles.css next to this file) --------------------
 
css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
show_html(f"<style>{css}</style>")
 
 
# -------------------- NAVIGATION --------------------
 
nav_left, nav_middle, nav_right = st.columns([2, 3, 1])
 
with nav_left:
    show_html('<div class="brand">🧠 GitContext</div>')
 
with nav_middle:
    show_html('<div class="tagline">Repository Intelligence · RAG · Developer Tools</div>')
 
with nav_right:
    st.page_link(APP_PAGE, label="Open App ↗", icon=":material/terminal:", width="stretch")
 
st.divider()
 
 
# -------------------- HERO --------------------
 
hero_left, hero_right = st.columns([1.35, 0.85], gap="large")
 
with hero_left:
    show_html('<div class="hero-label">✦ AI-POWERED CODE UNDERSTANDING</div>')
 
    show_html(
        """
        <div class="hero-title">
            Understand your<br>
            codebase.<br>
            <span class="gradient-text">Ask better questions.</span>
        </div>
        """
    )
 
    show_html(
        '<div class="hero-description">'
        "Index a GitHub repository, ask questions in plain English, and get answers "
        "grounded in its source files, commit history and pull requests, "
        "with the sources used listed alongside each answer."
        "</div>"
    )
 
    st.write("")
    st.page_link(
        APP_PAGE,
        label="Explore Your Repository  →",
        icon=":material/rocket_launch:",
    )
    st.caption("Built for developers navigating unfamiliar or complex codebases.")
 
with hero_right:
    show_html(
        """
        <div class="card example-card">
            <div class="example-label">EXAMPLE QUESTION</div>
            <div class="example-heading">💬 Ask your codebase</div>
            <div class="example-question">
                How does the reranker choose the final documents?
            </div>
            <div style="font-weight:650; margin-bottom:4px;">📚 Answer grounded in</div>
            <span class="source-chip">gitcontext/reranking/reranker.py</span>
            <span class="source-chip">commit 1bae769 · added reranking feature</span>
            <div class="example-note">Example from GitContext's own repository</div>
        </div>
        """
    )
 
 
# -------------------- WHY GITCONTEXT --------------------
 
section(
    "Codebases grow. Understanding takes time.",
    "Understanding an unfamiliar repository usually means jumping between files, "
    "tracing function calls, reading old commits and investigating pull requests. "
    "GitContext brings that context together so you can explore a codebase through questions.",
)
 
problem_left, problem_right = st.columns(2, gap="large")
 
with problem_left:
    card(
        "🔎",
        "The traditional workflow",
        "Search files by hand, trace dependencies, switch between Git history "
        "and source code, and repeat the whole investigation for every new question.",
    )
 
with problem_right:
    card(
        "🧠",
        "The GitContext approach",
        "Index the repository once, ask in natural language, and use the answer "
        "and its source references to decide what to read next.",
    )
 
 
# -------------------- FEATURES --------------------
 
section("Explore the capabilities", "A retrieval-augmented approach to exploring software repositories.")
 
features = [
    ("📦", "Repository indexing",
     "Paste a GitHub URL. GitContext clones the repository, splits files into chunks, "
     "embeds them and stores them in Supabase pgvector."),
    ("💬", "Natural-language questions",
     "Ask about implementation details, project structure or how a feature works, in plain English."),
    ("🕰️", "Commit history",
     "Every commit is indexed per changed file together with its diff, "
     "so you can ask what changed and why."),
    ("🔀", "Pull request context",
     "PR descriptions, discussion comments and review comments are indexed, "
     "so decisions and feedback become searchable."),
    ("🎯", "Hybrid retrieval",
     "BM25 keyword search and vector search are merged with Reciprocal Rank Fusion, "
     "then a cross-encoder reranks the top results."),
    ("📚", "Source references",
     "Each answer lists the files, commits or pull requests it used, so you can verify it."),
]
 
for row_start in range(0, len(features), 3):
    columns = st.columns(3, gap="medium")
    for column, (icon, title, description) in zip(columns, features[row_start:row_start + 3]):
        with column:
            card(icon, title, description)
 
 
# -------------------- HOW IT WORKS --------------------
 
section("How GitContext works", "From a repository URL to an answer grounded in retrieved context.")
 
steps = [
    ("01 / INGEST", "Connect a repository",
     "Provide a GitHub URL. Files, commits and pull requests are fetched."),
    ("02 / INDEX", "Prepare searchable context",
     "Content is split into chunks, embedded, and stored in the vector database."),
    ("03 / RETRIEVE", "Find relevant information",
     "BM25 and vector search find candidates, RRF merges them, and a reranker picks the best chunks."),
    ("04 / GENERATE", "Explore the answer",
     "An LLM answers from the retrieved context only and lists the sources it used."),
]
 
step_columns = st.columns(4, gap="medium")
 
for column, (number, title, description) in zip(step_columns, steps):
    with column:
        show_html(
            '<div class="step">'
            f'<div class="step-number">{number}</div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-description">{description}</div>'
            "</div>"
        )
 
 
# -------------------- TECH STACK --------------------
 
section("Built with", "The tools used across the GitContext pipeline.")
 
technologies = [
    "Python",
    "FastAPI",
    "Streamlit",
    "LangChain",
    "Langgraph",
    "Jina Embeddings",
    "Supabase pgvector",
    "BM25 + RRF",
    "Jina Reranker",
    "PyDriller",
    "LiteLLM Gateway",
    "Pydantic Logfire",
    "PyGithub",
    "Deepeval"
]
 
show_html("".join(f'<span class="tech-pill">{name}</span>' for name in technologies))
 
 
# -------------------- CALL TO ACTION --------------------
 
show_html(
    """
    <div class="cta-panel">
        <div style="font-size:2rem;">🚀</div>
        <div class="cta-title">Ready to explore a codebase?</div>
        <div class="cta-text">
            Choose a GitHub repository, index its context, and start asking questions about the code.
        </div>
    </div>
    """
)
 
st.write("")
 
_, center, _ = st.columns([1, 1.2, 1])
 
with center:
    st.page_link(APP_PAGE, label="Launch GitContext →", icon=":material/arrow_forward:", width="stretch")
 
show_html(
    '<div class="footer">'
    "GitContext · Understand repositories through contextual retrieval.<br>"
    "Answers depend on the indexed content and the quality of the retrieved context."
    "</div>"
)