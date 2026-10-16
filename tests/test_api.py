from fastapi.testclient import TestClient

import gitcontext.api.main as api_main

client = TestClient(api_main.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_reports_vector_store_configuration(monkeypatch):
    monkeypatch.setenv("SUPABASE_DATABASE_URL", "postgresql://example")

    response = client.get("/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["embedding_model_ready"] is False
    assert payload["vector_store_configured"] is True


def test_query_uses_selected_repository(monkeypatch):
    monkeypatch.setattr(api_main, "_get_repo_list", lambda: ["owner/project"])
    api_main.app.state.embedding_model = object()

    class DummyGraph:
        def invoke(self, state, config):
            assert state["repo_name"] == "owner/project"
            assert config["configurable"]["thread_id"] == "thread-a"
            return {"answer": "ok", "sources": []}

    api_main.app.state.conversation_graph = DummyGraph()

    response = client.post(
        "/query",
        json={
            "question": "What changed?",
            "repo_name": "owner/project",
            "thread_id": "thread-a",
        },
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "ok"
    assert response.json()["sources"] == []
    assert response.json()["thread_id"] == "thread-a"


def test_index_route_sets_status_and_returns_repo_name(monkeypatch):
    monkeypatch.setattr(api_main, "extract_github_repo", lambda url: "owner/project")

    def fake_ingest(repo_url, max_commits, max_prs, embedding_model):
        return {
            "repo_name": "owner/project",
            "commit_documents": 12,
            "pr_documents": 2,
            "chunks": 100,
            "removed_old_chunks": 5,
        }

    monkeypatch.setattr(api_main, "ingest_repository", fake_ingest)

    response = client.post(
        "/index",
        json={"repo_url": "https://github.com/owner/project", "max_commits": 10, "max_prs": 5},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["repo_name"] == "owner/project"
    assert payload["result"]["chunks"] == 100
    assert api_main._INGESTION_STATUS["owner/project"]["status"] == "done"


def test_index_rejects_invalid_github_url():
    response = client.post(
        "/index",
        json={"repo_url": "https://gitlab.com/owner/project"},
    )
    assert response.status_code == 400


def test_query_rejects_malformed_repository_identifier():
    response = client.post(
        "/query",
        json={"question": "What changed?", "repo_name": "project"},
    )
    assert response.status_code == 400
