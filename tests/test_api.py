from fastapi.testclient import TestClient

import gitcontext.api.main as api_main

client = TestClient(api_main.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_reports_vector_store_configuration(monkeypatch):
    monkeypatch.setenv("SUPABASE_DATABASE_URL", "postgresql://example")
    monkeypatch.setattr(api_main, "_get_repo_list", lambda: ["owner/project"])

    response = client.get("/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["vector_store_configured"] is True
    assert payload["repositories"] == ["owner/project"]


def test_query_uses_selected_repository(monkeypatch):
    monkeypatch.setattr(api_main, "_get_repo_list", lambda: ["owner/project"])

    class DummyRAG:
        def __init__(self, query, filter):
            self.query = query
            self.filter = filter

        def generate_response(self):
            return {"answer": "ok", "sources": []}

    monkeypatch.setattr(api_main, "RAG", DummyRAG)

    response = client.post(
        "/query",
        json={"question": "What changed?", "repo_name": "owner/project"},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "ok"
    assert response.json()["sources"] == []


def test_index_route_sets_status_and_returns_repo_name(monkeypatch):
    monkeypatch.setattr(api_main, "extract_github_repo", lambda url: "owner/project")

    def fake_ingest(repo_url, max_commits, max_prs):
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
