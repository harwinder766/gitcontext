from langgraph.checkpoint.memory import MemorySaver

from gitcontext.generation.classify_intent import PlannerDecision
from gitcontext.generation.normal_message import Answer
from gitcontext.graph.graph import build_graph


class FakeResponse:
    answer = "Repository answer"
    sources = []


class FakeRAG:
    calls = []

    def __init__(self, **kwargs):
        self.__class__.calls.append(kwargs)

    def generate_response(self):
        return FakeResponse()


class FakeLLM:
    def with_structured_output(self, schema):
        class Structured:
            def invoke(_self, messages):
                if schema == PlannerDecision:
                    query = messages.get("query", "") if isinstance(messages, dict) else str(messages)
                    if any(w in query.lower() for w in ["hello", "hi", "hey"]):
                        return PlannerDecision(intent="conversational", search_query=query)
                    return PlannerDecision(intent="repository", search_query=f"rewritten: {query}")
                elif schema == Answer:
                    return Answer(answer="Conversational greeting response")
                return schema()

        return Structured()


def make_graph(llm=None):
    FakeRAG.calls = []
    return build_graph(
        llm=llm or FakeLLM(),
        embedding_model=object(),
        rag_factory=FakeRAG,
        checkpointer=MemorySaver(),
    )


def test_greeting_skips_rag():
    graph = make_graph()

    result = graph.invoke(
        {
            "thread_id": "thread-greeting",
            "repo_name": "owner/project",
            "question": "Hello!",
        },
        config={"configurable": {"thread_id": "thread-greeting"}},
    )

    assert result["intent"] in ("conversational", "greeting")
    assert result["sources"] == []
    assert FakeRAG.calls == []


def test_repository_question_uses_repository_filter_and_history():
    graph = make_graph()
    config = {"configurable": {"thread_id": "thread-repo"}}

    first = graph.invoke(
        {
            "thread_id": "thread-repo",
            "repo_name": "owner/project",
            "question": "Explain the ingestion pipeline.",
        },
        config=config,
    )
    second = graph.invoke(
        {
            "thread_id": "thread-repo",
            "repo_name": "owner/project",
            "question": "Explain that function in more detail.",
        },
        config=config,
    )

    assert first["intent"] == "repository"
    assert second["intent"] == "repository"
    assert len(FakeRAG.calls) == 2
    assert FakeRAG.calls[1]["filter"] == {"repo_name": "owner/project"}
    assert len(second["messages"]) == 4


def test_threads_do_not_share_messages():
    graph = make_graph()
    graph.invoke(
        {
            "thread_id": "thread-one",
            "repo_name": "owner/project",
            "question": "Explain the retriever.",
        },
        config={"configurable": {"thread_id": "thread-one"}},
    )
    result = graph.invoke(
        {
            "thread_id": "thread-two",
            "repo_name": "owner/project",
            "question": "What did we discuss earlier?",
        },
        config={"configurable": {"thread_id": "thread-two"}},
    )

    # Thread two has only 2 messages (one user, one assistant turn), isolated from thread-one
    assert len(result["messages"]) == 2
