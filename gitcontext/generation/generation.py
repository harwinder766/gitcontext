import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sympy import python

from gitcontext.retrieval.retriver import Retriever

load_dotenv()

class SourceReference(BaseModel):
    """Metadata about a source document."""

    file_path: str = Field(..., description="Relative path of the source file.")
    absolute_path: str = Field(..., description="Absolute path of the source file.")
    size_bytes: int = Field(..., description="Size of the source file in bytes.")
    extension: str = Field(..., description="File extension of the source file.")
    repo_name: str = Field(..., description="Name of the repository containing the source file.")
    content: str = Field(..., description="Content of the source file.")


class RAGResponse(BaseModel):
    """Response from the RAG system."""

    answer: str = Field(..., description="The generated answer to the query.")
    sources: list[SourceReference] = Field(..., description="List of source documents used to generate the answer.")


class RAG:
    """Retrieve relevant documents and generate an answer."""

    def __init__(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ) -> None:
        if not query.strip():
            raise ValueError("Query must be a non-empty string.")

        api_key = os.getenv("XKIRO_API_KEY")
        base_url = os.getenv("XKIRO_BASE_URL")
        model = os.getenv("XKIRO_MODEL")

        if not all([api_key, base_url, model]):
            raise ValueError("Missing xKiro API configuration in .env.")

        self.query = query
        self.filter = filter
        self.retriever = Retriever(k=k)

        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
            max_tokens=2048,
        )


    def generate_response(self) -> RAGResponse:
        """Retrieve relevant documents and generate a structured answer."""
    
        retrieved_docs = self.retriever.retrieve(
            self.query,
            filter=self.filter,
        )
    
        if not retrieved_docs:
            return RAGResponse(
                answer="No relevant documents found.",
                sources=[],
            )
    
        # Build source metadata using the actual retrieved documents.
        sources = [
            SourceReference(
                file_path=str(doc.metadata.get("source", "unknown")),
                absolute_path=str(
                    doc.metadata.get("absolute_path", "unknown")
                ),
                size_bytes=int(doc.metadata.get("size_bytes", 0)),
                extension=str(doc.metadata.get("extension", "unknown")),
                repo_name=str(doc.metadata.get("repo_name", "unknown")),
                content=doc.page_content,
            )
            for doc in retrieved_docs
        ]
    
        # Give the LLM the retrieved code and source labels.
        context = "\n\n".join(
            f"Source [{i}]: {source.file_path}\n"
            f"Content:\n{source.content}"
            for i, source in enumerate(sources, start=1)
        )
    
        prompt = f"""
    You are GitContext, an assistant that explains code repositories.
    
    Answer the question using the supplied context.
    If the context is insufficient, say so.
    Do not invent code details.
    Cite supporting sources using their labels, such as [1] or [2].
    
    Context:
    {context}
    
    Question:
    {self.query}
    """
    
        structured_llm = self.llm.with_structured_output(RAGResponse)
        response = structured_llm.invoke(prompt)
    
        # Keep metadata from the actual retrieved documents rather than
        # trusting the LLM to generate or reproduce it.
        return RAGResponse(
            answer=response.answer,
            sources=sources,
        )