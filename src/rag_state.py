from typing import TypedDict

class RAGState(TypedDict):
    vector_store: any
    retriever: any
    question: str
    docs: list
    metadata: dict
    context: str
    answer: str
    score: float
    retry: int
    prompt: str
    refined_query: str