from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

from src.metadata import extract_metadata
from src.retrieval import find_docs_by_question
from src.prompt_template import prompt
from src.llm_config import llm

def run_chain(vector_store, retriever, question):
    metadata_chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(lambda x: {"docs": find_docs_by_question(x["question"], vector_store, retriever)})
        | RunnableLambda(lambda x: extract_metadata(x["docs"]))
    )

    rag_chain = (
        {
            "context": metadata_chain | itemgetter("context"),
            "source": metadata_chain | itemgetter("source"),
            "org": metadata_chain | itemgetter("org"),
            "category": metadata_chain | itemgetter("category"),
            "budget": metadata_chain | itemgetter("budget"),
            "open_date": metadata_chain | itemgetter("open_date"),
            "end_date": metadata_chain | itemgetter("end_date"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke({"question": question})