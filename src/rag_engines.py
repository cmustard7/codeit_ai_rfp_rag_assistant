from langchain.rag_chain import run_chain as run_langchain
from langgraph_base.rag_graph import run_rag_graph as run_langgraph_base
from langgraph_multisearch.rag_graph import run_rag_graph as run_langgraph_multi

ENGINES = {
    "langchain": run_langchain,
    "langgraph_base": run_langgraph_base,
    "langgraph_multisearch": run_langgraph_multi
}
