from langgraph.graph import StateGraph, END

from src.metadata import extract_metadata
from src.prompt_template import prompt
from src.llm_config import llm
from src.rag_evaluate import node_score
from src.adaptive_retreival import node_retrieve
from src.rag_state import RAGState

"""
def node_retrieve(state: RAGState):
    docs = find_docs_by_question(
        {'question': state['question']},
        state["vector_store"],
        state["retriever"],
        top_n=state.get('retry', 0) + 1
    )

    return {
        'docs': docs,
        'retry': state.get('retry', 0) + 1
    }
"""

def node_extract_metadata(state: RAGState):
    metadata = extract_metadata(state['docs'])
    return {'metadata': metadata}

def node_build_prompt(state: RAGState):
    formatted_prompt = prompt.format(
        context=state['metadata']['context'],
        question=state['question'],
        source=state['metadata']['source'],
        org=state['metadata']['org'],
        category=state['metadata']['category'],
        budget=state['metadata']['budget'],
        open_date=state['metadata']['open_date'],
        end_date=state['metadata']['end_date']
    )
    
    return {'prompt': formatted_prompt}

def node_llm(state: RAGState):
    raw = llm.invoke(state['prompt'])
    answer = raw.content if hasattr(raw, "content") else str(raw)
    return {'answer': answer}


def route_after_scoring(state: RAGState):
    score = state['score']
    retry = state.get('retry', 0)

    print(f"[DECISION] score={score}, retry={retry}")

    if score >= 0.75:
        print(" → GOOD (종료)")
        return 'good'

    if retry >= 5:
        print(" → BAD but retry limit reached (종료)")
        return 'good'

    print(" → BAD (재검색)")
    return 'bad'


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node('retrieve_docs', node_retrieve)
    graph.add_node('extract_meta', node_extract_metadata)
    graph.add_node('build_prompt', node_build_prompt)
    graph.add_node('generate', node_llm)
    graph.add_node('score', node_score)

    graph.set_entry_point('retrieve_docs')

    graph.add_edge('retrieve_docs', 'extract_meta')
    graph.add_edge('extract_meta', 'build_prompt')
    graph.add_edge('build_prompt', 'generate')
    graph.add_edge('generate', 'score')

    graph.add_conditional_edges(
        'score',
        route_after_scoring,
        {
            'good': END,
            'bad': 'retrieve_docs'
        }
    )

    return graph.compile()

def run_rag_graph(question, vector_store, retriever):
    print(f'입력 질문: {question}')
    app = build_graph()
    result = app.invoke({
        "question": question,
        "vector_store": vector_store,
        "retriever": retriever,
        "retry": 0
    })
    return result["answer"]