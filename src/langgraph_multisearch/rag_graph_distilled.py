from langgraph.graph import StateGraph, END
import wandb

from metadata import extract_metadata, extract_metadata_compare
from llm_config import get_llm
from prompt_template import prompt
from langgraph_multisearch.rag_evaluate import node_score
from langgraph_multisearch.adaptive_retreival import node_retrieve
from langgraph_multisearch.rag_state import RAGState
from compare_judge_template import classify_question_with_llm
from langgraph_multisearch.distilled_prompt import distill_context

import rag_logger as logger
# --------------------------------------------------------
# 1) 비교 여부 판단 노드
# --------------------------------------------------------
def node_compare_judge(state: RAGState):
    question = state["question"]
    vector_store = state["vector_store"]

    parsed = classify_question_with_llm(question)
    q_type = parsed.get("질문유형", "단일")
    compare_keys = parsed.get("비교_사업", []) if q_type == "비교" else []

    full_vs = vector_store.get(include=["metadatas", "documents"], limit=999999)

    print(f"[COMPARE_JUDGE] type={q_type}, compare_keys={compare_keys}")

    return {
        "is_compare": (q_type == "비교"),
        "compare_keys": compare_keys,
        "full_vs": full_vs
    }


# --------------------------------------------------------
# 2) 메타데이터 추출
# --------------------------------------------------------
def node_extract_metadata(state: RAGState):
    docs = state["docs"]

    if state["is_compare"]:
        metadata = extract_metadata_compare(state, docs, state["compare_keys"])
    else:
        metadata = extract_metadata(docs)

    return {"metadata": metadata}

def node_distill(state: RAGState):
    question = state["question"]
    meta = state["metadata"]
    
    if meta.get("mode") == "compare":
        item_a, item_b = meta["items"]
        distilled_a = distill_context(question, item_a["context"])
        distilled_b = distill_context(question, item_b["context"])

        logger.log({
            "distilled_A": distilled_a,
            "distilled_B": distilled_b
        })

        return {"distilled_context": f"[A]\n{distilled_a}\n\n[B]\n{distilled_b}"}

    else:
        distilled = distill_context(question, meta["context"])

        logger.log({"distilled_single": distilled})

        return {"distilled_context": distilled}


# --------------------------------------------------------
# 3) 프롬프트 생성
# --------------------------------------------------------
def node_build_prompt(state: RAGState):
    meta = state["metadata"]
    question = state["question"]

    # 비교 모드일 때: 문서 2개의 context를 이어붙임
    if meta.get("mode") == "compare":
        # meta["items"] = [문서A, 문서B]
        item_a, item_b = meta["items"]

        merged_context = f"[문서A]\n{item_a['context']}\n\n[문서B]\n{item_b['context']}"
        merged_source = f"{item_a['source']} / {item_b['source']}"
        merged_org = f"{item_a['org']} / {item_b['org']}"
        merged_category = f"{item_a['category']} / {item_b['category']}"
        merged_budget = f"{item_a['budget']} / {item_b['budget']}"
        merged_open_date = f"{item_a['open_date']} / {item_b['open_date']}"
        merged_end_date = f"{item_a['end_date']} / {item_b['end_date']}"

        filled = prompt.format(
            question=question,
            source=merged_source,
            org=merged_org,
            category=merged_category,
            budget=merged_budget,
            open_date=merged_open_date,
            end_date=merged_end_date,
            context=state["distilled_context"]
        )
        return {"prompt": filled}

    # 단일 모드
    filled = prompt.format(
        question=question,
        source=meta.get("source", ""),
        org=meta.get("org", ""),
        category=meta.get("category", ""),
        budget=meta.get("budget", ""),
        open_date=meta.get("open_date", ""),
        end_date=meta.get("end_date", ""),
        context=state["distilled_context"]
    )
    return {"prompt": filled}

# --------------------------------------------------------
# 4) LLM 생성
# --------------------------------------------------------
def node_llm(state: RAGState):
    llm = get_llm('llm')
    raw = llm.invoke(state['prompt'])
    answer = raw.content if hasattr(raw, "content") else str(raw)

    logger.log({
        'final_answer': answer,
        'prompt_length': len(state['prompt']),
        'answer_length': len(answer)
    })

    return {"answer": answer}


# --------------------------------------------------------
# 5) 스코어링
# --------------------------------------------------------
def route_after_scoring(state: RAGState):
    score = state["score"]
    print(f"[DECISION] score={score}")
    return "good"


# --------------------------------------------------------
# 6) 그래프 구축
# --------------------------------------------------------
def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("compare_judge", node_compare_judge)
    graph.add_node("retrieve_docs", node_retrieve)
    graph.add_node("extract_meta", node_extract_metadata)
    graph.add_node("distilled", node_distill)
    graph.add_node("build_prompt", node_build_prompt)
    graph.add_node("generate", node_llm)
    graph.add_node("score", node_score)

    graph.set_entry_point("compare_judge")

    graph.add_edge("compare_judge", "retrieve_docs")
    graph.add_edge("retrieve_docs", "extract_meta")
    graph.add_edge("extract_meta", "distilled")
    graph.add_edge("distilled", "build_prompt")
    graph.add_edge("build_prompt", "generate")
    graph.add_edge("generate", "score")

    graph.add_conditional_edges(
        "score",
        route_after_scoring,
        {"good": END},
    )

    return graph.compile()


# --------------------------------------------------------
# 7) 실행
# --------------------------------------------------------
def run_rag_graph(question, vector_store, retriever):
    print(f"입력 질문: {question}")
    app = build_graph()
    result = app.invoke({
        "question": question,
        "vector_store": vector_store,
        "retriever": retriever,
        "retry": 0
    })
    return result["answer"]
