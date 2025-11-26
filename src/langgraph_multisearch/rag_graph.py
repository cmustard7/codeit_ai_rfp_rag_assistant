# langgraph_multisearch/rag_graph.py
from langgraph.graph import StateGraph, END
import rag_logger as rl
import time

from metadata import extract_metadata, extract_metadata_compare
from llm_config import get_llm, log_usage
from prompt_template import prompt
from langgraph_multisearch.rag_evaluate import node_score
from langgraph_multisearch.adaptive_retreival import node_retrieve
from langgraph_multisearch.rag_state import RAGState
from compare_judge_template import classify_question_with_llm
from langgraph_multisearch.distilled_prompt import distill_context


# ============================================
# 1) 비교 여부 판단 노드 + 로깅
# ============================================
def node_compare_judge(state: RAGState):
    """질문 유형 분석 + 로깅"""
    judge_start = time.time()
    
    question = state["question"]
    vector_store = state["vector_store"]

    # GPT-5-nano로 질문 분류
    parsed = classify_question_with_llm(question)
    q_type = parsed.get("질문유형", "단일")
    compare_keys = parsed.get("비교_사업", []) if q_type == "비교" else []

    # Vector store 전체 로드
    full_vs = vector_store.get(include=["metadatas", "documents"], limit=999999)

    judge_time = time.time() - judge_start

    print(f"[COMPARE_JUDGE] type={q_type}, compare_keys={compare_keys}")

    # 🔥 판별 결과 로깅
    rl.log({
        "compare_judge/time_sec": judge_time,
        "compare_judge/question_type": q_type,
        "compare_judge/is_compare": (q_type == "비교"),
        "compare_judge/num_compare_keys": len(compare_keys),
        "compare_judge/compare_keys": str(compare_keys),
        "compare_judge/total_docs_in_vs": len(full_vs.get("documents", []))
    })

    return {
        "is_compare": (q_type == "비교"),
        "compare_keys": compare_keys,
        "full_vs": full_vs
    }


# ============================================
# 2) 메타데이터 추출 + 로깅
# ============================================
def node_extract_metadata(state: RAGState):
    """메타데이터 추출 + 로깅"""
    extract_start = time.time()
    
    docs = state["docs"]
    is_compare = state["is_compare"]

    if is_compare:
        metadata = extract_metadata_compare(state, docs, state["compare_keys"])
        mode = "compare"
    else:
        metadata = extract_metadata(docs)
        mode = "single"

    extract_time = time.time() - extract_start

    # 🔥 메타데이터 통계 로깅
    if mode == "compare":
        item_a, item_b = metadata["items"]
        rl.log_metadata({
            "metadata/time_sec": extract_time,
            "metadata/mode": mode,
            "metadata/num_items": 2,
            
            # Item A
            "metadata/item_a_context_length": len(item_a.get("context", "")),
            "metadata/item_a_source": item_a.get("source", "unknown")[:50],
            "metadata/item_a_has_budget": bool(item_a.get("budget", "미기재") != "미기재"),
            
            # Item B
            "metadata/item_b_context_length": len(item_b.get("context", "")),
            "metadata/item_b_source": item_b.get("source", "unknown")[:50],
            "metadata/item_b_has_budget": bool(item_b.get("budget", "미기재") != "미기재"),
        })
    else:
        rl.log_metadata({
            "metadata/time_sec": extract_time,
            "metadata/mode": mode,
            "metadata/context_length": len(metadata.get("context", "")),
            "metadata/source": metadata.get("source", "unknown")[:50],
            "metadata/has_budget": bool(metadata.get("budget", "미기재") != "미기재"),
            "metadata/has_org": bool(metadata.get("org", "미상") != "미상"),
            "metadata/has_dates": bool(metadata.get("open_date", "미기재") != "미기재")
        })

    return {"metadata": metadata}


# ============================================
# 3) Distillation 노드 + 로깅
# ============================================
def node_distill(state: RAGState):
    """컨텍스트 증류 + 로깅"""
    distill_start = time.time()
    
    question = state["question"]
    meta = state["metadata"]
    
    if meta.get("mode") == "compare":
        item_a, item_b = meta["items"]
        
        # Item A distill
        distill_a_start = time.time()
        distilled_a = distill_context(question, item_a["context"])
        distill_a_time = time.time() - distill_a_start
        
        # Item B distill
        distill_b_start = time.time()
        distilled_b = distill_context(question, item_b["context"])
        distill_b_time = time.time() - distill_b_start

        total_distill_time = time.time() - distill_start
        
        # 🔥 비교 모드 distillation 로깅
        rl.log({
            "distillation/total_time_sec": total_distill_time,
            "distillation/mode": "compare",
            
            # Item A
            "distillation/item_a_time_sec": distill_a_time,
            "distillation/item_a_original_length": len(item_a["context"]),
            "distillation/item_a_distilled_length": len(distilled_a),
            "distillation/item_a_compression_ratio": len(distilled_a) / len(item_a["context"]) if len(item_a["context"]) > 0 else 0,
            
            # Item B
            "distillation/item_b_time_sec": distill_b_time,
            "distillation/item_b_original_length": len(item_b["context"]),
            "distillation/item_b_distilled_length": len(distilled_b),
            "distillation/item_b_compression_ratio": len(distilled_b) / len(item_b["context"]) if len(item_b["context"]) > 0 else 0,
        })

        return {"distilled_context": f"[A]\n{distilled_a}\n\n[B]\n{distilled_b}"}

    else:
        # 단일 모드 distill
        original_length = len(meta["context"])
        distilled = distill_context(question, meta["context"])
        distill_time = time.time() - distill_start

        # 🔥 단일 모드 distillation 로깅
        rl.log({
            "distillation/time_sec": distill_time,
            "distillation/mode": "single",
            "distillation/original_length": original_length,
            "distillation/distilled_length": len(distilled),
            "distillation/compression_ratio": len(distilled) / original_length if original_length > 0 else 0,
            "distillation/chars_per_sec": len(distilled) / distill_time if distill_time > 0 else 0
        })

        return {"distilled_context": distilled}


# ============================================
# 4) 프롬프트 생성 + 로깅
# ============================================
def node_build_prompt(state: RAGState):
    """프롬프트 구성 + 로깅"""
    prompt_start = time.time()
    
    meta = state["metadata"]
    question = state["question"]
    use_distill = state.get("use_distill", False)

    # 최종 context 선택
    if use_distill:
        ctx = state["distilled_context"]
        context_source = "distilled"
    else:
        if meta.get("mode") == "compare":
            item_a, item_b = meta["items"]
            ctx = f"[문서A]\n{item_a['context']}\n\n[문서B]\n{item_b['context']}"
            context_source = "original_compare"
        else:
            ctx = meta["context"]
            context_source = "original_single"

    # 비교 모드 프롬프트
    if meta.get("mode") == "compare":
        item_a, item_b = meta["items"]

        filled = prompt.format(
            question=question,
            source=f"{item_a['source']} / {item_b['source']}",
            org=f"{item_a['org']} / {item_b['org']}",
            category=f"{item_a['category']} / {item_b['category']}",
            budget=f"{item_a['budget']} / {item_b['budget']}",
            open_date=f"{item_a['open_date']} / {item_b['open_date']}",
            end_date=f"{item_a['end_date']} / {item_b['end_date']}",
            context=ctx
        )
    else:
        # 단일 모드 프롬프트
        filled = prompt.format(
            question=question,
            source=meta["source"],
            org=meta["org"],
            category=meta["category"],
            budget=meta["budget"],
            open_date=meta["open_date"],
            end_date=meta["end_date"],
            context=ctx
        )

    prompt_time = time.time() - prompt_start

    # 🔥 프롬프트 통계 로깅
    rl.log_prompt({
        "prompt/build_time_sec": prompt_time,
        "prompt/total_length": len(filled),
        "prompt/question_length": len(question),
        "prompt/context_length": len(ctx),
        "prompt/context_source": context_source,
        "prompt/mode": meta.get("mode", "single"),
        "prompt/use_distill": use_distill,
        "prompt/context_ratio": len(ctx) / len(filled) if len(filled) > 0 else 0
    })

    return {"prompt": filled}


# ============================================
# 5) LLM 생성 + 로깅
# ============================================
def node_llm(state: RAGState):
    """LLM 생성 + 로깅"""
    generation_start = time.time()
    
    llm = get_llm('llm')
    raw = llm.invoke(state['prompt'])
    answer = raw.content if hasattr(raw, "content") else str(raw)
    
    generation_time = time.time() - generation_start

    # 토큰 + 비용 로깅
    log_usage('main', raw)

    # 🔥 생성 통계 로깅
    rl.log_generation({
        'generation/time_sec': generation_time,
        'generation/answer_length': len(answer),
        'generation/answer_word_count': len(answer.split()),
        'generation/chars_per_sec': len(answer) / generation_time if generation_time > 0 else 0,
        'generation/prompt_length': len(state['prompt']),
        'generation/answer_preview': answer[:100] + "..." if len(answer) > 100 else answer
    })

    return {"answer": answer}


# ============================================
# 6) 스코어링 및 라우팅
# ============================================
def route_after_scoring(state: RAGState):
    """점수에 따른 라우팅 + 로깅"""
    score = state["score"]
    
    print(f"[DECISION] score={score}")
    
    # 🔥 라우팅 결정 로깅
    rl.log_routing({
        "routing/decision": "accept",
        "routing/final_score": score,
        "routing/is_compare": state.get("is_compare", False),
        "routing/used_distillation": state.get("use_distill", False)
    })
    
    return "good"


# ============================================
# 7) 그래프 구축
# ============================================
def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("compare_judge", node_compare_judge)
    graph.add_node("retrieve_docs", node_retrieve)
    graph.add_node("extract_meta", node_extract_metadata)
    graph.add_node("distill", node_distill)
    graph.add_node("build_prompt", node_build_prompt)
    graph.add_node("generate", node_llm)
    graph.add_node("score", node_score)

    graph.set_entry_point("compare_judge")

    graph.add_edge("compare_judge", "retrieve_docs")
    graph.add_edge("retrieve_docs", "extract_meta")

    # distill toggle
    graph.add_conditional_edges(
        "extract_meta",
        lambda s: "distill" if s.get("use_distill", False) else "build",
        {
            "distill": "distill",
            "build": "build_prompt",
        }
    )

    graph.add_edge("distill", "build_prompt")
    graph.add_edge("build_prompt", "generate")
    graph.add_edge("generate", "score")

    graph.add_conditional_edges(
        "score",
        route_after_scoring,
        {"good": END},
    )

    return graph.compile()


# ============================================
# 8) 실행 + 종합 로깅
# ============================================
def run_rag_graph(question, vector_store, retriever, use_distill=False):
    """Langgraph multisearch 실행 + 종합 로깅"""
    print(f"입력 질문: {question}")

    # 초기 설정 로깅
    rl.log({
        "config/use_distill": use_distill,
        "config/question": question,
        "config/question_length": len(question)
    })

    pipeline_start = time.time()

    app = build_graph()
    
    try:
        result = app.invoke({
            "question": question,
            "vector_store": vector_store,
            "retriever": retriever,
            "use_distill": use_distill,
            "retry": 0
        })

        pipeline_duration = time.time() - pipeline_start

        # 🔥 최종 파이프라인 통계
        rl.log_pipeline({
            'pipeline/total_time_sec': pipeline_duration,
            'pipeline/question': question,
            'pipeline/answer': result["answer"],
            'pipeline/answer_length': len(result["answer"]),
            'pipeline/final_score': result.get("score", 0.0),
            'pipeline/is_compare': result.get("is_compare", False),
            'pipeline/used_distillation': use_distill,
            'pipeline/num_retrieved_docs': len(result.get("docs", [])),
            'pipeline/success': True
        })

        print(f'Duration Time: {pipeline_duration:.2f}s')
        print(f'Final Score: {result.get("score", 0.0):.2f}')

        return result["answer"]
        
    except Exception as e:
        pipeline_duration = time.time() - pipeline_start
        
        rl.log_pipeline({
            'pipeline/total_time_sec': pipeline_duration,
            'pipeline/success': False,
            'pipeline/error': str(e)
        })
        
        print(f'❌ 오류 발생: {e}')
        raise