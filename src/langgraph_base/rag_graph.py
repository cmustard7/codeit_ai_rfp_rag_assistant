# langgraph_base/rag_graph.py
import rag_logger as rl
import time
import json

from langgraph.graph import StateGraph, END

from metadata import extract_metadata
from prompt_template import prompt
from llm_config import get_llm, log_usage
from langgraph_base.rag_evaluate import node_score
from langgraph_base.adaptive_retreival import node_retrieve
from langgraph_base.rag_state import RAGState

# ============================================
# 개선된 노드들 - 상세 로깅 포함
# ============================================

def node_extract_metadata(state: RAGState):
    """메타데이터 추출 + 로깅"""
    extract_start = time.time()
    
    metadata = extract_metadata(state['docs'])
    
    extract_time = time.time() - extract_start
    
    # 메타데이터 통계 로깅
    rl.log_metadata({
        "metadata/extraction_time_sec": extract_time,
        "metadata/context_length": len(metadata.get('context', '')),
        "metadata/num_sources": len(metadata.get('source', '').split(' / ')),
        "metadata/has_budget": bool(metadata.get('budget', '미기재') != '미기재'),
        "metadata/has_org": bool(metadata.get('org', '미상') != '미상'),
        "metadata/has_dates": bool(metadata.get('open_date', '미기재') != '미기재'),
        "metadata/retry_count": state.get('retry', 0)
    })
    
    return {'metadata': metadata}

def node_build_prompt(state: RAGState):
    """프롬프트 구성 + 로깅"""
    prompt_start = time.time()
    
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
    
    prompt_time = time.time() - prompt_start
    
    # 프롬프트 통계 로깅
    rl.log_prompt({
        "prompt/build_time_sec": prompt_time,
        "prompt/total_length": len(formatted_prompt),
        "prompt/question_length": len(state['question']),
        "prompt/context_length": len(state['metadata']['context']),
        "prompt/context_ratio": len(state['metadata']['context']) / len(formatted_prompt) if len(formatted_prompt) > 0 else 0,
        "prompt/retry_count": state.get('retry', 0)
    })
    
    return {'prompt': formatted_prompt}

def node_llm(state: RAGState):
    """LLM 생성 + 로깅"""
    generation_start = time.time()
    
    llm = get_llm('llm')
    raw = llm.invoke(state['prompt'])
    answer = raw.content if hasattr(raw, "content") else str(raw)
    
    generation_time = time.time() - generation_start
    
    # 생성 통계 로깅
    log_usage('main', raw)  # 토큰 + 비용
    
    rl.log_generation({
        'generation/time_sec': generation_time,
        'generation/answer_length': len(answer),
        'generation/answer_word_count': len(answer.split()),
        'generation/chars_per_sec': len(answer) / generation_time if generation_time > 0 else 0,
        'generation/retry_count': state.get('retry', 0)
    })

    return {'answer': answer}

def route_after_scoring(state: RAGState):
    """점수에 따른 라우팅 + 로깅"""
    score = state['score']
    retry = state.get('retry', 0)

    print(f"[DECISION] score={score}, retry={retry}")

    # 라우팅 결정 로깅
    if score >= 0.75:
        print(" → GOOD (종료)")
        rl.log_routing({
            "routing/decision": "accept",
            "routing/final_score": score,
            "routing/total_retries": retry
        })
        return 'good'

    if retry >= 5:
        print(" → BAD but retry limit reached (종료)")
        rl.log_routing({
            "routing/decision": "forced_accept",
            "routing/final_score": score,
            "routing/total_retries": retry,
            "routing/max_retries_reached": True
        })
        return 'good'

    print(" → BAD (재검색)")
    rl.log_routing({
        "routing/decision": "retry",
        "routing/current_score": score,
        "routing/retry_number": retry + 1
    })
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
    """Langgraph RAG 실행 + 종합 로깅"""
    print(f'입력 질문: {question}')
    
    # 전체 파이프라인 시작
    pipeline_start = time.time()

    app = build_graph()
    
    try:
        result = app.invoke({
            "question": question,
            "vector_store": vector_store,
            "retriever": retriever,
            "retry": 0
        })
        
        pipeline_duration = time.time() - pipeline_start
        
        # 🔥 최종 파이프라인 통계
        rl.log_pipeline({
            'pipeline/total_time_sec': pipeline_duration,
            'pipeline/question': question,
            'pipeline/answer_preview': result["answer"][:100] + "..." if len(result["answer"]) > 100 else result["answer"],
            'pipeline/final_retry_count': result.get('retry', 0),
            'pipeline/final_score': result.get('score', 0.0),
            'pipeline/success': True
        })
        
        print(f'Duration Time: {pipeline_duration:.2f}s')
        
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