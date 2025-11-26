# langgraph_base/adaptive_retreival.py
import time
import rag_logger as rl
from llm_config import get_llm
from langgraph_base.retrieval import find_docs_by_question
from langgraph_base.rag_state import RAGState

def generate_refined_query(question: str):
    """쿼리 리파인 + 로깅"""
    refine_start = time.time()
    
    llm = get_llm('llm')
    refined = llm.invoke(
        f"다음 질문을 정보 검색에 더 최적화된 형태로 다시 작성해줘: {question}"
    )
    
    refine_time = time.time() - refine_start
    refined_text = refined.content if hasattr(refined, "content") else str(refined)
    
    rl.log({
        "query_refinement/time_sec": refine_time,
        "query_refinement/original_length": len(question),
        "query_refinement/refined_length": len(refined_text)
    })
    
    return refined_text

# 🔥 헬퍼 함수 추가
def _safe_get_page_content(doc):
    """Document 객체나 dict 모두 처리"""
    if isinstance(doc, dict):
        return doc.get("page_content", "")
    return getattr(doc, "page_content", "")

def node_retrieve(state: RAGState):
    """적응형 검색 + 상세 로깅"""
    retrieve_start = time.time()
    
    retry = state.get("retry", 0)
    question = state['question']
    
    vector_store = state["vector_store"]
    retriever = state["retriever"]
    
    print(f"\n[RETRIEVE] 재시도 번호: {retry}")

    # Case 0: 기본 strict 검색
    if retry == 0:
        print(" → Strategy: strict search (top_n=1)")
        docs = find_docs_by_question(
            {'question': question},
            vector_store,
            retriever,
            top_n=1
        )
        strategy = "strict"

    # Case 1: top_n 확장
    elif retry == 1:        
        print(" → Strategy: expanded search (top_n=5)")
        docs = find_docs_by_question(
            {'question': question},
            vector_store,
            retriever,
            top_n=5
        )
        strategy = "expanded"

    # Case 2: retriever fallback
    elif retry == 2:
        print(" → Strategy: semantic retriever fallback")
        docs = retriever.invoke(question)
        strategy = "semantic_fallback"

    # Case 3: refined query
    elif retry == 3:
        refined_q = generate_refined_query(question)
        print(f" → Strategy: refined-query search\n    refined: {refined_q}")
        docs = retriever.invoke(refined_q)
        strategy = "refined_query"
        
        rl.log({
            "retrieval/refined_query": refined_q
        })

    # Case 4: metadata keyword fallback
    elif retry == 4:
        print(" → Strategy: keyword metadata fallback")
        all_docs = vector_store.get(include=["documents", "metadatas"], limit=99999)
        matched = []

        for meta, content in zip(all_docs["metadatas"], all_docs["documents"]):
            src = meta.get("source", "")
            if question.replace(" ", "") in src.replace(" ", ""):
                from langchain_core.documents import Document
                matched.append(Document(page_content=content, metadata=meta))

        if matched:
            docs = matched
            strategy = "keyword_match"
        else:
            docs = retriever.invoke(question)
            strategy = "keyword_fallback"

    # Case 5: 더 이상 시도 안 함
    else:
        docs = state.get("docs", [])
        strategy = "exhausted"
    
    retrieve_time = time.time() - retrieve_start
    
    # 🔥 검색 통계 로깅 (수정됨!)
    total_chars = sum(len(_safe_get_page_content(d)) for d in docs)
    
    rl.log_retrieval({
        "retrieval/time_sec": retrieve_time,
        "retrieval/strategy": strategy,
        "retrieval/retry_number": retry,
        "retrieval/num_docs": len(docs),
        "retrieval/has_results": len(docs) > 0,
        "retrieval/total_chars": total_chars
    })
    
    # 문서별 상세 정보 (retry 0일 때만)
    if retry == 0:
        for i, doc in enumerate(docs[:3]):  # 최대 3개만
            if hasattr(doc, 'metadata'):
                meta = doc.metadata
                rl.log({
                    f"retrieval/doc_{i}_source": meta.get('source', 'unknown')[:50],
                    f"retrieval/doc_{i}_length": len(_safe_get_page_content(doc))
                })

    return {
        "docs": docs, 
        "retry": retry + 1
    }