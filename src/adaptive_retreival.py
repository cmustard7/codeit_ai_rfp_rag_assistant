from src.llm_config import llm as refine_llm
from src.rag_state import RAGState
from src.retrieval import find_docs_by_question

def generate_refined_query(question: str):
    refined = refine_llm.invoke(
        f"다음 질문을 정보 검색에 더 최적화된 형태로 다시 작성해줘: {question}"
    )
    return refined.content if hasattr(refined, "content") else str(refined)

def node_retrieve(state: RAGState):
    retry = state.get("retry", 0)
    question = state['question']
    
    vector_store = state["vector_store"]
    retriever = state["retriever"]
    
    print(f"\n[RETRIEVE] 재시도 번호: {retry}")

    # Case 0: 기본 strict 검색 (현재 방식)
    if retry == 0:
        print(" → Strategy: strict search (top_n=3)")
        docs = find_docs_by_question(
            {'question': question},
            vector_store,
            retriever,
            top_n=1
        )
        return {"docs": docs, "retry": retry + 1}

    # Case 1: top_n 확장해서 더 많은 문서 확인
    if retry == 1:        
        print(" → Strategy: expanded search (top_n=5)")
        docs = find_docs_by_question(
            {'question': question},
            vector_store,
            retriever,
            top_n=5
        )
        return {"docs": docs, "retry": retry + 1}

    # Case 2: retriever fallback (semantic search 직접)
    if retry == 2:
        print(" → Strategy: semantic retriever fallback")
        docs = retriever.invoke(question)
        return {"docs": docs, "retry": retry + 1}

    # Case 3: refined multi-query 검색
    if retry == 3:
        refined_q = generate_refined_query(question)
        print(f" → Strategy: refined-query search\n    refined: {refined_q}")
        docs = retriever.invoke(refined_q)
        return {
            "docs": docs,
            "retry": retry + 1,
            "refined_query": refined_q
        }

    # Case 4: metadata 기반 keyword fallback
    if retry == 4:
        print(" → Strategy: keyword metadata fallback")
        all_docs = vector_store.get(include=["documents", "metadatas"], limit=99999)
        matched = []

        for meta, content in zip(all_docs["metadatas"], all_docs["documents"]):
            src = meta.get("source", "")
            if question.replace(" ", "") in src.replace(" ", ""):
                matched.append({"page_content": content, "metadata": meta})

        if matched:
            return {"docs": matched, "retry": retry + 1}

        # fallback → retriever로라도 가져오기
        fallback_docs = retriever.invoke(question)
        return {"docs": fallback_docs, "retry": retry + 1}

    # Case 5: 더 이상 할 수 있는 게 없으므로 종료
    return {
        "docs": state.get("docs", []),
        "retry": retry
    }