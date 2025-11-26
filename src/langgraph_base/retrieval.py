import unicodedata, re
from difflib import SequenceMatcher
from langchain_core.documents import Document

from compare_judge_template import classify_question_with_llm

def format_docs(docs):
    formatted = []
    for d in docs:
        title = d.metadata.get("source", None)
        if title:
            formatted.append(f"📄 원문 문서명: {title}\n{d.page_content}")
        else:
            formatted.append(d.page_content)
    return "\n\n".join(formatted)

def find_docs_by_question(input_data, vector_store, retriever, top_n=1):
    """rag_chain은 여전히 question만 전달. 내부에서 비교형이면 자동 처리"""
    question = input_data["question"] if isinstance(input_data, dict) else input_data
    normalized_q = unicodedata.normalize("NFC", str(question))
    docs = vector_store.get(include=["metadatas", "documents"], limit=99999)

    # 🔥 1) GPT-5-nano로 질문 유형 판별
    parsed = classify_question_with_llm(normalized_q)
    q_type = parsed.get("질문유형", "단일")

    # 🔥 2) 비교형 질문이면 compare list 사용
    if q_type == "비교":
        sub_questions = parsed.get("비교_사업", [])
    else:
        sub_questions = [normalized_q]

    # fallback
    if not sub_questions:
        sub_questions = [normalized_q]

    selected_docs = []
    for sub_q in sub_questions[:2]:  # 비교형은 최대 2개만
        scored = []
        for meta, content in zip(docs["metadatas"], docs["documents"]):
            src = unicodedata.normalize("NFC", str(meta.get("source", ""))).strip()
            if not src:
                continue
            sim = SequenceMatcher(None, sub_q, src).ratio()
            scored.append((sim, src, content, meta))
        if scored:
            scored.sort(reverse=True, key=lambda x: x[0])
            top = scored[0]
            selected_docs.append(top)
            print(f"📄 자동선택된 문서: {top[1]} (유사도 {top[0]:.3f})")

    if not selected_docs:
        print("⚠️ 문서 없음 → retriever fallback 사용")
        return retriever.invoke(question)
    
    # 🔥 병합된 content와 metadata를 Document 객체로 변환
    merged_content = "\n\n--- 비교 문서 구분선 ---\n\n".join([c for _, _, c, _ in selected_docs])
    merged_meta = {
        "sources": [m for _, _, _, m in selected_docs],
        "source": " / ".join([m.get("source", "미기재") for _, _, _, m in selected_docs]),
        "org": " / ".join([m.get("org", "미상") for _, _, _, m in selected_docs]),
        "category": " / ".join([m.get("category", "미상") for _, _, _, m in selected_docs]),
        "budget": " / ".join([m.get("budget", "미기재") for _, _, _, m in selected_docs]),
        "open_date": " / ".join([m.get("open_date", "미기재") for _, _, _, m in selected_docs]),
        "end_date": " / ".join([m.get("end_date", "미기재") for _, _, _, m in selected_docs])
    }
    
    # Document 객체로 반환!
    return [Document(page_content=merged_content, metadata=merged_meta)]





