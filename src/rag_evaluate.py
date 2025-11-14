import json

from src.rag_state import RAGState
from src.llm_config import judge_llm

def _get_page_content(d):
    # dict 형태
    if isinstance(d, dict):
        return d.get("page_content", "")
    # LangChain Document 형태
    page = getattr(d, "page_content", "")
    return page

def _get_metadata(doc):
    if hasattr(doc, "metadata"):
        return doc.metadata
    if isinstance(doc, dict):
        return doc.get("metadata", {})
    return {}

def node_score(state: RAGState):
    answer = state["answer"]
    question = state["question"]
    docs = state["docs"]

    doc_text = "\n\n".join(
        f"[문서 {i+1}]\n{_get_page_content(d)}"
        for i, d in enumerate(docs)
    )

    metadata_text = "\n\n".join(
        f"[문서 {i+1} 메타데이터]\n" + 
        "\n".join(f"{k}: {v}" for k, v in _get_metadata(d).items())
        for i, d in enumerate(docs)
    )

    prompt = f"""
    당신은 질문-답변 일치도를 평가하는 전문 심사관입니다.

    ▣ 질문:
    {question}

    ▣ 답변:
    {answer}

    ▣ 참고 문서 내용:
    {doc_text}

    ▣ 문서 메타데이터 (발주기관/예산/날짜 등):
    {metadata_text}

    이 정보를 기반으로 다음 기준에 따라 엄격하게 평가하세요.

    평가 기준:
    - 답변이 문서 내용 또는 해당 문서의 메타데이터에 기반했는가?
    - 문서에 없는 상세 수치, 날짜, 예산 등을 추측했는가?
    - 문서 근거와 메타데이터를 정확히 반영했는가?

    JSON ONLY:
    {{
        "score": 0.0,
        "reason": "..."
    }}
    """

    judge_raw = judge_llm.invoke(prompt)
    judge_text = judge_raw.content if hasattr(judge_raw, "content") else str(judge_raw)

    print("[JUDGE RAW OUTPUT]\n", judge_text)

    try:
        parsed = json.loads(judge_text)
        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "")
    except:
        score = 0.0
        reason = "JSON 파싱 실패"

    return {"score": score, "evaluate_reason": reason}