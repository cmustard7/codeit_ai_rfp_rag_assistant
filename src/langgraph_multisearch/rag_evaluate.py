# langgraph_multisearch/rag_evaluate.py
import json
import time
import rag_logger as rl

from langgraph_multisearch.rag_state import RAGState
from llm_config import get_llm, log_usage

def _get_page_content(d):
    if isinstance(d, dict):
        return d.get("page_content", "")
    page = getattr(d, "page_content", "")
    return page

def _get_metadata(doc):
    if hasattr(doc, "metadata"):
        return doc.metadata
    if isinstance(doc, dict):
        return doc.get("metadata", {})
    return {}

def node_score(state: RAGState):
    """답변 평가 + 상세 로깅"""
    eval_start = time.time()
    
    answer = state["answer"]
    question = state["question"]
    docs = state["docs"]

    doc_text = "\n\n".join(
        f"[문서 {i+1}]\n{_get_page_content(d)[:1500]}"
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
    
    judge_llm = get_llm('judge')
    judge_raw = judge_llm.invoke(prompt)
    judge_text = judge_raw.content if hasattr(judge_raw, "content") else str(judge_raw)

    print("[JUDGE RAW OUTPUT]\n", judge_text)

    try:
        cleaned = judge_text.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(cleaned)
        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "")
        parse_success = True
    except Exception as e:
        print(f"⚠️ JSON 파싱 실패: {e}")
        score = 0.0
        reason = "JSON 파싱 실패"
        parse_success = False

    eval_time = time.time() - eval_start
    
    # 토큰 + 비용 로깅
    log_usage('judge', judge_raw)
    
    # 🔥 평가 통계 로깅
    rl.log_evaluation({
        "evaluation/time_sec": eval_time,
        "evaluation/score": score,
        "evaluation/reason": reason[:100],  # 너무 길면 자르기
        "evaluation/reason_length": len(reason),
        "evaluation/parse_success": parse_success,
        "evaluation/prompt_length": len(prompt),
        "evaluation/num_docs_evaluated": len(docs),
        "evaluation/is_compare": state.get("is_compare", False)
    })

    return {
        "score": score, 
        "evaluate_reason": reason
    }