# langgraph_multisearch/distilled_prompt.py
import time
import rag_logger as rl
from langchain_core.prompts import ChatPromptTemplate
from llm_config import get_llm, log_usage

DISTLL_PROMPT = ChatPromptTemplate.from_template("""
당신은 문서 분석을 전문으로 하는 Distillation AI입니다.
아래 문서 내용을 읽고, 질문에 답변하는 데 반드시 필요한 핵심 정보만 추출하세요.
요약이 아니라, **질문에 직접 필요한 정보만 발라내서 압축**하는 것이 목적.

규칙:
1. 문서 전체 요약 금지
2. 질문과 직접 관련 없는 내용은 절대 넣지 말 것
3. 핵심 키워드 / 핵심 문장 / 중요한 수치만 포함
4. 최종 generate LLM이 사용하므로 최대한 간결하고 정확하게 작성

━━━━━━━━━━━━━━━━━━━━━━━━
📩 질문:
{question}

📄 문서:
{context}
━━━━━━━━━━━━━━━━━━━━━━━━

📘 distillation 출력 형식:
- 핵심 키워드:
- 중요 문장:
- 숫자 / 일정:
- 질문과 직접 연관된 논점 요약:
""")

def distill_context(question: str, context: str):
    """컨텍스트 증류 + 로깅"""
    distill_start = time.time()
    
    prompt = DISTLL_PROMPT.format(
        question=question,
        context=context
    )
    
    distill_llm = get_llm('distill')
    raw = distill_llm.invoke(prompt)
    result = raw.content if hasattr(raw, 'content') else str(raw)
    
    distill_time = time.time() - distill_start
    
    # 토큰 + 비용 로깅
    log_usage('distill', raw)
    
    # 🔥 증류 통계 로깅 (함수 레벨)
    rl.log({
        "distill_call/time_sec": distill_time,
        "distill_call/input_length": len(context),
        "distill_call/output_length": len(result),
        "distill_call/compression_ratio": len(result) / len(context) if len(context) > 0 else 0
    })
    
    return result