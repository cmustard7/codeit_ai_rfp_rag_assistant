import json
from langchain_openai import ChatOpenAI

from src.llm_config import classifier_llm

classifier_system_prompt = """
너는 한국어 RAG 시스템을 위한 질문 분석기다.

사용자의 질문을 분석해서 아래 JSON 스키마로 정확히 판단해라:
{
  "질문유형": "단일" 또는 "비교",
  "비교_사업": ["string"]  # 비교형일 경우만 2개 또는 그 이상
}

규칙:
- 질문이 사업 A와 B를 비교하려는 의도가 명확하면 "비교"로 분류
- '이랑', '및', 'vs', 'VS', '대비', '비교', '차이', '무엇이 더', '상대적으로' 등 문맥도 고려
- 비교가 아닌 경우 "질문유형": "단일" 로 설정
- JSON 외 다른 텍스트는 출력하지 마라
"""

def classifier_user_prompt(question: str):
    return f"""
다음 질문을 분석해서 JSON으로만 답하세요:

질문: "{question}"
"""

def classify_question_with_llm(question: str):
    messages = [
        {"role": "system", "content": classifier_system_prompt},
        {"role": "user", "content": classifier_user_prompt(question)}
    ]

    raw = classifier_llm.invoke(messages).content

    try:
        return json.loads(raw)
    except:
        # JSON 파싱 실패하면 다시 정제 시도
        fix_messages = [
            {"role": "system", "content": "아래 텍스트에서 JSON만 추출해서 정확히 반환해라."},
            {"role": "user", "content": raw}
        ]
        fixed = classifier_llm.invoke(fix_messages).content
        return json.loads(fixed)