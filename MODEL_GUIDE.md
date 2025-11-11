# 🤖 LLM 모델 선택 가이드

## 현재 설정

**기본 모델**: `gpt-4o-mini` (OpenAI)

---

## 📊 OpenAI 모델 비교 (2025.11 기준)

### 추천 모델

| 모델 | 용도 | 비용 | Temperature | 특징 |
|------|------|------|-------------|------|
| **gpt-4o-mini** | 베이스라인 (권장) | 💰 저렴 | ✅ 지원 | 빠르고 비용 효율적 |
| **gpt-4o** | 고성능 필요 시 | 💰💰 중간 | ✅ 지원 | 높은 품질, 복잡한 추론 |
| **gpt-3.5-turbo** | 저비용 테스트 | 💰 매우 저렴 | ✅ 지원 | 기본 기능, 빠른 속도 |

### ❌ 사용 불가 모델
- **gpt-5-mini**, **gpt-5-nano**: 아직 공개되지 않음

---

## 💰 비용 비교 (대략적)

### 입력 토큰 (1M 토큰당)
- gpt-3.5-turbo: $0.50
- gpt-4o-mini: $0.15
- gpt-4o: $2.50

### 출력 토큰 (1M 토큰당)
- gpt-3.5-turbo: $1.50
- gpt-4o-mini: $0.60
- gpt-4o: $10.00

### RAG 프로젝트 예상 비용 (100개 문서 기준)

#### Embedding (text-embedding-3-small)
- 약 $0.50-1.00

#### Generation (질문당)
- gpt-3.5-turbo: $0.005-0.01
- gpt-4o-mini: $0.01-0.02 ✅ **권장**
- gpt-4o: $0.05-0.10

#### 총 예상 비용 (10개 질문)
- **gpt-4o-mini**: $1.60-2.20 ✅ **최적**
- gpt-3.5-turbo: $1.55-2.00 (품질 낮음)
- gpt-4o: $2.00-3.00 (과도한 비용)

---

## 🔧 모델 변경 방법

### 방법 1: config.py 수정 (영구적)

```python
# src/config.py

# LLM 설정
LLM_MODEL = "gpt-4o-mini"  # 또는 "gpt-4o", "gpt-3.5-turbo"
TEMPERATURE = 0.3
MAX_TOKENS = 500
```

### 방법 2: 노트북에서 직접 변경 (임시)

```python
# 05_baseline_rag.ipynb

llm = ChatOpenAI(
    model="gpt-4o",  # 다른 모델로 변경
    temperature=0.3,
    max_tokens=500,
    openai_api_key=api_key
)
```

---

## 🎯 모델 선택 기준

### gpt-4o-mini를 사용하는 경우 (베이스라인 권장)
- ✅ 비용 효율적
- ✅ 충분한 성능
- ✅ 빠른 응답 속도
- ✅ Temperature 제어 가능

### gpt-4o를 사용하는 경우
- 복잡한 문서 분석 필요
- 높은 추론 능력 필요
- 다국어 지원 필요
- 비용 제약 적음

### gpt-3.5-turbo를 사용하는 경우
- 초기 프로토타입 테스트
- 극도의 비용 절감 필요
- 단순한 질문만 처리

---

## ⚙️ Temperature 설정

### Temperature란?
모델의 출력 다양성을 제어하는 파라미터 (0.0 ~ 2.0)

| Temperature | 특징 | 용도 |
|-------------|------|------|
| **0.0** | 가장 확정적, 일관됨 | 사실 기반 답변 |
| **0.3** | 약간 다양, 안정적 | ✅ **RAG 권장** |
| **0.7** | 균형잡힌 창의성 | 일반 대화 |
| **1.0+** | 매우 창의적 | 창작, 브레인스토밍 |

### RAG에서 0.3을 권장하는 이유
- ✅ 문서 내용을 충실히 반영
- ✅ Hallucination 최소화
- ✅ 일관된 답변
- ✅ 사실 기반 응답

---

## 🔬 실험 가이드

### 다양한 모델 비교 실험

```python
models_to_test = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4o"
]

results = {}

for model in models_to_test:
    llm = ChatOpenAI(model=model, temperature=0.3)
    # 동일한 질문으로 테스트
    result = rag_query(test_question, llm=llm)
    results[model] = result

# 비교 분석
compare_results(results)
```

### 평가 지표
1. **답변 품질**: 정확성, 완성도
2. **응답 시간**: 속도
3. **비용**: API 호출 비용
4. **Hallucination**: 문서 외 정보 포함 여부

---

## 📝 모델별 특징 상세

### gpt-4o-mini (권장)
- **장점**:
  - 비용 대비 성능 최고
  - 빠른 응답 속도
  - 한국어 지원 우수
  - Temperature 제어 가능
- **단점**:
  - gpt-4o보다는 추론 능력 낮음
- **추천 용도**: 베이스라인, 프로덕션

### gpt-4o
- **장점**:
  - 최고 수준의 추론 능력
  - 복잡한 문서 이해
  - 멀티모달 지원
- **단점**:
  - 높은 비용
  - 느린 응답 속도
- **추천 용도**: 최종 평가, 고급 분석

### gpt-3.5-turbo
- **장점**:
  - 매우 저렴
  - 빠른 속도
- **단점**:
  - 낮은 추론 능력
  - 복잡한 문서 이해 어려움
- **추천 용도**: 초기 테스트만

---

## 🔮 향후 모델 업데이트

### 예상되는 새 모델들
- **gpt-5** (출시 예정): 더 강력한 추론 능력
- **gpt-4.5**: 중간 성능 모델
- **새로운 mini 버전들**: 비용 효율성 개선

### 업데이트 체크
- [OpenAI 공식 문서](https://platform.openai.com/docs/models)
- [OpenAI 가격 페이지](https://openai.com/pricing)
- 월 1회 모델 정보 확인 권장

---

## 💡 실전 팁

### 1. 개발 vs 프로덕션
```python
# 개발/테스트
LLM_MODEL = "gpt-3.5-turbo"  # 저렴

# 평가/프로덕션
LLM_MODEL = "gpt-4o-mini"  # 균형
```

### 2. Temperature 조정 실험
```python
temperatures = [0.0, 0.3, 0.7]

for temp in temperatures:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temp)
    # 동일 질문으로 답변 비교
```

### 3. Max Tokens 조정
```python
# 짧은 답변
MAX_TOKENS = 300

# 상세한 답변
MAX_TOKENS = 800

# 매우 상세한 답변
MAX_TOKENS = 1500
```

### 4. 비용 추적
```python
import tiktoken

def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# 비용 계산
input_tokens = count_tokens(prompt)
output_tokens = count_tokens(response)
total_cost = calculate_cost(input_tokens, output_tokens, model)
```

---

## 📞 문의 및 참고

- [OpenAI 공식 문서](https://platform.openai.com/docs)
- [모델 비교표](https://platform.openai.com/docs/models)
- [가격 정보](https://openai.com/pricing)
- [API 가이드](https://platform.openai.com/docs/api-reference)

---

**마지막 업데이트**: 2025-11-11
**다음 확인 예정**: 2025-12-01
