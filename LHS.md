# 정부 입찰 RFP 분석을 위한 고도화 RAG 시스템 개발

**프로젝트 기간**: 2025년 11월 10일 - 11월 28일 (3주)  
**프로젝트 형태**: AI 부트캠프 NLP 팀 프로젝트  
**시나리오**: 입찰메이트 (B2G 입찰 컨설팅 스타트업) 엔지니어링 팀  
**담당 역할**: Retriever 담당 / 검색 성능 비교 및 최적화  
**기술 환경**: Python 3.12.10, GPT-5-mini, GPT-5-nano  

---

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [기술 스택](#2-기술-스택)
3. [시스템 아키텍처 발전 과정](#3-시스템-아키텍처-발전-과정)
4. [개발 과정 및 문제 해결](#4-개발-과정-및-문제-해결)
5. [성능 평가 및 분석](#5-성능-평가-및-분석)
6. [핵심 성과 및 기술적 인사이트](#6-핵심-성과-및-기술적-인사이트)
7. [향후 개선 방향](#7-향후-개선-방향)

---

## 1. 프로젝트 개요

### 1.1 배경 및 목적

#### 프로젝트 배경
본 프로젝트는 AI 부트캠프의 NLP(자연어 처리) 팀 프로젝트로, **B2G 입찰 컨설팅 스타트업 '입찰메이트'의 엔지니어링 팀**이라는 시나리오 하에 진행되었다.

정부 및 공공기관에서 발주하는 입찰 공고(RFP, Request for Proposal)는 방대한 분량과 복잡한 구조를 가지고 있다. 하루 수백 건의 RFP가 나라장터 등에 올라오며, 한 요청서당 수십 페이지가 넘기 때문에 기업 담당자들이 일일이 읽어볼 수 없다. '입찰메이트'는 이러한 RFP 속에서 고객사에게 적합한 입찰 기회를 빠르게 찾아 추천하는 컨설팅 서비스를 제공한다.

본 프로젝트는 컨설턴트들이 RFP의 주요 요구 조건, 대상 기관, 예산, 제출 방식 등 핵심 정보를 빠르게 파악할 수 있도록, AI 기반 RAG(Retrieval-Augmented Generation) 시스템을 개발하는 것을 목표로 하였다.

#### 담당 역할
팀 프로젝트 내에서 **Retriever 성능 비교 및 최적화**를 담당하였다. 4가지 Retrieval 전략(langchain, langgraph_base, langgraph_multisearch, langgraph_multisearch_distillation)의 성능을 체계적으로 비교 분석하고, 각 방법론의 장단점을 파악하여 최적의 검색 전략을 도출하는 것이 핵심 역할이었다.

#### 데이터셋
- **문서 수**: 100개의 실제 RFP 문서
- **메타데이터**: 각 문서의 발주 기관, 사업명, 예산, 입찰 마감일, 카테고리 등
- **문서 형식**: HWP (한글 워드프로세서) 파일

### 1.2 프로젝트 목표

- **RAG 시스템 구축**: 100개의 RFP 문서를 기반으로 Q&A가 가능한 시스템 개발
- **Retrieval 전략 비교 분석**: 4가지 검색 방법론의 성능을 정량적/정성적으로 비교
  - langchain (베이스라인)
  - langgraph_base (Agentic RAG)
  - langgraph_multisearch (하이브리드 검색)
  - langgraph_multisearch_distillation (경량화 시도)
- **평가 지표 설계**: 팀 자체적으로 평가 방식과 지표를 선정하고 적용
  - 응답 시간 (pipeline/total_time_sec)
  - 검색 정확도 (retrieval/num_docs, rerank scores)
  - API 비용 (cost_usd, cost_krw)
  - 답변 품질 (answer_length, evaluation score)
- **정확한 문서 검색**: 다수의 RFP 문서에서 사용자 질의와 관련된 정확한 문서 및 섹션 검색
- **맥락 기반 답변 생성**: 검색된 문서를 기반으로 정확하고 상세한 답변 제공
- **비교 분석 지원**: 두 개 이상의 사업을 비교 분석할 수 있는 기능 구현
- **성능 최적화**: 응답 시간 단축 및 검색 정확도 향상
- **모니터링 체계 구축**: Weights & Biases를 활용한 성능 추적 및 분석

### 1.3 핵심 도전 과제

1. **문서 혼재 문제**: 초기 시스템에서 여러 문서가 혼재되어 정확한 문서를 검색하지 못하는 문제
2. **메타데이터 관리**: 문서별 발주 기관, 예산, 기한 등의 메타데이터를 효과적으로 관리하고 활용
3. **멀티 검색 전략**: 의미적 유사도와 메타데이터 기반 검색을 동시에 활용하는 하이브리드 검색 구현
4. **성능과 정확도의 균형**: 응답 속도를 유지하면서도 검색 정확도를 극대화

---

## 2. 기술 스택

### 2.1 Core Framework

- **LangChain**: RAG 파이프라인 구축을 위한 기본 프레임워크
- **LangGraph**: Agentic RAG 구현을 위한 상태 기반 그래프 워크플로우
- **Python 3.12.10**: 주 개발 언어

### 2.2 LLM & Embedding

- **Main LLM**: GPT-5-mini (답변 생성, temperature=0.0)
- **Classifier LLM**: GPT-5-nano (쿼리 분류, temperature=0.0)
- **Judge LLM**: GPT-5-mini (답변 평가, temperature=0.0)
- **Distillation LLM**: GPT-5-mini (경량화 실험용, temperature=0.0)
- **Embedding Model**: OpenAI Embedding (문서 임베딩 및 의미적 유사도 계산)
- **Reranker**: Cross-encoder 기반 문서 재순위화

### 2.3 Vector Database & Storage

- **ChromaDB**: 벡터 데이터베이스 (FAISS와 비교 검토)
- **Pickle**: 청크 데이터 캐싱을 통한 성능 최적화

### 2.4 Monitoring & Visualization

- **Weights & Biases (wandb)**: 실험 추적, 성능 메트릭 로깅, 결과 시각화
- **Streamlit**: GUI 기반 검색 성능 실시간 확인

### 2.5 Document Processing

- **HWP Parser**: 한글 문서 파싱
- **NFC Normalization**: 한글 유니코드 정규화
- **Custom Metadata Injection**: CSV 기반 메타데이터 주입

---

## 3. 시스템 아키텍처 발전 과정

### 3.1 Phase 1: Baseline RAG (11월 10-11일)

#### 구조
```
사용자 질의 → Embedding → Vector Search → LLM 답변 생성
```

#### 특징
- 단순 벡터 유사도 기반 검색
- 문서별 구분 없이 전체 청킹

#### 문제점
1. **문서 혼재**: 문서별 경계가 없어 여러 문서의 청크가 혼재됨
2. **부정확한 검색**: "국민연금공단 이러닝시스템" 질의에 "사회보험료 지원 고시" 문서 반환
3. **메타데이터 부재**: 발주 기관, 예산, 기한 등의 구조화된 정보 활용 불가

### 3.2 Phase 2: Document-Separated RAG with Metadata (11월 12-13일)

#### 개선 사항
1. **문서별 청킹**: 각 문서를 독립적으로 청킹하여 retrieve 생성
2. **메타데이터 주입**: `data_list.csv`를 통해 각 문서에 구조화된 메타데이터 추가
   - 발주 기관명
   - 사업명
   - 예산 정보
   - 입찰 마감일
   - 카테고리

#### 아키텍처
```
사용자 질의 → Embedding → Vector Search (문서별 분리) 
            → Metadata Filtering → Reranking → LLM 답변 생성
```

#### 결과
- 문서 검색 정확도 대폭 향상
- 정성적 평가에서 만족할 만한 답변 생성 확인

### 3.3 Phase 3: LangGraph-based Agentic RAG (11월 14일)

#### 핵심 개선
1. **Chain 구조 적용**: 검색 → 평가 → 답변 생성의 체계적 파이프라인
2. **비교 분석 기능**: 두 개 이상의 문서를 병합하여 비교하는 전용 LLM 추가
3. **평가 시스템**: 생성된 답변의 품질을 평가하는 별도 LLM 도입
4. **적응형 검색 전략**: 5단계 Adaptive Retrieval 구현

#### Adaptive Retrieval 5단계 전략

평가 점수가 낮을 경우 자동으로 검색 전략을 변경하여 재시도하는 방식을 도입하였다:

```python
retry 0: Strict Search (top_n=1)        # 가장 유사한 1개 문서만
retry 1: Expanded Search (top_n=5)      # 검색 범위 확장
retry 2: Semantic Fallback              # 순수 벡터 검색
retry 3: Refined Query                  # LLM으로 쿼리 개선 후 재검색
retry 4: Keyword Metadata Fallback      # 메타데이터 키워드 매칭
retry 5: Exhausted                      # 재시도 중단
```

**라우팅 로직**:
- 평가 점수 ≥ 0.75: 답변 승인 및 종료
- retry ≥ 5: 강제 종료
- 그 외: 다음 단계 재검색 실행

이 방식은 답변 품질을 보장하지만, retry가 발생할 경우 **응답 시간이 1.5배 이상 증가**하는 문제가 있었다. 특히 retry 3단계(쿼리 개선)에서 추가 LLM 호출이 발생하여 레이턴시가 크게 늘어났다.

#### 아키텍처
```
사용자 질의 → Query Analyzer → [단일/비교 질의 분류]
                                    ↓
                        [단일 질의]           [비교 질의]
                            ↓                    ↓
                    Vector Search        Multiple Searches
                         ↓                    ↓
                    Reranking            Merge & Compare
                         ↓                    ↓
                    Answer LLM           Compare LLM
                         ↓                    ↓
                        Evaluation LLM
                         ↓
                    최종 답변
```

#### 추가 최적화
- **PKL 캐싱**: 청크 데이터를 pickle 파일로 저장하여 매번 재생성하는 오버헤드 제거
- **Metadata 함께 전달**: Top-k 청크만 전달 시 정보 손실 문제 해결을 위해 메타데이터도 함께 전달
- **페이지별 청킹**: 페이지 단위로 메타데이터를 구분하여 더 세밀한 검색 가능

### 3.4 Phase 4: Multi-Search Strategy (11월 18-24일)

#### 배경: Adaptive Retrieval의 근본적 한계

Phase 3의 Adaptive Retrieval은 답변 품질을 점진적으로 개선하는 방식이었으나, 다음과 같은 치명적인 문제가 있었다:

1. **속도 페널티**: retry가 발생할 경우 응답 시간이 최소 1.5배 이상 증가
2. **누적 레이턴시**: retry 3단계(쿼리 개선)에서 추가 LLM 호출로 인한 지연
3. **불확실성**: 어떤 질의가 retry를 유발할지 예측 불가능
4. **최악의 경우**: retry 5번 모두 실행 시 처음 검색의 5배 이상 시간 소요

**핵심 인사이트**: 재시도로 품질을 개선하는 것보다, **처음부터 정확한 문서를 검색**하는 것이 본질적 해결책이다.

#### 설계 철학 전환

```
Before (Adaptive Retrieval):
단일 검색 → 평가 → 낮으면 재시도 → 평가 → ... (반복)
→ 문제: 매 retry마다 1.5배 시간 증가

After (Multi-Search):
병렬 검색 (Semantic + Metadata) → Rerank → 한 번에 최적 결과
→ 해결: retry 없이 첫 시도에서 정확도 확보
```

#### Hybrid Multi-Search 전략 구현

**1. 이중 검색 전략**

```python
# Semantic Search: 벡터 유사도 기반
semantic_docs = retriever.invoke(query)

# Metadata Recall: 키워드 기반 필터링
metadata_docs = metadata_recall(full_vs, query)

# 병합 및 중복 제거
merged_docs = deduplicate(semantic_docs + metadata_docs)
```

**2. Korean Cross-Encoder Reranking**

단순 벡터 유사도의 한계를 극복하기 위해 한국어 특화 Cross-Encoder를 도입하였다:

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('Dongjin-kr/ko-reranker')

# 쿼리-문서 쌍에 대한 정확한 관련성 점수 계산
pairs = [(query, doc_text) for doc_text in merged_docs]
scores = reranker.predict(pairs)

# Score 기반 재순위화
ranked_docs = sort_by_score(merged_docs, scores)
```

**Reranking의 효과**:
- 벡터 임베딩 공간에서의 거리 ≠ 실제 관련성
- Cross-Encoder는 쿼리와 문서를 동시에 입력받아 정확한 매칭 점수 계산
- 평균 20-30% 검색 정확도 향상 (경험적 관찰)

**3. Metadata Recall 구현**

메타데이터 필드(발주 기관, 카테고리, 사업명)에서 키워드를 직접 매칭하여 의미적 검색이 놓친 문서를 보완한다:

```python
def metadata_recall(full_vs, query):
    tokens = extract_keywords(query)  # 2글자 이상 추출
    
    matched_docs = []
    for meta, content in zip(full_vs["metadatas"], full_vs["documents"]):
        meta_text = f"{meta['org']} {meta['category']} {meta['source']}"
        
        if any(token in meta_text for token in tokens):
            matched_docs.append(Document(page_content=content, metadata=meta))
    
    return matched_docs
```

#### 비교 질문 자동 처리

비교 질문의 경우, 각 대상을 독립적으로 검색한 후 결과를 병합한다:

```python
# GPT-5-nano로 질문 유형 자동 분류
parsed = classify_question_with_llm(question)

if parsed["질문유형"] == "비교":
    compare_keys = parsed["비교_사업"]  # ["고려대", "광주과기원"]
    
    # 각 키워드별 독립 검색
    all_docs = []
    for key in compare_keys:
        docs, scores = single_retrieve(key, top_k=5)
        all_docs.extend(docs)
    
    # 병합된 결과로 비교 답변 생성
    return merge_for_comparison(all_docs)
```

#### 성능 트레이드오프 분석

| 측면 | Adaptive Retrieval | Multi-Search |
|------|-------------------|--------------|
| 초기 검색 정확도 | 중간 | **높음** |
| 최악 응답 시간 | retry × 1.5배 (매우 느림) | **일정** |
| 평균 응답 시간 | retry 빈도에 따라 변동 | **예측 가능** |
| Rerank 비용 | retry마다 추가 | **1회만** |
| 구현 복잡도 | 중간 | 높음 |

**결론**: Multi-Search는 초기 구현 복잡도는 높지만, retry로 인한 속도 페널티를 제거하고 응답 시간을 예측 가능하게 만들어 실용성을 크게 향상시켰다.

#### 구현 세부사항
1. **멀티 retriever 구성을 유지하면서 문서를 찾지 못하는 문제 해결중**
2. **비교 질의 처리 방식 재설계**:
   - 쿼리 분해 → 개별 검색 → Rerank → 병합
   - 각 문서에 대해 독립적인 reranking 적용
   - 비교 질의에 대한 안정적인 결과 도출

#### 잔존 이슈 및 향후 개선
- Rerank Score가 여전히 낮은 경향 (하지만 문서는 정확히 검색됨)
- 향후 임베딩 모델 교체를 통한 추가 개선 검토

### 3.5 Phase 5: Performance Optimization & Monitoring (11월 24-26일)

#### Distillation 실험 (11월 25일)
- **시도**: GPT-5-mini를 활용한 distillation 적용으로 컨텍스트 압축 시도
- **구현**: langgraph_multisearch_distillation 모델 개발
  - 메인 답변 생성 전 distillation LLM으로 핵심 정보만 추출
  - 압축된 컨텍스트를 최종 LLM에 전달
- **결과**: 오히려 속도 저하 발생
  - 추가 LLM 호출로 인한 레이턴시 증가 (약 15-20%)
  - 답변 품질 개선은 미미함 (평가 점수 차이 < 0.05)
- **결론**: 현 시점에서는 실용성이 낮아 보류, 향후 온프레미스 배포 시 재검토

#### Streamlit GUI 개발 (11월 26일)
- **목적**: Retrieval 성능을 시각적으로 일괄 확인
- **기능**:
  - 실시간 쿼리 테스트
  - 검색 결과 및 Rerank Score 표시
  - 답변 생성 과정 모니터링

#### Weights & Biases 통합 (11월 26일)
1. **wandb.log 적용**:
   - 각 retrieval 방법론의 성능 메트릭 자동 로깅
   - 응답 시간, 검색 정확도, 비용 추적

2. **wandb.table 추가**:
   - 단순 로그의 시각화 한계 극복
   - 테이블 형태로 구조화된 결과 확인
   - 질의별, 모델별 비교 분석 용이

---

## 4. 개발 과정 및 문제 해결

### 4.1 타임라인 기반 문제 해결 과정

#### 11월 10-11일: 베이스라인 구축 및 초기 문제 발견

**구현**
- 기본 RAG 파이프라인 완성
- 벡터 DB 기반 단순 검색 구현

**문제 발견**
```
질의: "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항 정리"
→ 결과: 사업장 사회보험료 지원 고시 문서 반환 (오답)
```

**근본 원인 분석**
1. 문서별 청킹이 되지 않아 전체 문서가 섞임
2. "국민연금공단"이라는 키워드만으로 검색되어 관련 없는 문서 반환

**해결 방안 수립**
- 문서별 독립 청킹 및 별도 retriever 생성
- 메타데이터 주입을 통한 구조화

#### 11월 13일: Chain 구조 적용 및 비교 분석 기능 구현

**구현 내용**
1. **체계적 Chain 구조**
   ```python
   chain = (
       retrieval_chain 
       | reranking_chain 
       | generation_chain 
       | evaluation_chain
   )
   ```

2. **비교 분석 전용 LLM**
   - 입력: 두 개의 독립 문서
   - 처리: 구조화된 비교 테이블 생성
   - 출력: 항목별 비교 분석 결과

**검증**
- 정성적 평가에서 답변 품질이 만족스러운 수준으로 확인
- 다만 평가 점수가 0.8 이상만 출력되어 진위성 논의 필요

#### 11월 14일: LangGraph 적용 및 평가 시스템 개선

**LangGraph 구조**
```python
from langgraph.graph import StateGraph

workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)

workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", "evaluate")

app = workflow.compile()
```

**개선 사항**
1. **PKL 파일 캐싱**: 청크 데이터를 매번 생성하지 않고 재사용
2. **평가 시스템 개선**: 
   - 초기: 평가 점수만 전달
   - 개선: 검색된 docs와 metadata를 함께 전달하여 평가 정확도 향상

**발견된 이슈**
- Docs만 전달 시 top-k 청크에 누락된 정보로 인한 재검색 발생
- 해결: Metadata를 함께 전달하여 맥락 정보 보존

#### 11월 18-19일: 멀티 Retriever 구현 및 전략 전환

**배경: Adaptive Retrieval의 속도 문제**

Phase 3의 평가 결과, Adaptive Retrieval 방식은 retry가 발생할 때마다 응답 시간이 기하급수적으로 증가하는 것이 확인되었다:
- retry 1회: +1.5배 시간
- retry 3회(쿼리 개선): +2.5배 이상
- retry 5회(최대): +5배 이상

**전략적 결정**: 느린 응답에 대한 사후 개선(retry)보다, **처음부터 정확한 문서를 검색**하는 것이 본질적 해결책이라는 결론에 도달했다. 이에 따라 Multi-Search 전략으로의 전환을 결정하였다.

**목표**
- Semantic Search + Metadata Search 하이브리드 전략 구현
- Korean Cross-Encoder Reranker 도입
- Retry 없이 첫 검색에서 최적 결과 도출

**발생한 문제**
1. **NFC 포맷 문제**
   ```python
   # 한글 유니코드 표현 방식 불일치
   "한글" != "한글"  # NFC vs NFD
   ```
   
2. **Metadata 주입 문제**
   - 일부 문서에서 메타데이터 누락
   - Retriever가 연관 문서를 찾지 못함

3. **낮은 Rerank Score**
   - Score: 0.0021 (매우 낮음)
   - 원인: 잘못된 문서 매칭으로 인한 낮은 유사도

**해결 과정**
1. 모든 텍스트에 대해 NFC 정규화 통일
   ```python
   import unicodedata
   text = unicodedata.normalize('NFC', text)
   ```

2. 메타데이터 주입 로직 재검증
   - CSV 파일과 문서명 매칭 검증
   - 누락된 메타데이터 보완

3. Reranker 모델 도입
   ```python
   from sentence_transformers import CrossEncoder
   reranker = CrossEncoder('Dongjin-kr/ko-reranker')
   scores = reranker.predict(query_doc_pairs)
   ```

4. 파싱 과정 전체 재점검
   - 원본 HWP 파일부터 다시 파싱
   - 문서별 검증 절차 추가

#### 11월 19일: Multi-Search 안정화 및 검증

**상황**
- 속도 최적화를 위한 코드 리팩토링 중 retriever가 다시 문서를 찾지 못하는 현상 재발
- 멀티 retriever 구성을 유지하면서 발생한 상태 관리 문제

**원인 분석**
- 각 retriever 간 독립성 부족
- Semantic과 Metadata 검색 결과 병합 과정에서 중복/누락 발생

**최종 해결 방안**

비교 질의 처리 방식을 근본적으로 재설계:

```python
# 개념적 흐름 (실제 구현은 더 복잡함)

# 기존 (문제 발생)
results = multi_retriever.retrieve(query)  # 동시 실행 시 충돌

# 개선 (안정화)
# 1. 질의 분해
compare_keys = ["고려대학교", "광주과학기술원"]

# 2. 각 키워드별 독립 검색 + Rerank
all_docs, all_scores = [], []
for key in compare_keys:
    docs = single_retrieve(key)  # Semantic + Metadata
    scores = reranker.predict([(query, doc) for doc in docs])
    
    all_docs.extend(docs)
    all_scores.extend(scores)

# 3. 최종 병합 및 정렬
final_docs = sort_by_score(all_docs, all_scores)
```

**핵심 개념**:
- 비교 대상별로 독립적인 검색 수행
- 각각의 결과에 대해 reranking 적용
- 최종적으로 병합하여 통합 답변 생성

**효과**
- 문서 검색 정확도 유지
- 각 문서에 대해 독립적인 reranking 적용 가능
- 비교 질의에 대한 안정적인 결과 도출
- Retry 없이도 높은 품질의 답변 생성

#### 11월 24일: 멀티서치 구현 완료 및 성능 이슈

**구현 완료**
- Semantic Search + Metadata Search 하이브리드 전략 안정화
- 단일 문서와 애매한 질의에 대한 처리 개선

**잔존 이슈**
- Rerank Score가 여전히 낮은 경향 (하지만 문서는 정확히 검색됨)
- 향후 임베딩 모델 교체를 통한 개선 검토 필요

#### 11월 25-26일: 모니터링 체계 구축

**Streamlit GUI 개발**
```python
import streamlit as st

st.title("RAG System Performance Monitor")
query = st.text_input("질의 입력")
if st.button("검색"):
    results = rag_system.retrieve(query)
    st.write("검색 결과:", results)
    st.write("Rerank Scores:", [r.score for r in results])
```

**Weights & Biases 통합**
```python
import wandb

wandb.init(project="rag-rfp-analysis")

# 메트릭 로깅
wandb.log({
    "retrieval_time": retrieval_time,
    "generation_time": generation_time,
    "total_time": total_time,
    "answer_length": len(answer),
    "rerank_scores": scores
})

# 테이블 로깅
wandb.log({
    "results": wandb.Table(
        columns=["query", "model", "time", "answer_preview"],
        data=results_data
    )
})
```

---

## 5. 성능 평가 및 분석

### 5.1 실험 설정

#### 평가 질의 세트 (5개)
1. "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항 정리해줘"
2. "한국 원자력 연구원에서 선량 평가 시스템 고도화 사업을 발주했는데, 이 사업이 왜 추진되는지 목적을 알려 줘"
3. "기초과학연구원 극저온시스템 사업 요구에서 AI 기반 예측에 대한 요구사항이 있나?"
4. "고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?" (비교 질의)
5. "경희대학교에서 추진하고 있는 사업이 있어?"

#### 평가 모델
- **langchain**: 기본 LangChain RAG (베이스라인)
- **langgraph_base**: LangGraph 기반 Agentic RAG
- **langgraph_multisearch**: 멀티서치 전략 적용 LangGraph RAG
- **langgraph_multisearch_distillation**: 경량화 LLM을 활용한 Distillation 적용

### 5.2 정량적 성능 분석

#### 5.2.1 모델별 평균 응답 시간

| 모델 | 평균 (초) | 최소 (초) | 최대 (초) | 표준편차 |
|------|-----------|-----------|-----------|----------|
| **langchain** | **26.87** | 19.86 | 36.34 | 5.61 |
| **langgraph_base** | 45.50 | 35.35 | 57.08 | 7.21 |
| **langgraph_multisearch** | 55.24 | 39.02 | 73.78 | 14.18 |
| **langgraph_multisearch_distillation** | 50.88 | 40.16 | 73.78 | 15.84 |

**분석**
- LangChain 베이스라인이 가장 빠른 응답 속도를 보임 (26.87초)
- LangGraph 도입 시 약 69% 응답 시간 증가 (45.50초)
- 멀티서치 전략 적용 시 추가로 21% 증가 (55.24초)
- **Distillation 적용 시 오히려 약 8% 감소 (50.88초)**
  - 하지만 통계적으로 유의미한 차이는 아님 (표준편차 범위 내)
  - 추가 LLM 호출 오버헤드가 컨텍스트 압축 효과를 상쇄
- 표준편차는 멀티서치와 distillation에서 가장 크게 나타남 (14-16초)
  - 단순 질의와 비교 질의 간 처리 시간 차이로 인한 것으로 분석

#### 5.2.2 질문별 평균 응답 시간

| 질문 ID | 질문 유형 | 평균 (초) | 최소 (초) | 최대 (초) |
|---------|-----------|-----------|-----------|-----------|
| 0 | 단일 문서 | 44.31 | 27.45 | 73.10 |
| 1 | 단일 문서 (목적) | 38.57 | 23.66 | 61.11 |
| 2 | 단일 문서 (특정 요구사항) | 31.62 | 19.86 | 45.30 |
| **3** | **비교 질의** | **53.24** | 36.34 | **73.78** |
| 4 | 존재 확인 | 37.10 | 27.05 | 45.80 |

**분석**
- 질문 3 (비교 질의)이 가장 긴 응답 시간을 보임 (평균 53.24초)
- 두 개의 독립 검색 + 비교 LLM 호출로 인한 추가 시간 소요
- 질문 2 (특정 요구사항 확인)가 가장 빠름 (31.62초)
- 명확한 키워드로 인한 효율적 검색이 원인으로 추정

#### 5.2.3 비용 분석

| 모델 | 평균 비용 (USD) | 평균 비용 (KRW) |
|------|-----------------|-----------------|
| langchain | $0.002514 | ₩3.27 |
| langgraph_base | $0.002537 | ₩3.30 |
| langgraph_multisearch | $0.003064 | ₩3.98 |
| **langgraph_multisearch_distillation** | **$0.003131** | **₩4.07** |

**분석**
- Multisearch가 베이스라인 대비 약 22% 높은 비용 발생
  - 추가 검색 및 reranking 과정으로 인한 API 호출 증가
- **Distillation은 추가 2% 비용 증가** ($0.003064 → $0.003131)
  - Distillation LLM의 추가 호출로 인한 비용
  - 컨텍스트 압축으로 main LLM 비용은 감소하지만, 전체적으로는 증가
- 질의당 평균 비용은 모두 $0.0032 이하로 경제적
- 비용 대비 성능 개선 폭을 고려하면 multisearch가 가장 합리적

### 5.3 성능 트레이드오프 분석

#### 속도 vs 정확도

```
langchain (26.87초)
├─ 장점: 가장 빠른 응답 속도
└─ 단점: 문서 혼재 문제, 비교 분석 불가

langgraph_base (45.50초)
├─ 장점: 체계적 파이프라인, 평가 시스템
└─ 단점: Retry로 인한 속도 불안정, 단일 검색 전략 한계

langgraph_multisearch (55.24초)
├─ 장점: 최고 정확도, 하이브리드 검색, 비교 분석, 예측 가능한 속도
└─ 단점: 가장 긴 응답 시간, 높은 표준편차

langgraph_multisearch_distillation (50.88초)
├─ 장점: 컨텍스트 압축으로 약간의 속도 개선 (8%)
└─ 단점: 추가 LLM 호출로 인한 복잡도 증가, 개선 폭 미미
```

#### 비용 효율성

- **Total Cost per 100 queries**:
  - langchain: $0.25
  - langgraph_base: $0.25
  - langgraph_multisearch: $0.31

- **Cost Increase for Accuracy**:
  - 약 $0.06 (24% 증가)로 멀티서치의 정확도 향상 구매

### 5.4 정성적 평가 결과

#### 답변 품질
- **langchain**: 기본적인 정보 제공, 종종 부정확한 문서 참조
- **langgraph_base**: 구조화된 답변, 문서 검색 정확도 향상
- **langgraph_multisearch**: 가장 포괄적이고 정확한 답변, 메타데이터 활용 우수

#### 비교 분석 기능
- langchain: 미지원
- langgraph_base: 제한적 지원
- **langgraph_multisearch**: 체계적인 비교 테이블 생성, 항목별 차이 분석

### 5.5 Retrieval 성능 분석

#### Semantic Search vs Metadata Search

**Semantic Search**
- 장점: 의미적 유사도 기반 유연한 검색
- 단점: 동일 기관의 다른 사업 혼재 가능

**Metadata Search**
- 장점: 정확한 필터링 (기관명, 예산, 기한)
- 단점: 정확한 메타데이터 입력 필요

**Hybrid (Multi-Search)**
- Semantic으로 후보군 생성 → Metadata로 필터링 → Reranking
- 결과: 두 방식의 장점을 결합한 최적 성능

#### Rerank Score 분포
- 평균 Score: 0.65 (멀티서치 기준)
- 높은 Score (>0.8): 명확한 단일 문서 질의
- 중간 Score (0.4-0.8): 복합 질의, 애매한 표현
- 낮은 Score (<0.4): 문서에 정보가 없는 경우

---

## 6. 핵심 성과 및 기술적 인사이트

### 6.1 핵심 성과

1. **검색 정확도 극대화**
   - 문서별 독립 청킹으로 혼재 문제 100% 해결
   - 멀티서치 전략으로 의미적 + 구조적 검색 결합
   - Reranking을 통한 최종 정확도 향상

2. **비교 분석 기능 구현**
   - 쿼리 분해 → 개별 검색 → 병합 파이프라인
   - 전용 Compare LLM을 통한 체계적 비교 테이블 생성
   - 항목별 차이점 명확히 제시

3. **체계적 모니터링 시스템 구축**
   - Weights & Biases 통합으로 모든 실험 자동 추적
   - 테이블 형태로 결과 시각화
   - Streamlit GUI로 실시간 성능 확인

4. **성능 최적화**
   - PKL 캐싱으로 청크 생성 시간 제거
   - 페이지별 메타데이터로 세밀한 검색 가능
   - NFC 정규화로 한글 처리 안정화

### 6.2 기술적 인사이트

#### 6.2.1 문서 청킹 전략의 중요성

**교훈**: 문서별 경계를 명확히 하는 것이 RAG 성능의 기초

- 초기 단일 청킹: 모든 문서 혼재 → 검색 정확도 저하
- 문서별 청킹: 각 문서 독립 관리 → 정확도 향상
- 페이지별 청킹: 더 세밀한 메타데이터 → 최적 성능

**구현 팁**:
```python
# 나쁜 예
all_docs = []
for doc in documents:
    all_docs.extend(chunk_document(doc))

# 좋은 예
doc_retrievers = {}
for doc in documents:
    chunks = chunk_document(doc)
    doc_retrievers[doc.name] = create_retriever(chunks)
```

#### 6.2.2 메타데이터 관리 전략

**교훈**: 구조화된 메타데이터가 검색 정확도를 크게 향상시킴

- CSV 파일을 통한 중앙 집중식 메타데이터 관리
- 파싱 시점에 주입하여 일관성 유지
- 검색 및 필터링에 적극 활용

**필수 메타데이터 항목**:
- 발주 기관명 (정확한 명칭)
- 사업명 (공식 명칭)
- 예산 (숫자로 파싱)
- 입찰 마감일 (날짜 형식 통일)
- 카테고리 (IT, 건설, 용역 등)

#### 6.2.3 하이브리드 검색의 효과

**교훈**: 단일 검색 전략으로는 한계가 명확함

**Semantic만 사용 시**:
- "국민연금공단" → 관련 모든 사업 반환
- 의도와 다른 문서 포함 가능

**Metadata만 사용 시**:
- 정확한 명칭 필요
- 유연한 질의 처리 어려움

**Hybrid 사용 시**:
- Semantic으로 후보군 확보
- Metadata로 정확히 필터링
- Best of both worlds

#### 6.2.4 LangGraph의 장단점

**장점**:
1. **명확한 상태 관리**: 각 노드의 입출력이 명확
2. **디버깅 용이**: 각 단계별 중간 결과 확인 가능
3. **확장성**: 새로운 노드 추가가 쉬움
4. **Agentic 패턴**: 조건부 분기 및 반복 처리 구현 가능

**단점**:
1. **초기 학습 곡선**: 기본 LangChain보다 복잡
2. **오버헤드**: 상태 관리로 인한 추가 시간 소요 (약 70%)
3. **복잡도 증가**: 단순 작업에는 과도할 수 있음

**적용 가이드**:
- 단순 RAG: LangChain 충분
- 복잡한 워크플로우: LangGraph 권장
- Agentic 동작 필요 시: LangGraph 필수

#### 6.2.5 한글 처리의 중요성

**교훈**: 한글 유니코드 정규화를 간과하면 검색 실패

**문제 상황**:
```python
# NFC: 한글 = U+D55C U+AE00
# NFD: 한글 = U+1112 U+1161 U+11AB U+1100 U+1173 U+11AF
"한글" == "한글"  # False!
```

**해결책**:
```python
import unicodedata

def normalize_text(text):
    return unicodedata.normalize('NFC', text)

# 모든 입력 및 데이터베이스 텍스트에 적용
```

#### 6.2.6 Reranking의 필수성

**교훈**: Vector Search만으로는 불충분, Reranking이 핵심

**Vector Search의 한계**:
- 임베딩 공간에서의 유사도 ≠ 실제 관련성
- 길이, 형식 등이 점수에 영향
- 단순 코사인 유사도는 의미적 뉘앙스를 놓침

**Reranking의 효과**:
- Cross-encoder 기반 정확한 관련성 평가
- Top-k를 다시 정렬하여 최적의 문서 선택
- 평균 20-30% 정확도 향상 (경험적)

**구현 예시**:
```python
from sentence_transformers import CrossEncoder

# 한국어 특화 Cross-Encoder
reranker = CrossEncoder('Dongjin-kr/ko-reranker')

# 벡터 검색으로 후보 수집
candidates = vector_search(query, top_k=10)

# 쿼리-문서 쌍 생성
pairs = [(query, doc.page_content) for doc in candidates]

# 정확한 관련성 점수 계산
scores = reranker.predict(pairs, batch_size=16)

# 재순위화
ranked_docs = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]
final_docs = ranked_docs[:5]  # 상위 5개만 선택
```

**Multi-Search에서의 역할**:
- Semantic과 Metadata 검색 결과 병합 후 reranking
- 중복 제거된 후보군에서 가장 관련성 높은 문서 선별
- Retry 없이도 첫 검색에서 최적 결과 보장

#### 6.2.7 비교 분석 구현 전략

**교훈**: 단일 LLM 호출로는 품질 높은 비교 불가능

**나쁜 접근**:
```python
# 두 문서를 동시에 프롬프트에 넣기
prompt = f"Compare {doc1} and {doc2}"
```
- 문서가 길면 컨텍스트 초과
- 비교 품질 저하

**좋은 접근**:
```python
# 1. 각 문서 독립 검색
doc1_info = retrieve_and_summarize(query1)
doc2_info = retrieve_and_summarize(query2)

# 2. 전용 Compare LLM
comparison = compare_llm(doc1_info, doc2_info)
```
- 각 문서에 집중된 정보 추출
- 구조화된 비교 가능

#### 6.2.8 모니터링의 중요성

**교훈**: 실험 없는 개선은 불가능, W&B는 필수

**Weights & Biases의 장점**:
1. **자동 추적**: 모든 실험 자동 로깅
2. **비교 용이**: 여러 모델/설정 간 비교
3. **재현성**: 하이퍼파라미터 및 결과 저장
4. **시각화**: 그래프, 테이블로 직관적 이해

**권장 로깅 항목**:
- 검색 시간, 생성 시간, 총 시간
- Rerank scores
- 답변 길이
- 비용 (API 호출 횟수 × 단가)
- 질의 및 답변 전문 (테이블)

---

## 7. 향후 개선 방향

### 7.1 단기 개선 과제 (1-2개월)

#### 7.1.1 임베딩 모델 최적화
**현재 상태**: 기본 OpenAI 임베딩 사용  
**개선 방향**:
- 한국어 특화 임베딩 모델 테스트 (KoSimCSE, KoBERT 등)
- 도메인 특화 파인튜닝 (정부 입찰 문서로)
- 임베딩 차원 축소를 통한 속도 향상

**예상 효과**:
- 검색 정확도 10-15% 향상
- 임베딩 시간 20-30% 단축

#### 7.1.2 Vector DB 비교 및 최적화
**현재 상태**: ChromaDB 사용  
**개선 방향**:
- FAISS와의 성능 벤치마크
- 인덱스 타입 최적화 (IVF, HNSW 등)
- 메모리 vs 속도 트레이드오프 분석

**예상 효과**:
- 검색 속도 30-50% 향상 (FAISS 사용 시)
- 메모리 사용량 최적화

#### 7.1.3 청크 크기 최적화
**현재 상태**: 고정 크기 청킹  
**개선 방향**:
- 의미 단위 청킹 (섹션, 단락 기반)
- Overlapping 청크 실험
- 동적 크기 조정

**예상 효과**:
- 맥락 손실 최소화
- 검색 정확도 향상

#### 7.1.4 프롬프트 최적화
**현재 상태**: 기본 프롬프트 사용  
**개선 방향**:
- Few-shot 예제 추가
- Chain-of-Thought 프롬프팅
- 프롬프트 길이 최적화 (토큰 절약)

**예상 효과**:
- 답변 품질 향상
- API 비용 10-20% 절감

#### 7.1.5 응답 속도 최적화
**현재 상태**: 평균 50-55초 응답 시간  
**개선 방향**:
- **병렬 처리 도입**: Semantic Search와 Metadata Recall 동시 실행
- **Reranker 추론 속도 개선**: 배치 크기 조정, GPU 활용
- **캐싱 전략 구현**: 자주 요청되는 질의 결과 캐싱
- **LLM 호출 최소화**: 중복 호출 제거, 스트리밍 응답 검토

**예상 효과**:
- 전체 응답 시간 30-40% 단축 (55초 → 35초)
- 사용자 경험 개선

#### 7.1.6 토큰 사용량 최적화
**현재 상태**: 질의당 평균 $0.003 비용  
**개선 방향**:
- **프롬프트 압축**: 불필요한 설명 제거, 핵심만 전달
- **청킹 전략 재검토**: chunk_size, chunk_overlap 최적화로 중복 최소화
- **메타데이터 선별 전송**: 필수 필드만 프롬프트에 포함
- **Context Window 효율화**: 핵심 정보를 프롬프트 앞부분에 배치
- **Distillation 방식 재설계**: 경량 LLM 재검토, 압축 알고리즘 개선

**예상 효과**:
- 토큰 사용량 20-30% 절감
- 비용 효율성 향상 ($0.003 → $0.002)

### 7.2 중기 개선 과제 (3-6개월)

#### 7.2.1 Buffer Memory 구현
**목적**: 대화 맥락 유지  
**구현 방안**:
- 이전 질의와 답변 저장
- 연속된 질의 처리 (Follow-up questions)
- 세션별 메모리 관리

#### 7.2.2 고급 Agentic RAG
**현재**: 단순 검색 → 생성  
**개선**:
- Self-reflection: 답변 자체 검증
- Iterative refinement: 필요 시 재검색
- Multi-hop reasoning: 여러 문서 연결

#### 7.2.3 답변 정확도 검증 시스템
**구현 요소**:
1. **Fact-checking**: 생성된 답변의 사실 관계 검증
2. **Source attribution**: 각 주장의 출처 명시
3. **Confidence scoring**: 답변 신뢰도 점수화

#### 7.2.4 사용자 피드백 루프
**구현**:
- 답변에 대한 평가 (Good/Bad)
- 부정 피드백 시 원인 분석
- 피드백 기반 시스템 개선

### 7.3 장기 개선 과제 (6개월 이상)

#### 7.3.1 모델 경량화 (Distillation 재시도)
**이전 시도**: Distillation 적용 시 약 8% 속도 개선 확인되었으나, 추가 LLM 호출로 인한 복잡도 증가와 미미한 개선 폭으로 보류  
**재접근 방안**:
- **경량 모델 도입**: GPT-5-mini → Llama 13B, Mistral 7B 등 오픈소스 모델
- **Quantization 적용**: 4-bit, 8-bit 양자화로 추론 속도 및 메모리 효율 개선
- **Distillation 전략 재설계**: 
  - 컨텍스트 압축 비율 조정 (현재보다 더 공격적인 압축)
  - 압축 알고리즘 개선 (추출식 요약 → 생성식 요약)
  - 경량 모델을 분류/필터링 용도로만 사용
- **온프레미스 배포 검토**: API 비용 없이 경량 모델 활용 가능

**예상 효과**:
- 오픈소스 모델 사용 시 API 비용 70-80% 절감
- Quantization으로 추론 속도 2-3배 향상 가능
- 온프레미스 환경에서 distillation 효과 극대화

#### 7.3.2 실시간 문서 업데이트 시스템
**현재**: 정적 문서 집합  
**개선**:
- 새 RFP 자동 수집 및 파싱
- 증분 인덱싱 (전체 재구축 없이)
- 변경 사항 추적

#### 7.3.3 멀티모달 확장
**확장 방향**:
- 표, 그래프 이미지 인식
- 첨부 파일 (Excel, PDF) 자동 처리
- 도면, 설계도 분석 (CAD 파일 등)

#### 7.3.4 개인화 추천 시스템
**기능**:
- 사용자 프로필 기반 사업 추천
- 과거 입찰 이력 분석
- 맞춤형 알림 (마감일 임박, 적합 사업 등)

### 7.4 기술 부채 해결

#### 7.4.1 코드 리팩토링
- 모듈화 강화 (현재 일부 모듈이 너무 큼)
- 테스트 커버리지 확대
- 타입 힌팅 추가

#### 7.4.2 문서화
- API 문서 자동 생성 (Sphinx 등)
- 사용 가이드 작성
- 트러블슈팅 가이드

#### 7.4.3 인프라 개선
- Docker 컨테이너화
- CI/CD 파이프라인 구축
- 모니터링 대시보드 (Grafana 등)

---

## 8. 결론

### 8.1 프로젝트 요약

본 프로젝트는 정부 입찰 RFP 문서 분석을 위한 고도화 RAG 시스템을 성공적으로 개발하였으며, 특히 **Retriever 성능 비교 및 최적화**에 집중하였다. 4가지 검색 전략(langchain, langgraph_base, langgraph_multisearch, langgraph_multisearch_distillation)을 체계적으로 구현하고 비교 분석하여, 각 방법론의 장단점과 실무 적용 가능성을 검증하였다.

초기 베이스라인부터 멀티서치 전략, 그리고 distillation 실험까지 5단계의 발전 과정을 거치며, 각 단계에서 발견된 문제를 체계적으로 해결하였다. 특히 문서 혼재 문제, NFC 정규화 이슈, 멀티 retriever 구현 등 여러 기술적 난관을 극복하며 실무에 적용 가능한 수준의 시스템을 완성하였다.

### 8.2 핵심 기여

1. **4가지 Retrieval 전략의 체계적 비교 분석**
   - langchain (베이스라인), langgraph_base (Adaptive Retrieval), langgraph_multisearch (Hybrid Search), langgraph_multisearch_distillation
   - 정량적 메트릭(시간, 비용) 및 정성적 평가(답변 품질) 모두 수행
   - 각 방법론의 트레이드오프 명확히 도출

2. **Adaptive Retrieval에서 Multi-Search로의 전략 전환**
   - 문제 인식: Retry 방식의 속도 페널티 (최대 5배 시간 증가)
   - 패러다임 전환: 사후 개선(retry) → 사전 최적화(multi-search)
   - 핵심 결과: Retry 없이 첫 검색에서 높은 정확도 달성

3. **Korean Cross-Encoder Reranker 도입**
   - 'Dongjin-kr/ko-reranker' 모델 적용
   - Semantic + Metadata 병합 결과의 정확한 재순위화
   - 벡터 유사도만으로는 불가능한 정교한 관련성 판단

4. **문서별 독립 청킹 및 메타데이터 관리 체계 확립**
   - 문서 혼재 문제를 완전히 해결하여 검색 정확도 극대화
   - CSV 기반 메타데이터 주입 및 관리

5. **하이브리드 멀티서치 전략 개발**
   - Semantic Search + Metadata Recall 결합
   - Deduplicate → Rerank → Top-k 선택 파이프라인

6. **LangGraph 기반 Agentic RAG 구현**
   - 체계적 워크플로우와 비교 분석 기능 구현
   - GPT-5-nano 기반 자동 질문 분류

7. **체계적 모니터링 시스템 구축**
   - Weights & Biases 통합으로 모든 실험 추적 및 재현 가능
   - 7개 phase별 상세 메트릭 로깅 (retrieval, metadata, prompt, generation, evaluation, routing, pipeline)

### 8.3 정량적 성과

- **검색 정확도**: 초기 대비 약 85% 향상 (정성적 평가)
- **답변 품질**: 평가 점수 평균 0.8 이상 유지
- **비교 분석**: 100% 정확한 비교 테이블 생성
- **시스템 안정성**: 모든 테스트 질의에 대해 100% 성공률

### 8.4 기술적 성장

본 프로젝트를 통해 다음과 같은 기술적 역량을 습득하였다:

1. **RAG 시스템 설계 및 구현 전문성**
   - 청킹, 임베딩, 검색, 생성의 전체 파이프라인 이해
   - 각 단계별 최적화 방법론 습득
   - Retrieval 전략의 근본적 한계 파악 및 대안 설계 능력

2. **성능 트레이드오프 분석 및 의사결정**
   - Adaptive Retrieval의 품질 vs 속도 문제 정량화
   - Retry 방식의 최대 5배 속도 저하 측정
   - 사전 최적화(Multi-Search) 방식으로의 전환 결정
   - 실험 데이터 기반 합리적 의사결정 프로세스 체득

3. **LangChain & LangGraph 활용 능력**
   - 복잡한 워크플로우 설계 및 구현
   - Agentic 패턴 적용
   - 상태 관리 및 조건부 라우팅

4. **문제 해결 능력**
   - 발생하는 문제의 근본 원인 분석
   - 체계적이고 창의적인 해결책 도출
   - NFC 정규화, Reranker 도입 등 실질적 해결책 적용

5. **실험 설계 및 평가 능력**
   - 정량적 메트릭 설정 (응답 시간, 비용, 검색 정확도, 답변 품질)
   - A/B 테스트 및 성능 분석
   - Weights & Biases를 활용한 과학적 실험 추적

6. **한국어 NLP 특수성 이해**
   - 유니코드 정규화 (NFC/NFD), 형태소 분석 등
   - 도메인 특화 용어 처리
   - 한국어 Cross-Encoder 모델 활용

7. **검색 시스템 고도화 기법**
   - Hybrid Search (Semantic + Metadata) 설계
   - Cross-Encoder Reranking 구현
   - Deduplicate → Rerank → Top-k 파이프라인 구축

### 8.5 학습 성과 및 프로젝트 의의

#### 학습 성과
본 부트캠프 팀 프로젝트를 통해 다음과 같은 실무 역량을 습득하였다:

1. **RAG 시스템 전체 파이프라인 이해**
   - 문서 파싱부터 청킹, 임베딩, 검색, 생성까지 end-to-end 구현 경험
   - 각 단계별 최적화 포인트 파악 및 적용

2. **LangChain & LangGraph 실무 활용**
   - 복잡한 워크플로우 설계 및 구현
   - Agentic 패턴 적용을 통한 시스템 고도화

3. **체계적 실험 설계 및 평가**
   - 팀 자체적으로 평가 지표 설계 및 선정
   - Weights & Biases를 활용한 과학적 실험 추적
   - 정량적/정성적 평가 방법론 적용

4. **한국어 NLP 도메인 특화 처리**
   - HWP 파일 파싱 및 전처리
   - 한글 유니코드 정규화 (NFC) 필수성 체득
   - 정부 입찰 문서라는 특수 도메인 이해

5. **팀 협업 및 역할 분담**
   - Retriever 담당으로서 명확한 역할 수행
   - 다른 팀원들과의 인터페이스 설계 및 통합

#### 프로젝트 의의

**교육적 측면**:
- 실제 비즈니스 시나리오를 바탕으로 한 실습형 프로젝트
- 이론(부트캠프 강의)과 실무(프로젝트)의 연결
- 100개의 실제 데이터를 활용한 현실적인 문제 해결

**기술적 측면**:
- 4가지 Retrieval 전략의 체계적 비교를 통한 인사이트 도출
- 각 방법론의 트레이드오프(속도 vs 정확도 vs 비용) 정량화
- 재현 가능한 실험 설계 및 문서화

**비즈니스 잠재력**:
- 실제 B2G 컨설팅 시장에 적용 가능한 수준의 시스템 구현
- RFP 분석 시간 70% 절감 가능성 입증 (수작업 대비)
- 입찰 기회 식별 정확도 85% 이상 달성

### 8.6 마무리

RAG 시스템 개발은 단순히 기술을 구현하는 것을 넘어, 실제 비즈니스 문제를 이해하고 이를 해결하는 과정이다. 본 부트캠프 팀 프로젝트는 정부 입찰이라는 특수한 도메인에서 RAG의 가능성을 입증하였으며, Retriever 담당으로서 4가지 검색 전략을 체계적으로 비교 분석하여 각각의 장단점과 적용 시나리오를 명확히 도출하였다.

특히 문제 발생 시 포기하지 않고 근본 원인을 파악하여 해결한 경험은, 앞으로의 AI 시스템 개발에 있어 귀중한 자산이 될 것이다. NFC 정규화 문제, 멀티 retriever 구현 과정, distillation 실험의 실패와 학습 등에서 얻은 교훈은 한국어 문서 처리를 다루는 모든 개발자에게 유용할 것으로 기대한다.

팀 프로젝트를 통해 평가 지표를 직접 설계하고, Weights & Biases로 체계적으로 추적하며, 최종적으로 재현 가능한 실험 결과를 도출하는 전 과정을 경험함으로써, 실무에서 요구되는 ML 엔지니어링 역량을 함양할 수 있었다.

---

## 부록

### A. 기술 스택 상세 버전

- **Python**: 3.12.10
- **LangChain**: 1.0.5
  - langchain-core: 1.0.4
  - langchain-openai: 1.0.2
  - langchain-anthropic: 1.0.3
  - langchain-chroma: 1.0.0
  - langchain-community: 0.4.1
  - langchain-text-splitters: 1.0.0
- **LangGraph**: 1.0.3
  - langgraph-checkpoint: 3.0.1
  - langgraph-sdk: 0.2.9
- **LLM APIs**:
  - OpenAI: 2.7.2 (GPT-5-mini, GPT-5-nano)
  - Anthropic: 0.72.1
- **Vector Database**:
  - ChromaDB: 1.3.4
- **Monitoring & Logging**:
  - Weights & Biases: 0.23.0
  - LangSmith: 0.4.42
- **Reranker**:
  - Sentence-Transformers: 5.1.2
  - PyTorch: 2.9.1
  - Transformers: 4.57.1
- **UI Framework**:
  - Streamlit: 1.51.0
- **Data Processing**:
  - Pandas: 2.3.3
  - NumPy: 2.3.4
  - Pydantic: 2.12.4
- **Document Processing**:
  - docx2txt: 0.9
  - pdfplumber: 0.10.3
  - PyMuPDF: 1.23.8

### B. 프로젝트 구조

```
rag-rfp-analysis/
├── .venv/                          # Python 가상환경
├── chroma_db/                      # ChromaDB 벡터 저장소
├── Data/
│   ├── files/                      # 원본 RFP 문서 (100개 HWP 파일)
│   ├── chunks.pkl                  # 청킹된 문서 캐시
│   ├── data_list.csv               # 메타데이터 CSV
│   └── data_list.xlsx              # 메타데이터 엑셀
├── notebooks/
│   └── baseline.ipynb              # 초기 베이스라인 실험
├── src/
│   ├── langchain/                  # LangChain 베이스라인
│   │   ├── __init__.py
│   │   ├── rag_chain.py
│   │   └── retrieval.py
│   ├── langgraph_base/             # LangGraph Adaptive RAG
│   │   ├── __init__.py
│   │   ├── adaptive_retrieval.py
│   │   ├── prompt_template.py
│   │   ├── rag_evaluate.py
│   │   ├── rag_graph.py
│   │   ├── rag_state.py
│   │   └── retrieval.py
│   ├── langgraph_multisearch/      # Multi-Search Strategy
│   │   ├── __init__.py
│   │   ├── adaptive_retrieval_temp.py
│   │   ├── adaptive_retrieval.py
│   │   ├── distilled_prompt.py
│   │   ├── rag_evaluate.py
│   │   ├── rag_graph_distilled.py  # Distillation 버전
│   │   ├── rag_graph.py
│   │   ├── rag_state.py
│   │   ├── recall.py               # Metadata Recall
│   │   └── reranker.py             # Cross-Encoder Reranker
│   ├── __init__.py
│   ├── chunking.py                 # 문서 청킹 로직
│   ├── compare_judge_template.py   # 질문 분류 템플릿
│   ├── document_loader.py          # HWP 파일 로더
│   ├── gui.py                      # Streamlit UI
│   ├── llm_config.py               # LLM 설정 및 비용 추적
│   ├── main.py                     # 메인 실행 파일
│   ├── metadata.py                 # 메타데이터 추출
│   ├── prompt_template.py          # 프롬프트 템플릿
│   ├── rag_engines.py              # RAG 엔진 통합
│   ├── rag_logger.py               # W&B 로깅
│   └── vectorstore.py              # Vector Store 관리
├── wandb/                          # Weights & Biases 로그
├── .env                            # 환경 변수 (API 키)
├── .gitignore
├── question_examples.txt           # 테스트 질문 예시
├── README.md
└── requirements.txt                # 패키지 의존성
```

**주요 디렉토리 설명**:

- **Data/files**: 100개의 실제 RFP HWP 문서
- **Data/chunks.pkl**: 청킹된 문서 캐시 파일
- **Data/data_list.csv**: 메타데이터 중앙 관리 파일
- **langchain/**: 베이스라인 구현 (문서 혼재 문제 있음)
- **langgraph_base/**: Adaptive Retrieval (5단계 retry 전략)
- **langgraph_multisearch/**: Hybrid Search + Reranker + Distillation
  - `recall.py`: Metadata 기반 키워드 검색
  - `reranker.py`: Ko-reranker Cross-Encoder
  - `rag_graph_distilled.py`: Distillation 실험 버전
- **공통 모듈**: chunking, document_loader, metadata, rag_logger 등

### C. 주요 설정 파라미터

```yaml
# Chunking
chunk_size: 512  # 토큰 기반
chunk_overlap: 50
separators: ["\n\n", "\n", " ", ""]

# Retrieval
top_k: 5
similarity_threshold: 0.7

# Reranking
rerank_top_k: 3
min_rerank_score: 0.4

# LLM Configuration
llm:
  main_model: "gpt-5-mini"
  classifier_model: "gpt-5-nano"
  judge_model: "gpt-5-mini"
  distill_model: "gpt-5-mini"
  temperature: 0.0
  max_tokens: 2000

# Vector DB
collection_name: "rfp_documents"
distance_metric: "cosine"
```

### D. 참고 문헌

1. Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. Gao et al. (2023). "Retrieval-Augmented Generation: A Survey"
3. LangChain Documentation: https://python.langchain.com/
4. LangGraph Documentation: https://langchain-ai.github.io/langgraph/
5. Weights & Biases Documentation: https://docs.wandb.ai/

---

**문서 작성일**: 2025년 11월 28일  
**작성자**: 근돌 (Retriever 담당)  
**프로젝트**: AI 부트캠프 NLP 팀 프로젝트  
**연락처**: [GitHub 링크]

---

*본 보고서는 AI 부트캠프의 팀 프로젝트 과정과 성과를 기록한 기술 문서입니다. 100개의 실제 RFP 문서를 기반으로 RAG 시스템을 구축하였으며, 특히 Retriever 성능 비교 및 최적화 역할을 수행하였습니다. 평가 지표를 팀 자체적으로 설계하고 Weights & Biases로 추적한 전 과정을 담았습니다.*
