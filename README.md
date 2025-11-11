# 입찰메이트 RFP RAG 시스템
**코드잇 AI 4기 4팀 - 공공입찰 컨설팅 스타트업 ‘입찰메이트(BidMate)’ AI 엔지니어링 팀**



## 📋 목차
- [프로젝트 개요](#프로젝트-개요)<br/>
- [핵심 기능](#핵심-기능)<br/>
- [팀 구성](#팀-구성)<br/>
- [기술 스택](#기술-스택)<br/>
- [시작하기](#시작하기)<br/>
- [개발 일정](#개발-일정)<br/>
- [성과 및 결과](#성과-및-결과)<br/>

---

## 🎯 프로젝트 개요

### 미션
자연어 기반 질의를 통해 **공공기관의 RFP(제안요청서)** 문서에서  
핵심 정보(사업명, 예산, 요구사항, 제출일 등)를 **자동 추출·요약·검색**하는  
**Retrieval-Augmented Generation (RAG)** 시스템을 개발합니다.

### 배경
공공입찰 컨설팅 스타트업 *입찰메이트(BidMate)* 는  
매일 수백 건의 RFP 문서를 검토해야 하며, 컨설턴트가 문서 전체를 수동으로 읽는 데 어려움을 겪고 있습니다.  
본 시스템은 RFP 분석을 자동화하여 **입찰 기회 탐색 및 컨설팅 효율을 향상**시키는 것을 목표로 합니다.

### 프로젝트 제약사항
- **개발 기간**: 2025.11.10 ~ 2025.11.28 (18일)
- **팀 구성**: 4명 (PM 1명 + 전문가 3명)
- **데이터**: 100개 RFP 문서(hwp/pdf) + 메타데이터(csv)
- **평가**: 팀별 보고서 및 발표 (정확도, 응답 품질, 응답 속도 중심)

---

## ✨ 핵심 기능

```mermaid
graph LR
    A["사용자 질의 (예: '한국전력의 AI 기반 예측 요구사항 요약')"] --> B["문서 파싱 및 청킹"]
    B --> C["임베딩 생성 및 Vector DB 저장"]
    C --> D["유사 문서 검색 (Retrieval)"]
    D --> E["LLM 응답 생성 (RAG Pipeline)"]
    E --> F["결과 요약 및 출력"]
```

- **문서 파싱**: PDF/HWP 포맷 자동 텍스트 변환
- **청킹 전략**: 문단 기반 중첩 청킹 (최적 토큰 단위 실험)
- **Retrieval:** FAISS + MMR 기반 하이브리드 검색
- **Generation**: GPT-5-mini / Gemma 2B 등 비교 실험
- **대화형 질의응답**: 후속 질문에서도 맥락 유지
- **결과 요약**: 핵심 요구조건·예산·대상기관 자동 추출

## 👥 팀 구성

| 역할 | 담당자 | 핵심 업무 |
|------|--------|-----------|
| **Project Manager** | 김민혁 | 프로젝트의 협업 과정을 매니징, 애자일/스프린트 방식으로 단위를 분리하고 회의를 주도, 성능 평가 |
| **데이터 처리 담당** | 이재영 | 문서별 메타데이터 처리, 문서 청킹 전략 설계 및 검증 |
| **Retrieval 담당** | 이현석 | 임베딩 생성 및 Vector DB 구축, 메타데이터 필터링 및 Retrieval 기법 고도화 |
| **Generation 담당** | 김남중 | 모델 선정 및 텍스트 생성 옵션 설정, 모델 응답 고도화, 대화 흐름 유지 및 히스토리 반영 전략 설계 |

## 🛠 기술 스택

**AI Frameworks**
- **LLM: OpenAI GPT-5-mini / HuggingFace LLaMA 3**
- **Embedding: text-embedding-3-small / all-MiniLM-L6-v2**
- **Vector DB: FAISS, Chroma**
- **Pipeline: LangChain, LlamaIndex**

**Backend & Infra**
- **Python 3.10+, FastAPI, GCP L4 GPU VM**
- **Supabase (vector API), Docker, GitHub Actions**

**Evaluation Tools**
- **BLEU / ROUGE / F1**
- **Human Eval (응답 품질)**
- **Latency 측정**

## 🚀 시작하기

### 시스템 요구사항
```yaml
Hardware:
  GPU: "NVIDIA RTX 3060 이상 (선택 사항, HuggingFace 모델 로컬 실행 시 필요)"
  CPU: "Intel i7 또는 AMD Ryzen 5 이상"
  RAM: "16GB 이상 권장 (벡터 검색 및 캐시 효율성 확보)"
  Storage: "20GB 이상 (임베딩 및 문서 캐시 저장용)"

Software:
  OS: "Ubuntu 22.04 / Windows 10 이상"
  Python: "3.10+"
  CUDA: "12.0+ (옵션, GPU 모델 실험 시)"
  Key Libraries:
    - "langchain>=0.2.5"
    - "openai>=1.30.0"
    - "faiss-cpu>=1.8.0"
    - "chromadb>=0.5.0"
    - "pandas>=2.0.0"
    - "numpy>=1.25.0"
    - "fastapi>=0.110.0"
```

### 프로젝트 구조

```
```


## 📅 개발 일정

### 전체 타임라인


##  📊 성과 및 결과