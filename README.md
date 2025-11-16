<<<<<<< HEAD
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
| **데이터 처리 담당** | 김남중 | 문서별 메타데이터 처리, 문서 청킹 전략 설계 및 검증 |
| **Retrieval 담당** | 이현석 | 임베딩 생성 및 Vector DB 구축, 메타데이터 필터링 및 Retrieval 기법 고도화 |
| **Generation 담당** | 이재영 | 모델 선정 및 텍스트 생성 옵션 설정, 모델 응답 고도화, 대화 흐름 유지 및 히스토리 반영 전략 설계 |

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
=======
# LangGraph RAG Starter

이 프로젝트는 LangGraph 기반으로 RAG 파이프라인을 빠르게 구성하기 위한 최소 템플릿입니다. 기존 `codeit_ai_rfp_rag_assistant` 프로젝트에서 사용한 라이브러리 버전을 그대로 맞췄으며, LangGraph 전용 워크플로를 `src/` 아래에서 정의합니다.

## 구성
```
LangGraph_rag/
├── README.md
├── requirements.txt
├── data/
│   ├── data_list.csv|xlsx   # 기관/사업 메타데이터
│   ├── files/               # 원본 문서(HWP/PDF/DOCX 등)
│   ├── questions.json       # 평가용 질문(자동 생성)
│   ├── results.json         # run_eval 결과
│   └── vectorstore.json     # 문서 임베딩 캐시
└── src/
    ├── build_vectorstore.py # data/files → vectorstore.json 생성
    ├── data_loader.py       # data_list.* + 문서 본문 로딩
    ├── document_parser.py   # TXT/PDF/DOCX 파서
    ├── generate_questions.py # 메타데이터 기반 질문 생성
    ├── graph_state.py       # LangGraph 상태 정의(TypedDict)
    ├── nodes/               # LangGraph 노드(question/retrieve/answer/update)
    ├── run_chat.py          # 대화형 테스트 CLI
    ├── run_eval.py          # 질문 세트 일괄 평가
    ├── vector_store.py      # 임베딩 생성/검색 헬퍼
    └── workflow.py          # LangGraph DAG 정의
```

## 데이터 준비
- `data/data_list.csv` 또는 `data/data_list.xlsx` : 기관/사업 메타데이터. 열 이름은 자유롭게 사용하되, 아래 키 중 하나와 매칭되면 자동으로 인식됩니다.
  - 기관 : `발주 기관`, `발주기관`, `agency`
  - 사업명 : `사업명`, `project_name`, `title`
  - 요약 : `사업 요약`, `요약`, `summary`
- `data/files/` : 실제 문서가 위치할 디렉터리(향후 문서 로딩/벡터스토어 구성에 사용).

## 빠른 시작
1. 가상환경 생성 및 패키지 설치
   ```bash
   python -m venv .venv
   .venv/Scripts/activate   # Windows
   pip install -r requirements.txt
   ```
2. `.env` 파일에 OpenAI API Key 설정
   ```bash
   echo OPENAI_API_KEY=sk-... > .env
   ```
   (기존 환경 변수 사용도 가능하며, `python-dotenv`가 자동으로 `.env`를 읽습니다.)
3. 사업 메타데이터 기반 질문 생성
   ```bash
   python -m src.generate_questions
   # (--limit, --follow-up 등의 옵션으로 샘플 수/후속 질문 생성 여부 조정 가능)
   ```
4. 벡터스토어 생성 (문서를 LLM이 참조하도록, 최초 1회 또는 데이터 변경 시)
   ```bash
   python -m src.build_vectorstore
   # 결과: data/vectorstore.json
   ```
5. LangGraph 워크플로 실행
   ```bash
   python -m src.run_eval
   # (--questions, --output 옵션으로 경로 커스터마이징 가능)
   ```
6. 대화형 테스트(선택)
   ```bash
   python -m src.run_chat
   # 'exit' 입력 시 종료, --reset 옵션으로 매 질문마다 상태 초기화 가능
   ```

### 주요 CLI 옵션
- `generate_questions`
  - `--limit <int>`: data_list 상단에서 몇 개의 행을 사용할지 지정 (기본 3).
  - `--follow-up`: 각 사업마다 후속 질문 추가 생성 여부(플래그).
  - `--csv/--xlsx/--output <path>`: 입력/출력 경로를 바꾸고 싶을 때 지정.
- `run_eval`
  - `--questions <path>`: 기본 `data/questions.json` 대신 다른 질문 파일 사용.
  - `--output <path>`: 결과 JSON 저장 위치 지정(기본 `data/results.json`).
- `run_chat`
  - `--reset`: 매 질문마다 상태 초기화(기본은 세션 상태 유지).
- `build_vectorstore`
  - `--output <path>`: 생성된 벡터스토어 JSON 위치 지정(기본 `data/vectorstore.json`).

## TODO
- `data/files` 내 HWP 등 특수 포맷 파서 보강
- 자동 평가 지표/스코어 산출 스크립트 추가
- run_chat UI/로그 저장 등 편의 기능 확장
>>>>>>> 462ebe9 (Initial LangGraph baseline)
