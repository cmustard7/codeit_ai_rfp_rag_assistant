<<<<<<< HEAD
# 입찰메이트 RFP RAG 시스템

> 정부·공공 RFP 문서를 빠르게 파악해 컨설턴트가 바로 쓸 수 있는 답변을 주는 사내 RAG.  
> 시나리오 B(클라우드 API)가 `orprem-hf-clean`, 시나리오 A(온프레미스 HF)가 `alpha` 브랜치입니다.

## 브랜치/시나리오
- **시나리오 B (main)**: OpenAI GPT-5 계열 + text-embedding-3-small, Chroma/JSON 벡터스토어, LangGraph 멀티서치, W&B 로깅.
- **시나리오 A (alpha)**: HuggingFace 엔드포인트(Llama/Gemma/Qwen, bge-m3 임베딩) + FAISS/Chroma. 온프레미스 환경 대비용.

| 구분 | LLM/임베딩 | Vector DB | 특징 |
| --- | --- | --- | --- |
| 시나리오 B (main) | GPT-5-mini/nano, text-embedding-3-small | Chroma/JSON | 클라우드 API, 속도/편의 우선 |
| 시나리오 A (alpha) | HF 엔드포인트 (Qwen/Llama/Gemma), bge-m3 | FAISS/Chroma | 온프레미스, 비용/보안 우선 |

## 주요 기능
- HWP/HWXP/PDF(DOCX 포함) 파싱 → 청킹 → 임베딩 → 하이브리드 검색(BM25+벡터, MMR) → 멀티쿼리(paraphrase) → Ko reranker → LangGraph 답변.
- 파싱 파이프라인:
  - 1차: **hwp2hwpx → XML 파싱** (성공 시 최선 품질)
  - 실패 시: **OLE 파서** (기본 HWP 텍스트)
  - 선택: **HWP → PDF 변환 → pdfplumber/pdftohtml + OCR/Camelot** (표/이미지 텍스트 보강)
- 웹 UI: 파일 업로드 후 분석이 끝나면 채팅창 노출, 업로드 파일 교체·초기화 가능(현재 파일 선택 기반).
- 평가: `run_eval` → `evaluate_retrieval` → `evaluate_llm_judge` (W&B/LangSmith 연동 선택).

## 기술 스택
- Python 3.10+ (로컬 기본), LangChain, LangGraph, Chroma/FAISS, W&B, Streamlit(모니터링 UI), sentence-transformers reranker.
- 파서: hwp2hwpx(JAR), pdfplumber, pdftohtml(poppler), PyMuPDF(옵션), PaddleOCR/Tesseract(옵션), Camelot(Table).

## 필수/권장 설치
- **필수 외부 프로그램**: Poppler(pdftohtml), Java 11+ & Maven, hwp2hwpx(JAR 빌드) — 파싱 품질을 위해 반드시 준비해야 합니다.
- **Python** 3.10+ & 가상환경
- (선택) LibreOffice(HWP→PDF), Ghostscript, PaddleOCR/Tesseract, Camelot
=======
# 입찰메이트 RFP RAG 파이프라인

LangGraph 기반으로 HWP/PDF RFP를 파싱 → 청킹 → 임베딩/벡터 저장 → 질의응답/평가를 수행합니다. HWPX 변환, PDF 테이블/Camelot, OCR(PaddleOCR), 멀티쿼리+리랭커까지 옵션으로 포함되어 있으며 웹 데모(업로드→QA)도 제공됩니다.

## 아키텍처 요약
- 입력: HWP/PDF 업로드 → 파싱(HWPX 우선, 실패 시 OLE / PDF는 pdfplumber+pdftohtml+OCR/Camelot)
- 청킹: 헤더/목차 기반, CHUNK_SIZE/OVERLAP(기본 1400/300), 표/이미지 텍스트 포함
- 임베딩/저장: OpenAI 또는 HF 임베딩 → Chroma/JSON/FAISS
- 검색: MMR+BM25 하이브리드, 멀티쿼리 패러프레이즈, 리랭커(score floor)
- 생성: 짧은 프롬프트, 불릿/길이 제한, “모른다” 처리, distillation(소형→대형) 옵션
- 평가: Retrieval 정량지표 + LLM Judge(가짜 질문 시 “모른다” 채점)
- 데모: FastAPI + 프론트(업로드→QA)
>>>>>>> b0458e2b420d0849fb337a0ba64f5d4fe65a0140

#### 협업 일지 링크
- [김민혁 협업일지 (Project Manager)](https://www.notion.so/2a7b412cdba48007871de7b7ad623783)
- [김남중 협업일지 (데이터 처리 담당)]()
- [이현석 협업일지 (Retrieval 담당)]()
- [이재영 협업일지 (Generation 담당)]()

<<<<<<< HEAD

## 실행 순서
```bash
# 0) 가상환경, 필수 패키지
pip install -r requirements.txt

# 1) 벡터스토어 생성 (청킹→임베딩)
python -m src.build_vectorstore

# 2) 평가 파이프라인 실행 (W&B 선택)
python -m src.run_eval --wandb-project my-rag --wandb-run <name>

# 3) Retrieval/LLM 평가
python -m src.evaluate_retrieval --results data/results.json --gold data/gold_targets.json --output data/eval/retrieval_scores.json
python -m src.evaluate_llm_judge --results data/results.json --output data/eval/judge_scores.json --model gpt-5-nano
```
- 웹 데모: 파일 업로드 → 파싱 완료 후 챗봇 노출 (현재 파일 선택 방식)
- 질문/골드 갱신 시 `build_vectorstore`를 다시 실행

## .env 예시
시나리오 B (클라우드/OpenAI)
```
OPENAI_API_KEY=...
LLM_PROVIDER=openai
EMBED_MODEL=text-embedding-3-small
ENABLE_RERANK=1
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_TOP_N=12
RERANK_SCORE_FLOOR=0.2
ENABLE_PARAPHRASE=1
PARAPHRASE_N=2
ENABLE_MMR=1
ENABLE_BM25=1
CHUNK_SIZE=1400
CHUNK_OVERLAP=300
ENABLE_HWPX=1
HWP2HWPX_BIN=F:/hwp2hwpx/hwp2hwpx.cmd
ENABLE_HWP_PDF=0
HWP2PDF_BIN=F:/hwp2pdf.cmd
ENABLE_PDF_HTML=1
PDFTOHTML_BIN=F:/poppler-25.11.0/Library/bin/pdftohtml.exe
ENABLE_OCR=1
ENABLE_CAMELOT=1
ENABLE_PYMUPDF=0
ENABLE_LANGSMITH=0
ENABLE_WANDB=1
```

시나리오 A (온프레미스/HF 예시)
```
LLM_PROVIDER=hf
HF_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
HF_ENDPOINT_URL=http://127.0.0.1:8080
HF_API_TOKEN=...
EMBED_MODEL=bge-m3
VECTOR_DB=faiss        # or chroma/json
ENABLE_RERANK=1
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_TOP_N=12
RERANK_SCORE_FLOOR=0.2
ENABLE_PARAPHRASE=1
PARAPHRASE_N=2
ENABLE_MMR=1
ENABLE_BM25=1
CHUNK_SIZE=1400
CHUNK_OVERLAP=300
ENABLE_HWPX=1
HWP2HWPX_BIN=/home/<user>/hwp2hwpx/hwp2hwpx.sh
ENABLE_HWP_PDF=1
HWP2PDF_BIN=/home/<user>/hwp2pdf.sh
ENABLE_PDF_HTML=1
PDFTOHTML_BIN=/usr/bin/pdftohtml
ENABLE_OCR=1
ENABLE_CAMELOT=1
ENABLE_PYMUPDF=0
=======
## 폴더 구성
```
.
├── app.py / index.html        # 업로드→QA 웹 데모 (FastAPI)
├── src/
│   ├── document_parser.py     # HWPX/힙(hwp2hwpx), PDF→HTML/OCR, Camelot, OLE 폴백
│   ├── text_chunker.py        # 헤더/목차 기반 청킹, CHUNK_SIZE/OVERLAP
│   ├── vector_store.py        # 임베딩/검색, MMR, BM25 하이브리드
│   ├── nodes/answer.py        # 프롬프트/답변, distillation 옵션
│   ├── nodes/retrieve.py      # 멀티쿼리, 리랭커, score floor
│   ├── run_eval.py            # 질문 세트 일괄 실행
│   ├── evaluate_retrieval.py  # 정량 리트리버 평가
│   ├── evaluate_llm_judge.py  # LLM Judge 평가(가짜 질문 “모른다” 처리 포함)
│   └── generate_questions.py  # data_list 기반 질문/가짜 질문 생성
├── data/                      # data_list, files, questions/results/gold/eval
└── scripts/                   # 파서 통계/누락 체크 등 유틸
```

## 설치
```bash
python -m venv .venv
.venv/Scripts/activate  # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
>>>>>>> b0458e2b420d0849fb337a0ba64f5d4fe65a0140
```
외부 툴(필요 시):
- Poppler(pdftohtml/pdf2image) : `choco install poppler` 또는 `apt install poppler-utils`
- Ghostscript(Camelot 테이블 추출) : `choco install ghostscript` 또는 `apt install ghostscript`
- Java + hwp2hwpx 변환기 : hwp2hwpx.cmd/JAR 빌드 후 .env에 경로 설정
- (선택) HWP→PDF 변환기 : LibreOffice/한컴 등 설치 후 .env에 HWP2PDF_BIN 지정
  - Java/Maven이 필요합니다. (Windows: `choco install openjdk maven`, Linux: `sudo apt install openjdk-17-jdk maven`)
  - hwp2hwpx 소스의 `pom.xml`에서 컴파일 버전을 11로 맞춥니다:
    ```
    <maven.compiler.source>11</maven.compiler.source>
    <maven.compiler.target>11</maven.compiler.target>
    ```
  - 테스트는 건너뛰고 빌드:
    `mvn -DskipTests=true clean package`
  - 예시 cmd (Windows, hwp2hwpx 루트에 배치):
    ```cmd
    @echo off
    set BASE=%~dp0
    set JAR=%BASE%target\hwp2hwpx-1.0.0.jar
    set DEP=%BASE%target\dependency
    java -cp "%JAR%;%DEP%\*;%DEP%" Hwp2HwpxCli %*
    ```
    (.sh는 `java -cp "$JAR:$DEP/*" Hwp2HwpxCli "$@"` 형태)
  - 빌드된 JAR을 hwp2hwpx.cmd/.sh로 실행하고, .env에 HWP2HWPX_BIN 경로 지정

<<<<<<< HEAD
## 외부 도구 설치
- **Poppler(pdftohtml)**:  
  - Windows: zip 설치 후 `PDFTOHTML_BIN=C:/.../poppler.../bin/pdftohtml.exe`  
  - Linux/GCP: `sudo apt-get install poppler-utils`
- **Ghostscript**: PDF 후처리/이미지 추출용 (선택)  
  - `sudo apt-get install ghostscript`
- **Java 11+ & Maven**: hwp2hwpx 빌드에 필요  
  - Windows: choco install maven (관리자 권한)  
  - Linux: `sudo apt-get install openjdk-17-jdk maven`
- **hwp2hwpx 빌드** (루트 예: `~/hwp2hwpx`)  
  - `pom.xml`에 `<maven.compiler.source>11</maven.compiler.source>`, `<maven.compiler.target>11</maven.compiler.target>`  
  - `mvn -DskipTests=true clean package`  
  - 실행 스크립트 예(Windows `hwp2hwpx.cmd`):
    ```
    @echo off
    set BASE=%~dp0
    set JAR=%BASE%target\hwp2hwpx-1.0.0.jar
    set DEP=%BASE%target\dependency
    java -cp "%JAR%;%DEP%\hwplib-1.1.10.jar;%DEP%\hwpxlib-1.0.8.jar;%DEP%\*" Hwp2HwpxCli %*
    ```
  - Linux `hwp2hwpx.sh`:
    ```bash
    #!/usr/bin/env bash
    BASE=$(cd "$(dirname "$0")" && pwd)
    JAR="$BASE/target/hwp2hwpx-1.0.0.jar"
    DEP="$BASE/target/dependency"
    java -cp "$JAR:$DEP/hwplib-1.1.10.jar:$DEP/hwpxlib-1.0.8.jar:$DEP/*" Hwp2HwpxCli "$@"
    ```
- **HWP→PDF (LibreOffice 대안)**:
  - Windows: choco install libreoffice-fresh, 스크립트에서 `soffice.exe --headless --convert-to pdf ...`
  - Linux: `sudo apt-get install libreoffice`
  - `.env`에서 `ENABLE_HWP_PDF=1`, `HWP2PDF_BIN=/full/path/hwp2pdf.sh`

## 사용 흐름
1) `.env` 작성 → 외부 툴 경로 설정.  
2) `python -m src.build_vectorstore`로 청킹/임베딩/벡터스토어 생성.  
3) `python -m src.run_eval --wandb-project my-rag --wandb-run <name>`로 멀티서치 LangGraph 실행 및 W&B 로깅.  
4) `evaluate_retrieval`, `evaluate_llm_judge`로 정량 평가.  
5) 웹 UI(앱)에서 파일 업로드 → 파싱 완료 후 챗봇 노출 → 질의응답.

## 팀
| 역할 | 이름 | 주요 담당 |
| --- | --- | --- |
| PM / 총괄 | 김민혁 | 일정/브랜치 운영, 통합 |
| 데이터/파서 | 김남중 | HWP/HWXP/PDF 파싱, 외부 도구 구축 |
| 리트리버 | 이현석 | 멀티서치, BM25+벡터, rerank, distillation 시도 |
| 제너레이션 | 이재영 | 프롬프트 최적화, 질문 세트, 평가 |

## 아키텍처 개요
```mermaid
flowchart LR
  A[업로드 HWP/PDF] --> B{파서}
  B -->|hwp2hwpx| C[텍스트/메타]
  B -->|PDF+OCR/Camelot| C
  C --> D[청킹/임베딩]
  D --> E[BM25+벡터 멀티서치]
  E --> F[Ko reranker]
  F --> G[LangGraph 답변/비교]
  G --> H[평가/로그(W&B, LangSmith)]
```

## 참고
- 파싱 실패 시 OLE 파서로 폴백, HWP→PDF 경로는 옵션이며 품질보다 회복용.
- reranker는 BAAI/bge-reranker-v2-m3(한국어 우수) 기본, 자원 부족 시 base 모델로 교체 가능.
- 질문 세트/골드(`data/questions.json`, `data/gold_targets.json`) 갱신 후 `build_vectorstore`를 다시 실행해야 반영됩니다.
=======
## 환경 변수(.env 예시)
```
OPENAI_API_KEY=sk-...
ENABLE_HWPX=1
HWP2HWPX_BIN=설치경로/hwp2hwpx.cmd
ENABLE_HWP_PDF=0
HWP2PDF_BIN=설치경로/hwp2pdf.cmd
ENABLE_PDF_HTML=1
PDFTOHTML_BIN=(설치경로)poppler-25.11.0/Library/bin/pdftohtml.exe
ENABLE_PYMUPDF=0
ENABLE_OCR=1
ENABLE_CAMELOT=1
ENABLE_RERANK=1
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_TOP_N=12
RERANK_SCORE_FLOOR=0.2
ENABLE_PARAPHRASE=1
PARAPHRASE_N=2
CHUNK_SIZE=1400
CHUNK_OVERLAP=300
ENABLE_MMR=1
ENABLE_BM25=1
# LLM_PROVIDER=hf
# HF_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
# HF_ENDPOINT_URL=http://127.0.0.1:8080   # TGI 등에서 열린 엔드포인트
# HUGGINGFACEHUB_API_TOKEN=hf_...     # 토큰 값
# HF_LLM_TASK=text-generation  
```
※ 온프레미스(HF)로 전환 시 EMBED_PROVIDER/LLM_PROVIDER=hf, HF_* 모델명을 지정.

## 실행 흐름
1) 질문 생성  
`python -m src.generate_questions --follow-up --fake-rate 0.2 --compare-rate 0.1`
2) 벡터스토어 구축  
`python -m src.build_vectorstore`
3) 일괄 평가 실행  
`python -m src.run_eval --wandb-project my-rag --wandb-run <name>`
4) 리트리버 정량 평가  
`python -m src.evaluate_retrieval --results data/results.json --gold data/gold_targets.json`
5) LLM Judge  
`python -m src.evaluate_llm_judge --results data/results.json --model gpt-5-nano`
6) 웹 데모 (업로드→QA)  
`uvicorn app:app --reload --port 8000` 후 http://localhost:8000 접속  
   - 파일 업로드가 끝나면 채팅창이 노출됩니다. (Drag&Drop/파일 선택)

## 주요 기능/옵션
- 파싱: HWP→HWPX(우선), 실패 시 OLE 폴백. PDF는 pdfplumber+pdftohtml, OCR(PaddleOCR), Camelot 테이블 추출. PyMuPDF는 옵션(기본 OFF).
- 검색: MMR, BM25 하이브리드, 멀티쿼리 패러프레이즈, 리랭커(score floor 포함).
- 생성: 짧은 시스템 프롬프트, 길이/불릿 제한, “모른다” 응답 가이드, distillation(소형→대형 LLM) 옵션.
- 평가: gold_targets 기반 Precision/Recall/F1, Judge는 가짜 질문 시 “모른다”를 긍정 채점.
- 실험 로깅: wandb/LangSmith로 토큰/지연/리트리버 설정 기록.

## 시나리오 A/B 비교
- 시나리오 A(온프레미스/HF): HF 임베딩/LLM, Chroma/FAISS 로컬 저장. 비용↓, 설치/자원↑.
- 시나리오 B(클라우드/OpenAI): OpenAI 임베딩/LLM, Chroma/JSON. 세팅 간단, 비용/토큰 관리 필요.

## 팀 역할
- 파이프라인/아키텍처: LangGraph DAG, 프롬프트/QA 흐름 정의
- 파싱/데이터: HWPX/표/OCR 개선, 파일명 정규화, 청킹 전략
- 검색/리랭크: 멀티쿼리, BM25 하이브리드, 리랭커/스코어 튜닝, top-k 실험
- 모델/생성: LLM 선택(HF/클라우드), 프롬프트 압축, distillation 플로우
- 평가/로그: gold_targets 관리, Judge/리트리버 지표, wandb/LangSmith 보드 작성
- 프론트/데모: 업로드→QA 웹 UI, 에러 핸들링, 배포 스크립트

### 팀 진행 노트
- 김민혁(팀장): 전체 일정/의사결정 관리, 파서·웹 데모 품질 확인
- 김남중: HWP 파서( hwp2hwpx + hwpx-owpml-model 검토), pdfplumber/pdftohtml, 이미지 OCR(Tesseract 계획) 실험
- 이현석: 시맨틱+메타데이터 멀티서치, distillation/프롬프트 최적화, rerank 점검, wandb 실험 구조화
- 이재영: Generation 프롬프트/토큰 최적화, 질문 템플릿 고도화

## 개선 아이디어
- 표/이미지 특화 파서 추가, 손상 PDF 대비 파서 우선순위 자동 조정
- 리랭커/멀티쿼리 파라미터 자동 탐색, 후보 문서 수 동적 조절
- “모른다” 검출 스코어 임계값 학습, 히스토리 요약 품질 개선
- CI 파이프라인에 소규모 샘플 평가 포함하여 회귀 방지

## 유지보수 메모
- HWPX 변환 실패 시 OLE로 폴백하므로, 변환기 경로가 올바른지 수동 테스트로 확인하세요.  
- PDF 손상 시 PyMuPDF를 끄고(pdfplumber+pdftohtml/OCR만 사용) 에러를 줄일 수 있습니다.  
- 외부 툴 미설치 시 Camelot/OCR 기능이 자동으로 스킵될 수 있습니다.
>>>>>>> b0458e2b420d0849fb337a0ba64f5d4fe65a0140
