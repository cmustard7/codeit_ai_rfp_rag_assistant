# 입찰메이트 RFP RAG 시스템

> 정부·공공 RFP 문서를 빠르게 파악해 컨설턴트가 바로 쓸 수 있는 답변을 주는 사내 RAG.  
> 시나리오 B(클라우드 API)가 `main`, 시나리오 A(온프레미스 HF)가 `alpha` 브랜치입니다.

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
```

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
