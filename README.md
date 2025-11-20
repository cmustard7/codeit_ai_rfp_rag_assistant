# RAG 베이스라인 - RFP 문서 분석 시스템

## 프로젝트 개요
100개의 RFP(제안요청서) 문서를 분석하여 질문에 답변하는 RAG(Retrieval-Augmented Generation) 시스템의 베이스라인 구현

## 기술 스택
- **Vector DB**: ChromaDB
- **임베딩**: OpenAI text-embedding-3-small
- **LLM**: GPT-5-mini (OpenAI)
- **프레임워크**: LangChain
- **문서 파싱**: 
  - PDF: `pdfplumber` (이미지 추출) — 필요시 poppler `pdftohtml -xml` 사용 권장
    - PDF layout/XML: poppler `pdftohtml -xml` 권장 (레이아웃/좌표 포함 변환)
  - HWP/HWPX: pyhwpx + hwpx-owpml-model (한컴 공식 오픈소스)
- **OCR**: Tesseract (이미지 텍스트 추출)

## 프로젝트 구조
```
rag-baseline/
├── data/
│   ├── raw/              # 원본 RFP 문서 (git 제외)
│   └── processed/        # 전처리된 데이터
├── chroma_db/            # ChromaDB 저장소 (git 제외)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_document_loading.ipynb
│   ├── 03_chunking_test.ipynb
│   ├── 04_build_vectordb.ipynb
│   ├── 05_baseline_rag.ipynb
│   └── 06_test_integrated_parser.ipynb  # 새로 추가
├── src/
│   ├── __init__.py
│   ├── chunking.py
│   ├── config.py
│   ├── document_loader.py
│   ├── hwp_parser.py          # 새로 추가
│   ├── image_processor.py     # 새로 추가
│   ├── logging_utils.py
│   ├── notebook_utils.py
│   ├── pipeline.py
│   └── vectorstore.py
├── hwpx-owpml-model/          # 한컴 오픈소스 (클론)
├── (removed) opendataloader-pdf/  # 레거시: 저장소에서 제거됨
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 빠른 시작 (Windows)

### 필수 준비물
- Python 3.13.5 (또는 3.11 이상 버전)
- Microsoft Visual C++ Build Tools (Desktop development with C++)
- OpenAI API Key
- **한글 프로그램 (한컴오피스)** - HWP 파일 변환용 (선택)
- **Tesseract OCR** - 이미지 텍스트 추출용 (선택)

추가 상세 설치(Windows 예시):

- Tesseract 설치 (권장: Chocolatey 사용):

```powershell
choco install tesseract -y
# 또는 수동 설치: https://github.com/tesseract-ocr/tesseract/releases
```

### PDF 레이아웃(XML) 추출 (poppler `pdftohtml -xml`)

문서의 레이아웃(글상자, 좌표, 글자 단위 정보)까지 필요하면 `pdftohtml -xml`(poppler 도구)를 권장합니다. 이 도구는 PDF를 레이아웃 정보를 포함한 XML로 변환하므로 테이블·단락·좌표 기반의 고정밀 파싱에 유리합니다. 단점은 시스템 의존성(팝플러 바이너리 설치)이 필요하다는 점입니다.

Windows 설치 예시:

```powershell
# Chocolatey로 설치 (권장 간편 방법)
choco install poppler -y

# 설치 후 pdftohtml 명령 확인
pdftohtml -v
```

수동 설치:

- Windows용 poppler 바이너리를 내려받아 PATH에 추가합니다: https://github.com/oschwartz10612/poppler-windows/releases

기본 사용 예:

```powershell
# XML 출력 파일로 저장
pdftohtml -xml -nodrm -enc UTF-8 input.pdf output.xml

# stdout으로 출력하여 파이프 처리
pdftohtml -xml -nodrm -enc UTF-8 input.pdf -stdout > output.xml
```

설명:
- `-xml`: XML 형식으로 출력합니다(글자/상자/좌표 등 포함).
- `-nodrm`: DRM 관련 오류가 있는 문서를 무시하고 처리합니다.
- `-enc UTF-8`: 출력 인코딩을 UTF-8로 지정합니다.

활용 팁:
- 변환된 XML은 좌표 기반의 후처리(머지, 테이블 추출, 텍스트 블록 재구성)에 적합합니다.
- 변환 후 `IntegratedDocumentLoader`에 통합하면 HWPX 처럼 구조적 파싱을 적용할 수 있습니다(파서 추가 필요).

대체(순수 Python) 옵션:
- `pdfminer.six`: 레이아웃 분석(글상자, 문자 좌표) 기능을 제공하여 XML 유사 구조로 가공할 수 있습니다. (`pip install pdfminer.six`)
- `pdfplumber`: 표·텍스트·좌표 추출에 편리한 고수준 API를 제공합니다. (`pip install pdfplumber`)

팁: 스캔형 PDF는 이미지 기반이므로 OCR(현재 프로젝트의 Tesseract 통합)을 병행해야 합니다.

- `pyhwpx` 설치 (venv 활성화 후):

```powershell
.\venv\Scripts\activate
pip install pyhwpx
```

- Visual C++ Build Tools: hnswlib 등 네이티브 확장 빌드에 필요합니다. 설치 후 터미널을 재시작하세요.


> 📌 **HWP 파일 처리**: HWP 파일을 파싱하려면 한글 프로그램이 시스템에 설치되어 있어야 합니다. `pyhwpx`는 가상환경에 설치되지만, 실제 변환은 시스템의 한글 프로그램을 통해 이루어집니다.

> 📌 **이미지 OCR**: PDF 내 이미지에서 텍스트를 추출하려면 [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)을 설치하세요.

> 📌 Windows에서 `hnswlib`를 빌드하려면 위 도구가 반드시 필요합니다. [Microsoft 공식 다운로드 페이지](https://visualstudio.microsoft.com/visual-cpp-build-tools/)에서 설치 후 진행하세요.

---

## 설치 및 실행 팁 (문제 해결 가이드)

- HWP 변환/파싱 문제

  - 이 프로젝트는 HWP → HWPX 변환 후 XML을 파싱하는 흐름(`src.hwp_parser.HwpParser`)을 사용합니다.
  - `pyhwpx`가 설치되어 있어야 변환 기능이 동작합니다(`pip install pyhwpx`).
  - 변환/파싱은 시스템의 한컴(또는 연동된 변환 도구)에 의존할 수 있으므로, 변환 실패 시 권한/설치 여부를 먼저 확인하세요.

- Tesseract가 동작하지 않을 때

  - 설치 후 `tesseract --version`으로 확인하세요.
  - 언어 데이터를 추가로 설치해야 한글 OCR이 동작합니다(예: `kor` 언어팩).

- `hnswlib` 빌드 에러

  - Visual C++ Build Tools가 필요합니다. 설치 후 `pip install -r requirements.txt`를 다시 실행하세요.

---
## 빠른 실행(옵션)
- 기본 `start.bat` 외에 Jupyter 없이 환경만 준비하려면 `start_no_jupyter.bat`를 사용하세요(venv 생성 + requirements 설치만 수행).

```powershell
start_no_jupyter.bat
```

또는 수동:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 환경 변수
- 로컬에서 빠르게 설정하려면:

```powershell
setx OPENAI_API_KEY "sk-..."
```

CI에서는 리포지토리 시크릿으로 `OPENAI_API_KEY`를 설정하세요.

### 원클릭 실행 ⭐
```cmd
start.bat
```
- ✅ 가상환경 자동 생성 (없는 경우)
- ✅ 패키지 자동 설치
- ✅ Jupyter Notebook 자동 실행
- ⏱️ 첫 실행: 3-5분 / 이후: 즉시 실행

### 종료
```cmd
stop.bat
```
- 실행 중인 Python 및 Jupyter 프로세스 종료

### 수동 설치 (선택사항)
```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
.\venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
copy .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
```

### 📝 유용한 스크립트 도구
- `start.bat` - 프로젝트 시작 (환경 설정 + Jupyter 실행)
- `stop.bat` - 프로세스 종료
- `scripts/build_vectorstore.py` - 문서를 임베딩해 ChromaDB 인덱스 생성
- `scripts/rag_cli.py` - CLI 환경에서 RAG 질의 실행

📖 자세한 사용법: [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)

### 🗂️ HWP 파싱 도구(hwp5txt) 준비
`pyhwp` 패키지가 설치되면 `hwp5txt` CLI가 함께 제공됩니다. HWP 문서를 로딩하기 전에 아래 단계를 통해 정상 동작을 확인하세요.

1. 가상환경을 활성화합니다: `.\venv\Scripts\activate`
2. `hwp5txt --help` 명령을 실행해 CLI가 인식되는지 확인합니다.
3. 명령을 찾을 수 없다면 `pyhwp`가 설치된 가상환경의 `Scripts` 경로를 `PATH`에 추가하거나, PowerShell에서 `Set-ExecutionPolicy RemoteSigned`를 설정해야 할 수 있습니다.

Windows 환경에서는 백신 프로그램이 실행 파일을 차단하는 경우가 있습니다. hwp5txt 실행이 거부되면 예외 목록에 추가하고 다시 시도하세요.

Jupyter Notebook 실행은 `start.bat`을 더블 클릭하거나, 명령 프롬프트에서:

```cmd
start.bat
```

## 사용 방법

Jupyter Notebook을 순서대로 실행:
1. `00_setup_gdrive.ipynb` - Google Drive 연동
2. `01_data_exploration.ipynb` - 데이터 탐색
3. `02_document_loading.ipynb` - 문서 로딩 테스트
4. `03_chunking_test.ipynb` - 청킹 전략 실험
5. `04_build_vectordb.ipynb` - ChromaDB 구축
6. `05_baseline_rag.ipynb` - RAG 파이프라인 실행

또는 CLI에서 바로 질의:

```powershell
python scripts/rag_cli.py "국민연금공단 이러닝시스템 요구사항을 정리해 줘"
```

인터랙티브 모드:

```powershell
python scripts/rag_cli.py
```

### ChromaDB 재구축 방법 비교

| 방법 | 실행 경로 | 주요 단계 | 장점 | 권장 상황 |
|------|-----------|-----------|------|-----------|
| 노트북 (`04_build_vectordb.ipynb`) | Jupyter Notebook | 청크 백업 로드 → 필요 시 API Key 입력 → `ChromaDBManager.create_vectorstore` 호출 | 단계별 출력과 에러 안내가 자세함, 실험/디버깅에 유용 | 파라미터를 조정하거나 청크 데이터를 직접 확인해야 할 때 |
| 스크립트 (`scripts/build_vectorstore.py`) | PowerShell / CMD | 문서 로드 → 청킹 → Vector Store 저장 | 한 줄 명령으로 전체 파이프라인 실행, 자동화/재빌드에 적합 | 빌드를 반복 실행하거나 서버·배치 환경에서 인덱스를 재생성할 때 |

## 베이스라인 하이퍼파라미터
- **Chunk Size**: 1000
- **Chunk Overlap**: 200
- **Top-K**: 5
- **Temperature**: 0.3
- **Max Tokens**: 500

## 주의사항
⚠️ 원본 RFP 문서는 비밀유지계약에 따라 외부 공유 금지
