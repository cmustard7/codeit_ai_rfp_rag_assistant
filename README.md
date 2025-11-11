# RAG 베이스라인 - RFP 문서 분석 시스템

## 프로젝트 개요
100개의 RFP(제안요청서) 문서를 분석하여 질문에 답변하는 RAG(Retrieval-Augmented Generation) 시스템의 베이스라인 구현

## 기술 스택
- **Vector DB**: ChromaDB
- **임베딩**: OpenAI text-embedding-3-small
- **LLM**: GPT-4o-mini (OpenAI)
- **프레임워크**: LangChain

## 프로젝트 구조
```
rag-baseline/
├── data/
│   ├── raw/              # 원본 RFP 문서 (git 제외)
│   └── processed/        # 전처리된 데이터
├── chroma_db/            # ChromaDB 저장소 (git 제외)
├── notebooks/
│   ├── 00_setup_gdrive.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_document_loading.ipynb
│   ├── 03_chunking_test.ipynb
│   ├── 04_build_vectordb.ipynb
│   └── 05_baseline_rag.ipynb
├── src/
│   ├── config.py
│   ├── gdrive_loader.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── vectorstore.py
│   ├── retrieval.py
│   └── generation.py
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 🚀 빠른 설치 (Windows)

### 방법 1: 자동 설치 (권장) ⭐
```cmd
setup.bat
```
- ✅ 모든 설정을 자동으로 완료
- ✅ 가상환경 생성, 패키지 설치, .env 파일 생성
- ⏱️ 소요 시간: 3-5분

### 방법 2: 수동 설치
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

### 📝 유용한 배치 스크립트
- `setup.bat` - 초기 설정 (1회 실행)
- `run_jupyter.bat` - Jupyter Notebook 실행
- `activate.bat` - 가상환경 활성화
- `update_packages.bat` - 패키지 업데이트
- `clean.bat` - 프로젝트 초기화

📖 자세한 사용법: [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)

## 사용 방법

Jupyter Notebook을 순서대로 실행:
1. `00_setup_gdrive.ipynb` - Google Drive 연동
2. `01_data_exploration.ipynb` - 데이터 탐색
3. `02_document_loading.ipynb` - 문서 로딩 테스트
4. `03_chunking_test.ipynb` - 청킹 전략 실험
5. `04_build_vectordb.ipynb` - ChromaDB 구축
6. `05_baseline_rag.ipynb` - RAG 파이프라인 실행

## 베이스라인 하이퍼파라미터
- **Chunk Size**: 1000
- **Chunk Overlap**: 200
- **Top-K**: 5
- **Temperature**: 0.3
- **Max Tokens**: 500

## 주의사항
⚠️ 원본 RFP 문서는 비밀유지계약에 따라 외부 공유 금지
