# RAG 베이스라인 시작 가이드

## 🚀 빠른 시작

### 1. 가상환경 설정 및 패키지 설치

```bash
# 프로젝트 디렉토리로 이동
cd "C:\Users\skawn\Development\중급 프로젝트\rag-baseline"

# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
.\venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 API 키를 입력하세요:

```bash
# .env 파일 생성
copy .env.example .env

# .env 파일을 열어서 API 키 입력
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. 데이터 준비

**옵션 A: Google Drive에서 다운로드**
- `notebooks/00_setup_gdrive.ipynb` 실행
- Google Drive API 인증 후 자동 다운로드

**옵션 B: 수동 다운로드**
- 구글 드라이브에서 RFP 문서와 `data_list.csv` 다운로드
- `data/raw/` 디렉토리에 배치

### 4. Jupyter Notebook 실행

```bash
# Jupyter Notebook 실행
jupyter notebook
```

### 5. 노트북 순서대로 실행

1. **00_setup_gdrive.ipynb** (선택) - Google Drive 연동
2. **01_data_exploration.ipynb** - 데이터 탐색
3. **02_document_loading.ipynb** - 문서 로딩 테스트
4. **03_chunking_test.ipynb** - 청킹 전략 실험
5. **04_build_vectordb.ipynb** - ChromaDB 구축 ⚠️ API 비용 발생
6. **05_baseline_rag.ipynb** - RAG 파이프라인 실행

---

## 📋 체크리스트

### 사전 준비
- [ ] Python 3.8+ 설치 확인
- [ ] OpenAI API Key 발급
- [ ] RFP 문서 100개 준비
- [ ] data_list.csv 파일 준비

### 설치 단계
- [ ] 가상환경 생성 및 활성화
- [ ] 패키지 설치 (`requirements.txt`)
- [ ] `.env` 파일 생성 및 API 키 입력
- [ ] 데이터 파일을 `data/raw/`에 배치

### 실행 단계
- [ ] 데이터 탐색 (노트북 01)
- [ ] 문서 로딩 테스트 (노트북 02)
- [ ] 청킹 전략 실험 (노트북 03)
- [ ] ChromaDB 구축 (노트북 04)
- [ ] RAG 파이프라인 실행 (노트북 05)

---

## ⚠️ 주의사항

### API 비용
- **노트북 04**: ChromaDB 구축 시 OpenAI Embedding API 호출 (100개 문서 기준 약 $0.50-1.00)
- **노트북 05**: RAG 질의응답 시 LLM API 호출 (질문당 약 $0.01-0.05)

### HWP 파일 처리
- olefile 방식으로 추출이 안되는 경우:
  - LibreOffice로 PDF 변환 후 사용
  - 또는 PDF 파일만 먼저 진행

### 디스크 공간
- RFP 문서: 약 100-500MB
- ChromaDB: 약 50-200MB
- 총 필요 공간: 약 500MB-1GB

---

## 🔧 문제 해결

### 패키지 설치 오류
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 개별 패키지 설치 시도
pip install chromadb
pip install langchain langchain-openai langchain-community
```

### ChromaDB 오류
```bash
# ChromaDB 재설치
pip uninstall chromadb -y
pip install chromadb==0.4.22
```

### HWP 파일 읽기 오류
```bash
# olefile 재설치
pip install olefile --upgrade
```

### API Key 오류
- `.env` 파일 위치 확인: 프로젝트 루트 디렉토리
- API Key 형식 확인: `OPENAI_API_KEY=sk-...`
- 따옴표 없이 입력

---

## 📊 베이스라인 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|-----|
| Chunk Size | 1000 | 청크 크기 (글자 수) |
| Chunk Overlap | 200 | 청크 간 중첩 크기 |
| Embedding Model | text-embedding-3-small | OpenAI 임베딩 모델 |
| LLM Model | gpt-4o-mini | OpenAI 언어 모델 |
| Temperature | 0.3 | 답변 일관성 제어 |
| Top-K | 5 | 검색할 문서 수 |
| Max Tokens | 500 | 최대 출력 토큰 |

---

## 📁 프로젝트 구조

```
rag-baseline/
├── data/
│   ├── raw/                    # RFP 원본 문서 (git 제외)
│   └── processed/              # 평가 결과 등
├── chroma_db/                  # ChromaDB 저장소 (git 제외)
├── notebooks/
│   ├── 00_setup_gdrive.ipynb   # Google Drive 연동
│   ├── 01_data_exploration.ipynb
│   ├── 02_document_loading.ipynb
│   ├── 03_chunking_test.ipynb
│   ├── 04_build_vectordb.ipynb
│   └── 05_baseline_rag.ipynb   # 최종 RAG 파이프라인
├── src/
│   ├── config.py               # 설정 관리
│   ├── document_loader.py      # PDF/HWP 로더
│   ├── chunking.py             # 청킹
│   ├── vectorstore.py          # ChromaDB 관리
│   └── __init__.py
├── .env                        # API 키 (git 제외)
├── .gitignore
├── requirements.txt
├── README.md
└── GETTING_STARTED.md          # 이 파일
```

---

## 🎯 다음 단계

베이스라인 완성 후:

1. **성능 평가**
   - 테스트 질문 세트로 답변 품질 평가
   - 응답 시간 측정
   - 검색 정확도 확인

2. **개선 실험**
   - Chunk Size 조정
   - Top-K 값 변경
   - 다른 임베딩 모델 시도
   - 프롬프트 엔지니어링

3. **고도화**
   - Multi-Query Retrieval
   - Re-Ranking
   - Hybrid Search
   - 메타데이터 필터링 활용

4. **문서화**
   - 실험 결과 정리
   - 보고서 작성
   - 발표 자료 준비

---

## 💡 팁

- **노트북 실행 순서 지키기**: 각 노트북은 이전 노트북의 결과를 사용합니다.
- **작은 데이터로 먼저 테스트**: 10개 문서로 먼저 전체 파이프라인을 테스트해보세요.
- **중간 결과 저장**: 각 노트북에서 `%store` 명령으로 결과를 저장합니다.
- **에러 발생 시**: 에러 메시지를 잘 읽고, 필요한 패키지가 설치되어 있는지 확인하세요.

---

## 📞 문의

- 프로젝트 가이드: 프로젝트 디렉토리의 PDF 파일 참조
- LangChain 문서: https://python.langchain.com/
- ChromaDB 문서: https://docs.trychroma.com/
