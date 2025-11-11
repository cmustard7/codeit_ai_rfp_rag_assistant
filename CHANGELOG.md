# 📦 패키지 버전 변경 내역

## 2025-11-11 - 최신 버전 업데이트 + 모델 설정 수정

### 🔧 모델 설정 변경
- **LLM 모델**: `gpt-5-mini` → `gpt-4o-mini`
  - GPT-5 모델은 아직 공개되지 않음 (2025.11 기준)
  - GPT-4o-mini는 temperature 설정 지원
  - 비용 효율적이며 성능 우수

### 🔄 주요 패키지 업데이트

#### Core RAG Packages

| 패키지 | 이전 버전 | 최신 버전 | 변경 사항 |
|--------|----------|----------|-----------|
| **chromadb** | 0.4.22 | **1.3.4** | 메이저 업데이트 (2025.11.05) |
| **langchain** | 0.1.20 | **0.3.27** | 마이너 업데이트 |
| **langchain-openai** | 0.1.0 | **1.0.2** | 메이저 업데이트 (2025.11.03) |
| **langchain-community** | 0.1.0 | **0.4.1** | 마이너 업데이트 (2025.10.27) |

#### Document Processing

| 패키지 | 이전 버전 | 최신 버전 | 변경 사항 |
|--------|----------|----------|-----------|
| **pypdf** | 4.0.0 | **6.2.0** | 메이저 업데이트 (2025.11.09) |
| **pdfplumber** | 0.11.0 | 0.11.0 | 변경 없음 |
| **python-docx** | 1.1.0 | 1.1.0 | 변경 없음 |

#### Utilities

| 패키지 | 이전 버전 | 최신 버전 | 변경 사항 |
|--------|----------|----------|-----------|
| **pandas** | 2.2.0 | **2.3.0** | 마이너 업데이트 |
| **numpy** | 1.26.0 | **2.1.0** | 메이저 업데이트 |

---

## 🚨 주요 변경 사항 및 주의 사항

### ChromaDB 1.3.4
- **메이저 업데이트**: 0.4.x → 1.3.x
- 안정화된 1.x 버전
- API 변경 가능성 있음 (마이그레이션 가이드 참조)
- 성능 개선 및 버그 수정

### LangChain 패키지
- **langchain-openai 1.0.2**: OpenAI SDK 통합 개선
- **langchain 0.3.27**: 최신 기능 및 버그 수정
- **langchain-community 0.4.1**: 서드파티 통합 업데이트

### pypdf 6.2.0
- **메이저 업데이트**: 4.0 → 6.2
- PDF 처리 성능 향상
- 더 많은 PDF 포맷 지원
- 버그 수정

### NumPy 2.1.0
- **메이저 업데이트**: 1.26 → 2.1
- API 변경 사항 있음
- 성능 개선
- 일부 deprecated 기능 제거

### Pandas 2.3.0
- 마이너 업데이트
- 새로운 기능 추가
- 버그 수정

---

## 💡 업데이트 방법

### 방법 1: 자동 업데이트 (권장)
```cmd
update_packages.bat
```

### 방법 2: 수동 업데이트
```bash
# 가상환경 활성화
.\venv\Scripts\activate

# 패키지 업데이트
pip install --upgrade -r requirements.txt
```

### 방법 3: 클린 설치
```cmd
# 1. 기존 환경 삭제
clean.bat

# 2. 새로 설치
setup.bat
```

---

## ⚠️ 호환성 체크리스트

업데이트 후 다음 사항을 확인하세요:

### ChromaDB 1.x 마이그레이션
- [ ] 기존 `chroma_db/` 디렉토리 백업
- [ ] Vector Store 재생성 필요 여부 확인
- [ ] 검색 쿼리 테스트

### NumPy 2.x 변경사항
- [ ] 코드에서 deprecated 함수 사용 여부 확인
- [ ] 배열 연산 결과 검증

### LangChain 업데이트
- [ ] 프롬프트 템플릿 작동 확인
- [ ] LLM 호출 테스트
- [ ] 체인 구성 검증

---

## 🔗 참고 자료

- [ChromaDB v1.x 릴리즈 노트](https://github.com/chroma-core/chroma/releases)
- [LangChain 변경 내역](https://github.com/langchain-ai/langchain/releases)
- [pypdf 변경 내역](https://github.com/py-pdf/pypdf/releases)
- [NumPy 2.0 마이그레이션 가이드](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [Pandas 변경 내역](https://pandas.pydata.org/docs/whatsnew/index.html)

---

## 📝 테스트 권장사항

업데이트 후 다음 노트북을 순서대로 테스트하세요:

1. `02_document_loading.ipynb` - PDF/HWP 로딩 테스트
2. `03_chunking_test.ipynb` - 청킹 정상 작동 확인
3. `04_build_vectordb.ipynb` - ChromaDB 구축 테스트
4. `05_baseline_rag.ipynb` - 전체 RAG 파이프라인 검증

---

## 🐛 문제 해결

### 업데이트 후 에러 발생 시

**1. 의존성 충돌**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**2. ChromaDB 초기화 오류**
```bash
# ChromaDB 디렉토리 삭제 후 재생성
rmdir /s /q chroma_db
```
그 후 `04_build_vectordb.ipynb` 재실행

**3. NumPy 관련 오류**
```bash
pip uninstall numpy -y
pip install numpy>=2.1.0
```

**4. 전체 재설치**
```cmd
clean.bat
setup.bat
```

---

## ✅ 업데이트 완료 체크리스트

- [ ] `update_packages.bat` 실행 완료
- [ ] 패키지 버전 확인 (`pip list`)
- [ ] 테스트 노트북 정상 실행 확인
- [ ] ChromaDB 재구축 (필요시)
- [ ] 평가 결과 비교 (성능 변화 확인)

---

**마지막 업데이트**: 2025-11-11
**다음 업데이트 예정**: 2025-12 (월간 업데이트 권장)
