# 🚀 배치 스크립트 사용 가이드

Windows에서 쉽게 프로젝트를 설정하고 실행할 수 있는 배치 스크립트 모음입니다.

---

## 📋 스크립트 목록

### 1. `setup.bat` - 🔧 초기 설정 (필수)
**처음 한 번만 실행**

프로젝트의 전체 환경을 자동으로 설정합니다.

```cmd
setup.bat
```

**수행 작업:**
- ✅ Python 설치 확인
- ✅ 가상환경 생성 (`venv/`)
- ✅ pip 업그레이드
- ✅ 패키지 설치 (`requirements.txt`)
- ✅ `.env` 파일 생성 (선택)
- ✅ 데이터 디렉토리 생성
- ✅ Jupyter Notebook 실행 옵션

**소요 시간:** 약 3-5분

---

### 2. `run_jupyter.bat` - 📓 Jupyter 실행
**노트북 작업 시 사용**

가상환경을 활성화하고 Jupyter Notebook을 실행합니다.

```cmd
run_jupyter.bat
```

**실행 후:**
- 브라우저가 자동으로 열림
- `notebooks/` 디렉토리의 노트북 실행 가능
- 종료: `Ctrl+C`

---

### 3. `activate.bat` - 🔌 가상환경 활성화
**명령어 실행 시 사용**

새 CMD 창에서 가상환경을 활성화합니다.

```cmd
activate.bat
```

**사용 예시:**
```cmd
activate.bat
# 활성화된 환경에서
python src/config.py
pip list
```

---

### 4. `update_packages.bat` - 🔄 패키지 업데이트
**패키지 버전 업데이트 시 사용**

설치된 모든 패키지를 최신 버전으로 업데이트합니다.

```cmd
update_packages.bat
```

**수행 작업:**
- pip 업그레이드
- `requirements.txt`의 모든 패키지 업데이트
- 설치된 버전 표시

**주의:** 업데이트 후 호환성 문제가 생길 수 있습니다.

---

### 5. `clean.bat` - 🧹 프로젝트 정리
**처음부터 다시 시작하고 싶을 때**

가상환경, 캐시, 생성된 데이터를 모두 삭제합니다.

```cmd
clean.bat
```

**삭제 항목:**
- ❌ `venv/` - 가상환경
- ❌ `chroma_db/` - Vector DB
- ❌ `data/processed/` - 처리된 데이터
- ❌ `__pycache__/` - Python 캐시
- ❌ `.ipynb_checkpoints/` - Jupyter 캐시

**⚠️ 경고:** 이 작업은 되돌릴 수 없습니다!

**정리 후:**
```cmd
setup.bat
```
다시 실행하여 환경 재설정

---

## 🎯 전형적인 사용 흐름

### 처음 시작할 때
```cmd
# 1. 초기 설정
setup.bat

# 2. .env 파일에 API Key 입력
notepad .env

# 3. 데이터 파일 배치
# data/raw/ 디렉토리에 RFP 문서 복사

# 4. Jupyter 실행
run_jupyter.bat
```

### 일상적인 작업
```cmd
# Jupyter로 작업
run_jupyter.bat

# 또는 터미널 작업
activate.bat
```

### 문제 해결 시
```cmd
# 1. 모두 삭제
clean.bat

# 2. 재설정
setup.bat
```

---

## 🔧 문제 해결

### 스크립트 실행이 안될 때

**실행 정책 오류:**
```powershell
# PowerShell 관리자 모드에서
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**권한 오류:**
- 마우스 우클릭 → "관리자 권한으로 실행"

### 한글 깨짐
- 스크립트는 UTF-8로 작성되어 있습니다
- CMD 창에서 자동으로 UTF-8로 전환됩니다 (`chcp 65001`)

### Python을 찾을 수 없음
1. Python 설치 확인: `python --version`
2. 환경 변수 PATH에 Python 추가
3. Python 재설치

---

## 📊 각 스크립트 비교

| 스크립트 | 용도 | 실행 빈도 | 소요 시간 |
|---------|------|----------|----------|
| `setup.bat` | 초기 설정 | 1회 | 3-5분 |
| `run_jupyter.bat` | Jupyter 실행 | 매일 | 5초 |
| `activate.bat` | 환경 활성화 | 필요시 | 1초 |
| `update_packages.bat` | 패키지 업데이트 | 월 1회 | 2-3분 |
| `clean.bat` | 환경 초기화 | 필요시 | 10초 |

---

## 💡 팁

### 바탕화면 바로가기 만들기
1. `run_jupyter.bat` 우클릭
2. "바로 가기 만들기"
3. 바탕화면으로 이동

### 빠른 실행
Windows 탐색기에서 주소창에 입력:
```
cmd
```
그 후:
```cmd
run_jupyter.bat
```

### 스크립트 수정
- 메모장으로 열기
- UTF-8 인코딩 유지
- `@echo off` 제거하면 디버깅 가능

---

## 🆘 도움말

### 자주 묻는 질문

**Q: setup.bat을 다시 실행해도 되나요?**
A: 네, 기존 가상환경을 삭제할지 물어봅니다.

**Q: requirements.txt를 수정했어요.**
A: `update_packages.bat` 실행

**Q: 가상환경을 삭제하고 싶어요.**
A: `clean.bat` 실행 또는 `venv` 폴더 직접 삭제

**Q: ChromaDB 데이터만 삭제하고 싶어요.**
A: `chroma_db` 폴더 직접 삭제

**Q: 스크립트가 멈췄어요.**
A: `Ctrl+C` 누르고 재실행

---

## 📞 문의

스크립트 관련 문제가 있으면:
1. 에러 메시지 확인
2. Python 버전 확인 (`python --version`)
3. 관리자 권한으로 재시도

---

**즐거운 코딩 되세요! 🎉**
