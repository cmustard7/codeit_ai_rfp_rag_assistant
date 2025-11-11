@echo off
chcp 65001 >nul
echo ========================================
echo RAG 베이스라인 프로젝트 설정
echo ========================================
echo.

:: 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [✓] 관리자 권한으로 실행 중
) else (
    echo [!] 관리자 권한 없음 (선택사항)
)
echo.

:: Python 설치 확인
echo [1/5] Python 설치 확인 중...
python --version >nul 2>&1
if %errorLevel% == 0 (
    python --version
    echo [✓] Python 설치 확인 완료
) else (
    echo [✗] Python이 설치되어 있지 않습니다!
    echo     https://www.python.org/downloads/ 에서 Python을 설치해주세요.
    pause
    exit /b 1
)
echo.

:: 가상환경 생성
echo [2/5] 가상환경 생성 중...
if exist "venv\" (
    echo [!] 가상환경이 이미 존재합니다. 삭제하고 다시 만드시겠습니까? (Y/N)
    set /p answer=
    if /i "%answer%"=="Y" (
        echo     기존 가상환경 삭제 중...
        rmdir /s /q venv
        python -m venv venv
        echo [✓] 가상환경 재생성 완료
    ) else (
        echo [✓] 기존 가상환경 사용
    )
) else (
    python -m venv venv
    if %errorLevel% == 0 (
        echo [✓] 가상환경 생성 완료
    ) else (
        echo [✗] 가상환경 생성 실패
        pause
        exit /b 1
    )
)
echo.

:: 가상환경 활성화
echo [3/5] 가상환경 활성화 중...
call venv\Scripts\activate.bat
if %errorLevel% == 0 (
    echo [✓] 가상환경 활성화 완료
) else (
    echo [✗] 가상환경 활성화 실패
    pause
    exit /b 1
)
echo.

:: pip 업그레이드
echo [4/5] pip 업그레이드 중...
python -m pip install --upgrade pip
if %errorLevel% == 0 (
    echo [✓] pip 업그레이드 완료
) else (
    echo [!] pip 업그레이드 실패 (계속 진행)
)
echo.

:: 패키지 설치
echo [5/5] 패키지 설치 중... (시간이 걸릴 수 있습니다)
pip install -r requirements.txt
if %errorLevel% == 0 (
    echo [✓] 패키지 설치 완료
) else (
    echo [✗] 패키지 설치 실패
    echo.
    echo 문제 해결:
    echo   1. 인터넷 연결 확인
    echo   2. pip install -r requirements.txt 수동 실행
    pause
    exit /b 1
)
echo.

:: .env 파일 확인
echo ========================================
echo 추가 설정
echo ========================================
if not exist ".env" (
    echo [!] .env 파일이 없습니다.
    echo     .env.example을 복사하여 .env 파일을 생성하시겠습니까? (Y/N)
    set /p create_env=
    if /i "%create_env%"=="Y" (
        copy .env.example .env >nul
        echo [✓] .env 파일 생성 완료
        echo [!] .env 파일을 열어서 OPENAI_API_KEY를 입력해주세요!
        echo.
        echo .env 파일을 지금 여시겠습니까? (Y/N)
        set /p open_env=
        if /i "%open_env%"=="Y" (
            notepad .env
        )
    )
) else (
    echo [✓] .env 파일이 이미 존재합니다.
)
echo.

:: 데이터 디렉토리 확인
if not exist "data\raw\" (
    echo [!] data\raw\ 디렉토리가 없습니다. 생성합니다...
    mkdir data\raw
    echo [✓] data\raw\ 디렉토리 생성 완료
    echo [!] RFP 문서 파일을 data\raw\ 디렉토리에 넣어주세요!
)
echo.

:: 완료
echo ========================================
echo 설정 완료!
echo ========================================
echo.
echo 다음 단계:
echo   1. .env 파일에 OpenAI API Key 입력
echo   2. data\raw\ 디렉토리에 RFP 문서 배치
echo   3. run_jupyter.bat 실행하여 Jupyter Notebook 시작
echo.
echo Jupyter Notebook을 지금 실행하시겠습니까? (Y/N)
set /p run_jupyter=
if /i "%run_jupyter%"=="Y" (
    echo.
    echo Jupyter Notebook 실행 중...
    jupyter notebook
) else (
    echo.
    echo 나중에 run_jupyter.bat을 실행하여 시작하세요.
)
echo.
pause
