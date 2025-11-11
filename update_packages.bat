@echo off
chcp 65001 >nul
echo ========================================
echo 패키지 업데이트
echo ========================================
echo.

:: 가상환경 확인
if not exist "venv\Scripts\activate.bat" (
    echo [✗] 가상환경이 없습니다!
    echo     먼저 setup.bat을 실행해주세요.
    pause
    exit /b 1
)

:: 가상환경 활성화
echo [1/3] 가상환경 활성화 중...
call venv\Scripts\activate.bat
echo [✓] 가상환경 활성화 완료
echo.

:: pip 업그레이드
echo [2/3] pip 업그레이드 중...
python -m pip install --upgrade pip
echo [✓] pip 업그레이드 완료
echo.

:: 패키지 업데이트
echo [3/3] 설치된 패키지 업데이트 중...
pip install --upgrade -r requirements.txt
if %errorLevel% == 0 (
    echo [✓] 패키지 업데이트 완료
) else (
    echo [✗] 일부 패키지 업데이트 실패
)
echo.

:: 설치된 패키지 목록 표시
echo ========================================
echo 설치된 주요 패키지 버전:
echo ========================================
pip show chromadb langchain langchain-openai pandas numpy | findstr "Name: Version:"
echo.

echo 완료!
pause
