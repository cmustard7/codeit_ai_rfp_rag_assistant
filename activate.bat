@echo off
chcp 65001 >nul

:: 가상환경 확인
if not exist "venv\Scripts\activate.bat" (
    echo [✗] 가상환경이 없습니다!
    echo     먼저 setup.bat을 실행해주세요.
    pause
    exit /b 1
)

:: 가상환경 활성화 (새 cmd 창에서)
echo ========================================
echo RAG 베이스라인 - 가상환경 활성화
echo ========================================
echo.
echo 가상환경이 활성화되었습니다.
echo 종료하려면 'exit' 또는 창을 닫으세요.
echo.

cmd /k "venv\Scripts\activate.bat"
