@echo off
chcp 65001 >nul
echo ========================================
echo Jupyter Notebook 실행
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
echo [✓] 가상환경 활성화 중...
call venv\Scripts\activate.bat

:: Jupyter 실행
echo [✓] Jupyter Notebook 실행 중...
echo.
echo 브라우저가 자동으로 열립니다.
echo 종료하려면 Ctrl+C를 누르세요.
echo.
jupyter notebook

pause
