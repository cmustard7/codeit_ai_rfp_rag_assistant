@echo off
chcp 65001 >nul
echo ========================================
echo 프로젝트 정리
echo ========================================
echo.

echo 다음 항목들을 삭제합니다:
echo   - venv\ (가상환경)
echo   - chroma_db\ (ChromaDB 데이터)
echo   - data\processed\ (처리된 데이터)
echo   - __pycache__\ (Python 캐시)
echo   - .ipynb_checkpoints\ (Jupyter 캐시)
echo.
echo [주의] 이 작업은 되돌릴 수 없습니다!
echo.
set /p confirm=정말 삭제하시겠습니까? (Y/N):

if /i not "%confirm%"=="Y" (
    echo.
    echo 취소되었습니다.
    pause
    exit /b 0
)

echo.
echo 정리 중...

:: 가상환경 삭제
if exist "venv\" (
    echo [1/5] 가상환경 삭제 중...
    rmdir /s /q venv
    echo [✓] 가상환경 삭제 완료
) else (
    echo [1/5] 가상환경 없음 (스킵)
)

:: ChromaDB 삭제
if exist "chroma_db\" (
    echo [2/5] ChromaDB 삭제 중...
    rmdir /s /q chroma_db
    echo [✓] ChromaDB 삭제 완료
) else (
    echo [2/5] ChromaDB 없음 (스킵)
)

:: 처리된 데이터 삭제
if exist "data\processed\" (
    echo [3/5] 처리된 데이터 삭제 중...
    rmdir /s /q data\processed
    mkdir data\processed
    echo [✓] 처리된 데이터 삭제 완료
) else (
    echo [3/5] 처리된 데이터 없음 (스킵)
)

:: Python 캐시 삭제
echo [4/5] Python 캐시 삭제 중...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
echo [✓] Python 캐시 삭제 완료

:: Jupyter 캐시 삭제
echo [5/5] Jupyter 캐시 삭제 중...
for /d /r . %%d in (.ipynb_checkpoints) do @if exist "%%d" rmdir /s /q "%%d"
echo [✓] Jupyter 캐시 삭제 완료

echo.
echo ========================================
echo 정리 완료!
echo ========================================
echo.
echo 다음 단계:
echo   setup.bat을 실행하여 다시 설정하세요.
echo.
pause
