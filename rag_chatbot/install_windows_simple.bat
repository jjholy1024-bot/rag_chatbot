@echo off
chcp 65001 >nul
echo ========================================
echo RAG System Windows 설치 스크립트
echo ========================================
echo.

echo [주의] chroma-hnswlib 빌드 문제를 피하기 위해
echo        chromadb 0.4.22 버전을 사용합니다.
echo.

echo [1/3] LangChain 패키지 설치...
pip install langchain>=0.3.0,<0.4.0
pip install langchain-core>=0.3.0,<0.4.0
pip install langchain-community>=0.3.0,<0.4.0
pip install langchain-text-splitters>=0.3.0,<0.4.0
pip install langchain-chroma>=0.1.0
pip install langchain-huggingface>=0.1.0

echo.
echo [2/3] ChromaDB 설치 (Windows 호환 버전)...
echo.
echo [주의] chroma-hnswlib 빌드를 위해 Visual C++ Build Tools가 필요합니다.
echo        설치되지 않은 경우 오류가 발생합니다.
echo        해결 방법: https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo.
pip install chromadb==0.4.15 --no-build-isolation
if errorlevel 1 (
    echo.
    echo [오류] ChromaDB 설치 실패!
    echo Visual C++ Build Tools를 설치한 후 다시 시도하세요.
    echo 또는 WINDOWS_INSTALL_FIX.md 파일을 참고하세요.
    pause
    exit /b 1
)

echo.
echo [3/3] 나머지 패키지 설치...
pip install transformers>=4.41.0,<5.0.0
pip install sentence-transformers>=2.2.0
pip install accelerate>=0.20.0
pip install peft>=0.6.0
pip install bitsandbytes>=0.41.0
pip install PyMuPDF>=1.23.0
pip install pymupdf4llm>=0.0.17
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install openpyxl>=3.1.0
pip install rank-bm25>=0.2.2
pip install tqdm>=4.65.0
pip install pytz>=2023.3
pip install torch>=2.0.0

echo.
echo ========================================
echo 설치 완료!
echo ========================================
echo.
echo 다음 명령을 실행하세요:
echo   python -m rag_system.ingest
echo.
pause

