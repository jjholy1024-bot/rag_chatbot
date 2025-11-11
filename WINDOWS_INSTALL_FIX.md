# Windows 설치 오류 해결 방법

## 문제
`chroma-hnswlib` 빌드 오류: Microsoft Visual C++ 14.0 이상 필요

## 해결 방법

### 방법 1: Visual C++ Build Tools 설치 (권장)

1. [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 다운로드
2. 설치 시 "C++ build tools" 워크로드 선택
3. 설치 완료 후 다시 시도:
   ```bash
   pip install -r requirements.txt
   pip install chromadb==0.4.15 --no-build-isolation
   ```

### 방법 2: 미리 빌드된 wheel 사용 (Python 3.11 이하)

Python 3.13은 아직 미리 빌드된 wheel이 없을 수 있습니다.

Python 3.11로 가상환경을 만들어서 사용:
```bash
python -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
pip install chromadb==0.4.15 --no-build-isolation
```

### 방법 3: chromadb 없이 BM25만 사용 (임시)

벡터 검색 없이 BM25 검색만 사용하도록 코드 수정 필요.
성능이 떨어질 수 있습니다.

## 현재 상태

- `requirements.txt`에서 `chromadb`가 주석 처리되어 있습니다
- `install_windows_simple.bat`를 사용하면 chromadb==0.4.22를 설치하려고 하지만, 여전히 빌드가 필요합니다

## 권장 사항

**Visual C++ Build Tools 설치**가 가장 확실한 해결 방법입니다.
설치 후 모든 chromadb 버전이 정상 작동합니다.

