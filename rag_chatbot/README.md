# RAG 시스템 (Retrieval-Augmented Generation)

대학 학사 일정 및 정보를 검색하고 답변하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

- **PDF/CSV 문서 파싱**: PDF 파일과 CSV 파일을 자동으로 파싱하여 벡터 데이터베이스에 저장
- **하이브리드 검색**: BM25와 벡터 검색을 결합한 앙상블 검색
- **Reranking**: Cross-Encoder를 사용한 검색 결과 재정렬
- **LLM 기반 답변 생성**: Qwen3-4B 모델을 사용한 자연어 답변 생성
- **대화형 인터페이스**: 터미널에서 직접 질문하고 답변받기

## 프로젝트 구조

```
.
├── rag_system/          # RAG 시스템 핵심 모듈
│   ├── config.py       # 설정 관리
│   ├── data_parser.py  # PDF/CSV 파싱
│   ├── csv_loader.py   # CSV 로더
│   ├── text_splitter.py # 텍스트 분할
│   ├── vector_store.py # 벡터 스토어 (ChromaDB)
│   ├── retriever.py    # 검색기 (BM25, Vector, Ensemble)
│   ├── llm_loader.py    # LLM 로더
│   ├── rag_pipeline.py # RAG 파이프라인 통합
│   └── ingest.py        # 데이터 적재 스크립트
├── data/               # 데이터 파일 (PDF, CSV)
├── output/             # 출력 파일 (파싱된 CSV, 벡터DB)
├── models/             # 다운로드된 모델 캐시
├── chat_rag.py         # 대화형 인터페이스
├── test_rag_query.py   # 검색 테스트 스크립트
└── requirements.txt    # 의존성 패키지

```

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

**Windows 사용자 주의**: ChromaDB 설치 시 C++ 빌드 도구가 필요할 수 있습니다. 
ChromaDB 없이도 BM25 검색만으로 시스템을 사용할 수 있습니다.

Windows에서 ChromaDB 설치:
```bash
pip install chromadb==0.4.15 --no-build-isolation
```

### 3. 디렉토리 구조 생성

```bash
python -c "from rag_system.config import Config; Config.create_directories()"
```

또는 수동으로 다음 디렉토리를 생성:
- `data/`
- `output/parsed/`
- `output/vectordb/`
- `models/`

## 사용 방법

### 1. 데이터 적재

PDF 파일이나 CSV 파일을 `data/` 폴더에 넣고 다음 명령어 실행:

```bash
python -m rag_system.ingest
```

### 2. 대화형 인터페이스 사용

```bash
python chat_rag.py
```

LLM 사용 여부를 선택하고 질문을 입력하면 됩니다.

### 3. 프로그래밍 방식 사용

```python
from rag_system.rag_pipeline import RAGPipeline

# RAG 파이프라인 초기화
rag = RAGPipeline(
    collection_name="security_info",
    use_reranker=True,
    load_llm=True  # LLM 사용 여부
)

# 질문하기
result = rag.query("2025년 1학기 수강신청 기간은?", use_llm=True)
print(result["answer"])
```

## 주요 모듈

### Config (`rag_system/config.py`)
- 모든 설정을 중앙에서 관리
- 디렉토리 경로, 모델 설정, 검색 파라미터 등

### Data Parser (`rag_system/data_parser.py`)
- PDF 파일 파싱 (PyMuPDF 사용)
- CSV 파일 검증 및 로드

### Vector Store (`rag_system/vector_store.py`)
- ChromaDB를 사용한 벡터 저장소
- 임베딩 모델: Snowflake/snowflake-arctic-embed-l-v2.0

### Retriever (`rag_system/retriever.py`)
- BM25 검색
- 벡터 검색
- 앙상블 검색 (BM25 + Vector)
- Reranking (Cross-Encoder)

### LLM Loader (`rag_system/llm_loader.py`)
- Qwen3-4B 모델 로드
- 4-bit 양자화 지원
- 답변 생성

## 설정

`rag_system/config.py`에서 다음 설정을 변경할 수 있습니다:

- **임베딩 모델**: `EMBEDDING_MODEL_NAME`
- **LLM 모델**: `LLM_MODEL_NAME`
- **텍스트 청크 크기**: `CHUNK_SIZE`, `CHUNK_OVERLAP`
- **검색 개수**: `VECTOR_SEARCH_K`, `BM25_SEARCH_K`
- **Reranker 모델**: `RERANKER_MODEL_NAME`

## 요구사항

- Python 3.8+
- CUDA 지원 GPU (선택사항, CPU도 가능)
- 최소 8GB RAM (LLM 사용 시)

## 라이선스

이 프로젝트는 교육 목적으로 개발되었습니다.

## 문제 해결

### ChromaDB 설치 오류 (Windows)
ChromaDB가 설치되지 않아도 BM25 검색만으로 시스템을 사용할 수 있습니다.

### LangChain 버전 충돌
`requirements.txt`에 명시된 버전을 사용하세요. 문제가 있으면 가상환경을 사용하는 것을 권장합니다.

### 모델 다운로드 실패
인터넷 연결을 확인하고, Hugging Face 토큰이 필요한 경우 설정하세요.

## 기여

이슈나 개선 사항이 있으면 이슈를 등록하거나 Pull Request를 보내주세요.

