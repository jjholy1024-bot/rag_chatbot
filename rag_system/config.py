"""
설정 관리 모듈
환경 변수나 설정 파일을 통해 관리할 수 있도록 구성
"""
import os
from pathlib import Path
from typing import Optional

class Config:
    """RAG 시스템 설정 클래스"""
    
    # 디렉토리 설정
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    PARSED_DIR = OUTPUT_DIR / "parsed"
    VECTORDB_DIR = OUTPUT_DIR / "vectordb"
    MODEL_CACHE_DIR = BASE_DIR / "models"
    
    # PDF 파싱 설정
    PDF_MAX_LEN = 10000
    PDF_MAX_LVL = 2
    
    # 텍스트 분할 설정
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 102
    
    # 임베딩 모델 설정
    EMBEDDING_MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"
    EMBEDDING_DEVICE = "cuda:0" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    EMBEDDING_NORMALIZE = True
    
    # LLM 설정
    LLM_MODEL_NAME = "Qwen/Qwen3-4B"
    LLM_QUANTIZATION = True  # 4-bit 양자화 사용 여부
    LLM_MAX_NEW_TOKENS = 1024
    
    # VectorDB 설정
    COLLECTION_NAME = "security_info"
    
    # Retriever 설정
    VECTOR_SEARCH_K = 10  # 더 많은 후보 검색
    BM25_SEARCH_K = 10  # 더 많은 후보 검색
    ENSEMBLE_WEIGHTS = [0.45, 0.55]  # [BM25, Vector]
    RERANKER_TOP_N = 5  # Reranker 후 더 많은 결과 선택
    RERANKER_MODEL_NAME = "dragonkue/bge-reranker-v2-m3-ko"
    
    # 검색 품질 개선
    MIN_DOC_LENGTH = 40  # 최소 문서 길이
    TOP_K_DEFAULT = 5  # 기본 검색 개수
    
    # 시드 설정
    RANDOM_SEED = 7
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 생성"""
        directories = [
            cls.DATA_DIR,
            cls.OUTPUT_DIR,
            cls.PARSED_DIR,
            cls.VECTORDB_DIR,
            cls.MODEL_CACHE_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_embedding_model_kwargs(cls):
        """임베딩 모델 kwargs 반환"""
        import torch
        return {
            "device": cls.EMBEDDING_DEVICE,
            "model_kwargs": {
                "dtype": torch.float16 if torch.cuda.is_available() else torch.float32
            }
        }
    
    @classmethod
    def get_embedding_encode_kwargs(cls):
        """임베딩 인코딩 kwargs 반환"""
        return {
            "normalize_embeddings": cls.EMBEDDING_NORMALIZE,
            "prompt_name": "query"
        }

