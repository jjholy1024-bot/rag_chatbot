# rag_system/config.py
import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    PARSED_DIR = OUTPUT_DIR / "parsed"
    VECTORDB_DIR = OUTPUT_DIR / "vectordb"
    MODEL_CACHE_DIR = BASE_DIR / "models"

    # 텍스트 분할
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100

    # 임베딩 (가볍고 빠름)
    EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
    EMBEDDING_DEVICE = "cpu"

    # LLM (필요시 변경)
    LLM_MODEL_NAME = "Qwen/Qwen3-4B"
    LLM_QUANTIZATION = False
    LLM_MAX_NEW_TOKENS = 512

    # VectorDB
    COLLECTION_NAME = "academic_schedule"

    # 검색 / 시드
    VECTOR_SEARCH_K = 8
    RANDOM_SEED = 42

    @classmethod
    def ensure_dirs(cls):
        for p in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.VECTORDB_DIR, cls.MODEL_CACHE_DIR]:
            p.mkdir(parents=True, exist_ok=True)
