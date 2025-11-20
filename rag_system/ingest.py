# rag_system/ingest.py
import sys
from pathlib import Path
from typing import List
from .config import Config
from .csv_loader import load_documents_from_csv, validate_csv_format
from .vector_store import VectorStore

def ingest_csv_to_chroma(csv_path: str):
    csv_path = str(csv_path)
    v = validate_csv_format(csv_path)
    if not v.get("valid", False):
        print(f"[오류] CSV 형식 문제: {v.get('message')}")
        return None

    docs = load_documents_from_csv(csv_path)
    if not docs:
        print("[오류] CSV에서 문서가 로드되지 않았습니다.")
        return None

    print(f"로드된 문서 수: {len(docs)} - 벡터 스토어에 추가합니다.")
    # VectorStore 생성 (Chroma + embedding)
    vs = VectorStore(collection_name=Config.COLLECTION_NAME, persist_directory=str(Config.VECTORDB_DIR))
    vs.add_documents(docs)
    print("벡터 적재 완료.")
    return vs

if __name__ == "__main__":
    Config.ensure_dirs()
    csv_file = Path(Config.DATA_DIR) / "final_schedule_for_rag.csv"
    ingest_csv_to_chroma(csv_file)
