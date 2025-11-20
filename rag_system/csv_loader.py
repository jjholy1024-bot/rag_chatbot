# rag_system/csv_loader.py
from typing import List, Dict
import pandas as pd
from langchain.schema import Document
from pathlib import Path

def load_documents_from_csv(csv_path: str) -> List[Document]:
    """
    CSV에서 Document 리스트 생성.
    기대 컬럼: start_date, end_date, context_text (또는 prep_text), title, category, source (선택)
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # 안전 처리: 컬럼 이름 보정
    if "context_text" not in df.columns and "prep_text" in df.columns:
        df["context_text"] = df["prep_text"]
    docs = []
    for i, row in df.iterrows():
        content = str(row.get("context_text", "")).strip()
        if not content:
            continue
        metadata: Dict = {
            "title": row.get("title", ""),
            "category": row.get("category", ""),
            "start_date": str(row.get("start_date", "")),
            "end_date": str(row.get("end_date", "")),
            "source": row.get("source", Path(csv_path).stem),
            "row_index": int(i)
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def validate_csv_format(csv_path: str) -> dict:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        return {"valid": False, "message": str(e)}
    required = {"start_date", "end_date"}
    # context_text or prep_text required
    if not ({"context_text"} & set(df.columns) or {"prep_text"} & set(df.columns)):
        return {"valid": False, "message": "context_text 또는 prep_text 컬럼 필요"}
    missing = required - set(df.columns)
    if missing:
        return {"valid": False, "message": f"필수 컬럼 누락: {missing}"}
    return {"valid": True, "message": "OK"}
