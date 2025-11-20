# rag_system/retriever.py
import re
from typing import List, Optional
from langchain.schema import Document
from .config import Config

class RetrieverManager:
    """
    Vector 기반 우선 검색 + BM25-like substring fallback
    """
    def __init__(self, documents: Optional[List[Document]] = None, vector_store: Optional[object] = None, top_k: int = 5):
        self.top_k = top_k
        self.vector_store = vector_store  # VectorStore 인스턴스
        self.documents = documents or []  # BM25/substring용 기본 문서 리스트

    def _extract_year(self, query: str) -> Optional[str]:
        m = re.search(r"(20\d{2})", query)
        return m.group(1) if m else None

    def get_relevant_documents(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        top_k = top_k or self.top_k
        results: List[Document] = []

        # 1) vector search 우선
        if self.vector_store:
            try:
                vec_docs = self.vector_store.similarity_search(query, k=max(top_k * 3, 8))
            except Exception:
                vec_docs = []
            results.extend(vec_docs)

        # 2) 연도 필터가 있으면 우선 필터
        year = self._extract_year(query)
        if year and results:
            filtered = [d for d in results if str(d.metadata.get("start_date", "")).startswith(year) or str(d.metadata.get("end_date", "")).startswith(year)]
            if filtered:
                results = filtered

        # 3) 부족하면 문서 본문(또는 title) substring으로 채움 (간단 BM25 대체)
        if len(results) < top_k and self.documents:
            ql = query.lower()
            for d in self.documents:
                if ql in (d.page_content or "").lower() or ql in str(d.metadata.get("title","")).lower():
                    if d not in results:
                        results.append(d)
                    if len(results) >= top_k:
                        break

        return results[:top_k]
