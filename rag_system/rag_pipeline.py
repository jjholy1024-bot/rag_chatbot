# rag_system/rag_pipeline.py
from typing import Optional, List
from .config import Config
from .vector_store import VectorStore
from .retriever import RetrieverManager
from .csv_loader import load_documents_from_csv, validate_csv_format
from langchain.schema import Document

# optional LLM loader (사용자 환경에 따라 주석/사용)
try:
    from .llm_loader import LLMLoader
    LLM_AVAILABLE = True
except Exception:
    LLMLoader = None
    LLM_AVAILABLE = False

class RAGPipeline:
    def __init__(self, collection_name: Optional[str] = None, load_llm: bool = False, top_k: int = 5):
        Config.ensure_dirs()
        self.collection_name = collection_name or Config.COLLECTION_NAME
        # VectorStore
        try:
            self.vector_store = VectorStore(collection_name=self.collection_name, persist_directory=str(Config.VECTORDB_DIR))
        except Exception as e:
            print(f"[Warn] Vector store init failed: {e}")
            self.vector_store = None

        # Load raw documents for BM25 fallback
        # look for CSV in data dir
        csv_files = list(Config.DATA_DIR.glob("*.csv"))
        all_docs: List[Document] = []
        for f in csv_files:
            try:
                if validate_csv_format(str(f)).get("valid", False):
                    docs = load_documents_from_csv(str(f))
                    all_docs.extend(docs)
            except Exception:
                continue

        self.retriever = RetrieverManager(documents=all_docs, vector_store=self.vector_store, top_k=top_k)
        self.llm = LLMLoader() if load_llm and LLM_AVAILABLE else None

    def query(self, question: str, top_k: int = 5, use_llm: bool = False) -> dict:
        results = self.retriever.get_relevant_documents(question, top_k=top_k)
        # prepare context text
        context_pieces = []
        for d in results:
            title = d.metadata.get("title") or ""
            sd = d.metadata.get("start_date", "")
            ed = d.metadata.get("end_date", "")
            ctx = d.page_content
            context_pieces.append(f"[{title}] {sd} ~ {ed}\n{ctx}")

        context_text = "\n\n".join(context_pieces) if context_pieces else "관련 문서를 찾을 수 없습니다."
        answer = None
        if use_llm and self.llm:
            prompt = self.llm.create_prompt(query=question, context=context_text)
            answer = self.llm.generate_answer(prompt)
        return {"question": question, "context": context_text, "answer": answer, "hits": results}
