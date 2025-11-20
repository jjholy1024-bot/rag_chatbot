# rag_system/vector_store.py
from typing import List, Optional
from pathlib import Path
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from .config import Config

try:
    import chromadb
    from langchain_chroma import Chroma
    CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None
    Chroma = None
    CHROMADB_AVAILABLE = False

class VectorStore:
    """
    Chroma 기반 VectorStore 래퍼.
    - 내부적으로 chromadb.PersistentClient와 langchain_chroma.Chroma 객체를 보관
    - similarity_search(query, k) -> List[Document]
    """
    def __init__(self, collection_name: str = None, persist_directory: Optional[str] = None, embedding_model: Optional[HuggingFaceEmbeddings] = None):
        if collection_name is None:
            collection_name = Config.COLLECTION_NAME
        self.collection_name = collection_name
        if persist_directory is None:
            persist_directory = str(Config.VECTORDB_DIR)
        self.persist_directory = persist_directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # embedding 모델 생성 (langchain_huggingface)
        if embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME, model_kwargs={"device": Config.EMBEDDING_DEVICE})
        else:
            self.embedding_model = embedding_model

        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb / langchain_chroma가 필요합니다.")

        # client 및 chroma 초기화
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        # create collection if not exists
        try:
            self._client.get_collection(self.collection_name)
        except Exception:
            self._client.create_collection(self.collection_name)

        # Chroma wrapper (langchain_chroma) for add_documents convenience
        self._chroma = Chroma(client=self._client, collection_name=self.collection_name, embedding_function=self.embedding_model, persist_directory=self.persist_directory)

    def add_documents(self, documents: List[Document]):
        if not documents:
            return
        # langchain_chroma.Chroma.add_documents works with langchain Documents
        self._chroma.add_documents(documents)

    def get_retriever(self, k: int = None):
        if k is None:
            k = Config.VECTOR_SEARCH_K
        return self._chroma.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        chromadb client의 query를 직접 호출하고, 결과를 langchain Document 리스트로 반환.
        """
        if self._client is None:
            return []
        res = self._client.get_collection(self.collection_name).query(query_texts=[query], n_results=k)
        # res['documents'] and res['metadatas'] are lists-of-lists for each query
        docs_list = []
        docs = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        for text, meta in zip(docs, metadatas):
            # meta might be dict; ensure keys are strings
            docs_list.append(Document(page_content=text, metadata=meta or {}))
        return docs_list

    def get_collection_count(self) -> int:
        return self._client.get_collection(self.collection_name).count()
