"""
Vector Store 관리 모듈
ChromaDB를 사용한 벡터 저장소 관리
"""
import os
from typing import List, Optional
from pathlib import Path

try:
    import chromadb
    from langchain_chroma import Chroma
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Chroma = None

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import Config


class VectorStore:
    """Vector Store 관리 클래스"""
    
    def __init__(
        self,
        collection_name: str = Config.COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[HuggingFaceEmbeddings] = None
    ):
        """
        Args:
            collection_name: 컬렉션 이름
            persist_directory: 벡터DB 저장 디렉토리
            embedding_model: 임베딩 모델 (None이면 자동 생성)
        """
        self.collection_name = collection_name
        
        if persist_directory is None:
            persist_directory = str(Config.VECTORDB_DIR)
        self.persist_directory = persist_directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        if embedding_model is None:
            self.embedding_model = self._create_embedding_model()
        else:
            self.embedding_model = embedding_model
        
        self.vector_store = None
        self._initialize_vector_store()
    
    def _create_embedding_model(self) -> HuggingFaceEmbeddings:
        """임베딩 모델 생성"""
        import torch
        
        model_kwargs = Config.get_embedding_model_kwargs()
        encode_kwargs = Config.get_embedding_encode_kwargs()
        
        return HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME,
            cache_folder=str(Config.MODEL_CACHE_DIR),
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    def _initialize_vector_store(self):
        """벡터 스토어 초기화"""
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb가 설치되지 않았습니다. "
                "BM25 검색만 사용하려면 vector_store=None으로 설정하세요."
            )
        
        # ChromaDB 클라이언트 생성
        client = chromadb.PersistentClient(path=self.persist_directory)
        
        # 기존 컬렉션이 있으면 삭제 (선택적)
        try:
            client.delete_collection(name=self.collection_name)
            print(f"기존 '{self.collection_name}' 컬렉션이 삭제되었습니다.")
        except Exception:
            pass  # 컬렉션이 없으면 무시
        
        # 빈 벡터 스토어 생성
        self.vector_store = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )
    
    def add_documents(self, documents: List[Document]):
        """문서를 벡터 스토어에 추가"""
        if not documents:
            print("추가할 문서가 없습니다.")
            return
        
        self.vector_store.add_documents(documents)
        print(f"{len(documents)}개의 문서가 벡터 스토어에 추가되었습니다.")
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Retriever 반환"""
        if search_kwargs is None:
            search_kwargs = {"k": Config.VECTOR_SEARCH_K}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def get_vector_store(self) -> Chroma:
        """벡터 스토어 객체 반환"""
        return self.vector_store
    
    def get_collection_count(self) -> int:
        """컬렉션의 문서 개수 반환"""
        return self.vector_store._collection.count()
    
    def delete_collection(self):
        """컬렉션 삭제"""
        client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            client.delete_collection(name=self.collection_name)
            print(f"'{self.collection_name}' 컬렉션이 삭제되었습니다.")
        except Exception as e:
            print(f"컬렉션 삭제 실패: {e}")

