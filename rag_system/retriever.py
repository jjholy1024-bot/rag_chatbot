"""
Retriever 모듈
BM25, Vector, Ensemble, Reranker 검색기 구현
"""
from typing import List, Optional
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.schema import Document
import torch

from .config import Config
from .vector_store import VectorStore


class RetrieverManager:
    """Retriever 관리 클래스"""
    
    def __init__(
        self,
        documents: List[Document],
        vector_store: Optional[VectorStore] = None,
        use_reranker: bool = True
    ):
        """
        Args:
            documents: 검색 대상 문서 리스트
            vector_store: VectorStore 인스턴스 (None이면 BM25만 사용)
            use_reranker: Reranker 사용 여부
        """
        self.vector_store = vector_store
        self.documents = documents
        self.use_reranker = use_reranker
        
        # BM25 Retriever 생성
        self.bm25_retriever = BM25Retriever.from_documents(
            documents=documents,
            k=Config.BM25_SEARCH_K
        )
        
        # Vector Retriever 생성 (vector_store가 있으면)
        self.vector_retriever = None
        if vector_store is not None:
            try:
                self.vector_retriever = vector_store.get_retriever(
                    search_kwargs={"k": Config.VECTOR_SEARCH_K}
                )
                # Ensemble Retriever 생성
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.bm25_retriever, self.vector_retriever],
                    weights=Config.ENSEMBLE_WEIGHTS
                )
                print("벡터 검색과 BM25 검색을 함께 사용합니다.")
            except Exception as e:
                print(f"벡터 검색 초기화 실패 (BM25만 사용): {e}")
                import traceback
                traceback.print_exc()
                self.vector_retriever = None
                self.ensemble_retriever = self.bm25_retriever
        else:
            # Vector Store가 없으면 BM25만 사용
            print("벡터 스토어가 없습니다. BM25 검색만 사용합니다.")
            self.ensemble_retriever = self.bm25_retriever
        
        # Reranker 설정
        self.compression_retriever = None
        if use_reranker:
            self._setup_reranker()
        else:
            self.compression_retriever = self.ensemble_retriever
    
    def _setup_reranker(self):
        """Reranker 설정"""
        model_kwargs = {
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "model_kwargs": {
                "dtype": torch.float16 if torch.cuda.is_available() else torch.float32
            },
            "cache_dir": str(Config.MODEL_CACHE_DIR)
        }
        
        reranker_model = HuggingFaceCrossEncoder(
            model_name=Config.RERANKER_MODEL_NAME,
            model_kwargs=model_kwargs
        )
        
        reranker = CrossEncoderReranker(
            model=reranker_model,
            top_n=Config.RERANKER_TOP_N
        )
        
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=self.ensemble_retriever
        )
    
    def get_relevant_documents(
        self,
        query: str,
        score_threshold: float = 0.8,
        top_k: int = 3
    ) -> str:
        """
        쿼리에 대한 관련 문서 검색 및 포맷팅
        
        Args:
            query: 검색 쿼리
            score_threshold: 점수 임계값 (현재 미사용)
            top_k: 반환할 문서 개수
        
        Returns:
            포맷팅된 문서 문자열
        """
        try:
            retrieved_docs = self.compression_retriever.invoke(query)
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return "검색 결과를 가져올 수 없습니다."
        
        if not retrieved_docs:
            return "관련 문서를 찾을 수 없습니다."
        
        # Document 객체인지 확인하고 변환
        processed_docs = []
        for doc in retrieved_docs:
            # 튜플 형태로 반환되는 경우 처리
            if isinstance(doc, tuple):
                if len(doc) >= 1:
                    doc = doc[0]  # 첫 번째 요소가 Document
                else:
                    continue
            
            # Document 객체인지 확인
            if not hasattr(doc, 'page_content'):
                continue
                
            processed_docs.append(doc)
        
        # 최소 길이 필터링
        retrieved_docs = [
            doc for doc in processed_docs
            if len(doc.page_content) > Config.MIN_DOC_LENGTH
        ]
        
        # 상위 k개만 선택 (더 많은 후보를 검색하고 필터링)
        retrieved_docs = retrieved_docs[:top_k]
        
        if not retrieved_docs:
            return "관련 문서를 찾을 수 없습니다."
        
        # 포맷팅 (제목 포함하여 더 명확하게)
        formatted_docs = []
        for i, doc in enumerate(retrieved_docs):
            content = doc.page_content.replace('  ', ' ').strip()
            # 제목이 있으면 포함
            title = doc.metadata.get('title', '')
            if title and title.strip():
                formatted_docs.append(f"<정보 {i+1}>\n제목: {title}\n내용: {content}")
            else:
                formatted_docs.append(f"<정보 {i+1}>\n{content}")
        
        return '\n\n'.join(formatted_docs)
    
    def get_retriever(self):
        """최종 Retriever 반환"""
        return self.compression_retriever

