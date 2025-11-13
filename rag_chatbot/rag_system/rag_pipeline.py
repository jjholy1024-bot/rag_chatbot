"""
RAG 파이프라인 통합 모듈
전체 RAG 시스템을 통합하여 사용할 수 있는 클래스
"""
from typing import Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from langchain.schema import Document
from langchain.document_loaders import DataFrameLoader

from .config import Config
from .vector_store import VectorStore
from .retriever import RetrieverManager
from .llm_loader import LLMLoader
from .data_parser import get_file_paths


class RAGPipeline:
    """RAG 파이프라인 통합 클래스"""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        use_reranker: bool = True,
        load_llm: bool = True
    ):
        """
        Args:
            collection_name: 벡터DB 컬렉션 이름
            use_reranker: Reranker 사용 여부
            load_llm: LLM 로드 여부
        """
        if collection_name is None:
            collection_name = Config.COLLECTION_NAME
        
        # Vector Store 초기화 (chromadb가 있으면 사용, 없으면 None)
        print("Vector Store 초기화 중...")
        self.vector_store = None
        try:
            self.vector_store = VectorStore(
                collection_name=collection_name,
                persist_directory=str(Config.VECTORDB_DIR)
            )
            print("벡터 스토어 초기화 완료.")
        except Exception as e:
            print(f"벡터 스토어 초기화 실패 (BM25만 사용): {e}")
            print("chromadb가 설치되지 않았거나 오류가 발생했습니다.")
            print("BM25 검색만 사용합니다.")
        
        # 문서 로드 (검색용)
        print("문서 로드 중...")
        self.documents = self._load_documents()
        
        # Retriever 초기화
        print("Retriever 초기화 중...")
        self.retriever_manager = RetrieverManager(
            documents=self.documents,
            vector_store=self.vector_store,
            use_reranker=use_reranker
        )
        
        # LLM 초기화
        self.llm_loader = None
        if load_llm:
            print("LLM 로드 중...")
            self.llm_loader = LLMLoader()
    
    def _load_documents(self) -> List[Document]:
        """벡터 스토어에서 문서 로드 (BM25용)"""
        from .csv_loader import load_documents_from_csv, validate_csv_format
        
        # parsed 디렉토리와 data 디렉토리 모두에서 CSV 파일 수집
        file_paths = []
        
        # 1) 파싱된 CSV 파일 (parsed 디렉토리)
        if Config.PARSED_DIR.exists():
            file_paths.extend(get_file_paths(str(Config.PARSED_DIR)))
        
        # 2) 직접 넣은 CSV 파일 (data 디렉토리)
        if Config.DATA_DIR.exists():
            csv_files = list(Config.DATA_DIR.glob("*.csv"))
            file_paths.extend([str(f) for f in csv_files])
        
        if not file_paths:
            print("경고: CSV 파일을 찾을 수 없습니다.")
            return []
        
        all_documents = []
        
        for file_path in file_paths:
            if file_path.lower().endswith('.csv'):
                # CSV 파일 형식 검증
                validation = validate_csv_format(file_path)
                if not validation['valid']:
                    print(f"경고: {file_path} - {validation.get('message', '형식 오류')}")
                    continue
                
                try:
                    docs = load_documents_from_csv(file_path)
                    all_documents.extend(docs)
                except Exception as e:
                    print(f"경고: {file_path} 로드 실패: {str(e)}")
                    continue
        
        if not all_documents:
            print("경고: 로드된 문서가 없습니다.")
            return []
        
        # 텍스트 분할
        from .text_splitter import split_documents, add_title_to_document
        
        split_docs = split_documents(
            all_documents,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        split_docs = [add_title_to_document(doc) for doc in split_docs]
        
        return split_docs
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        use_llm: bool = True
    ) -> dict:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 질문
            top_k: 검색할 문서 개수
            use_llm: LLM 사용 여부 (False면 검색 결과만 반환)
        
        Returns:
            답변 딕셔너리
        """
        # 1. 문서 검색 (더 많은 후보를 검색하여 정확도 향상)
        context = self.retriever_manager.get_relevant_documents(
            query=question,
            top_k=max(top_k, 5)  # 최소 5개 검색
        )
        
        result = {
            "question": question,
            "context": context
        }
        
        # 2. LLM으로 답변 생성
        if use_llm and self.llm_loader:
            prompt = self.llm_loader.create_prompt(
                query=question,
                context=context
            )
            answer = self.llm_loader.generate_answer(prompt)
            result["answer"] = answer
        else:
            result["answer"] = None
        
        return result

