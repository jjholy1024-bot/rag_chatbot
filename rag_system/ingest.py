"""
데이터 적재 스크립트
PDF 파일을 파싱하여 벡터 DB에 적재
"""
import os
import sys
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from langchain.document_loaders import DataFrameLoader
from langchain.schema import Document
from tqdm import tqdm
import random
import torch
from transformers import set_seed

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_system.config import Config
from rag_system.data_parser import parse_pdf_files, get_file_paths
from rag_system.text_splitter import split_documents, add_title_to_document
from rag_system.vector_store import VectorStore
from rag_system.csv_loader import load_documents_from_csv, validate_csv_format


def main():
    """메인 실행 함수"""
    # 시드 설정
    seed = Config.RANDOM_SEED
    torch.manual_seed(seed)
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 디렉토리 생성
    Config.create_directories()
    
    data_dir = Config.DATA_DIR
    if not data_dir.exists():
        print(f"데이터 디렉토리가 없습니다: {data_dir}")
        print("데이터 디렉토리 경로를 확인하거나 PDF/CSV 파일을 준비해주세요.")
        return
    
    # 1. PDF 파일 파싱 (있는 경우)
    print("=" * 50)
    print("1단계: PDF 파일 파싱")
    print("=" * 50)
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if pdf_files:
        parsed_files = parse_pdf_files(
            pdf_folder_path=str(data_dir),
            output_dir=str(Config.PARSED_DIR),
            max_len=Config.PDF_MAX_LEN,
            max_lvl=Config.PDF_MAX_LVL
        )
        print(f"PDF 파싱 완료: {len(parsed_files)}개 파일")
    else:
        print("PDF 파일이 없습니다. CSV 파일만 사용합니다.")
        parsed_files = []
    
    # 2. 문서 로드 (CSV 파일 포함)
    print("\n" + "=" * 50)
    print("2단계: 문서 로드 (CSV 파일 포함)")
    print("=" * 50)
    
    # parsed 디렉토리와 data 디렉토리 모두에서 CSV 파일 수집
    file_paths = []
    
    # 1) 파싱된 CSV 파일 (parsed 디렉토리)
    if Config.PARSED_DIR.exists():
        file_paths.extend(get_file_paths(str(Config.PARSED_DIR)))
    
    # 2) 직접 넣은 CSV 파일 (data 디렉토리)
    csv_files = list(data_dir.glob("*.csv"))
    file_paths.extend([str(f) for f in csv_files])
    
    if not file_paths:
        print("CSV 파일이 없습니다.")
        if not parsed_files:
            print("PDF 파일도 없습니다. 데이터를 준비해주세요.")
            return
    all_documents = []
    
    for file_path in tqdm(file_paths, desc="문서 로딩"):
        if file_path.lower().endswith('.csv'):
            # CSV 파일 형식 검증
            validation = validate_csv_format(file_path)
            if not validation['valid']:
                print(f"경고: {file_path} - {validation.get('message', '형식 오류')}")
                continue
            
            try:
                docs = load_documents_from_csv(file_path)
                all_documents.extend(docs)
                print(f"  ✓ {file_path}: {len(docs)}개 문서 로드")
            except Exception as e:
                print(f"  ✗ {file_path} 로드 실패: {str(e)}")
                continue
    
    print(f"총 {len(all_documents)}개의 문서가 로드되었습니다.")
    
    if not all_documents:
        print("로드된 문서가 없습니다.")
        return
    
    # 3. 텍스트 분할
    print("\n" + "=" * 50)
    print("3단계: 텍스트 분할")
    print("=" * 50)
    
    from rag_system.text_splitter import split_documents, add_title_to_document
    
    split_docs = split_documents(
        all_documents,
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    split_docs = [add_title_to_document(doc) for doc in split_docs]
    
    print(f"총 {len(split_docs)}개의 청크로 분할되었습니다.")
    
    # 4. 벡터 스토어에 적재 (chromadb가 있으면)
    print("\n" + "=" * 50)
    print("4단계: 벡터 스토어 적재")
    print("=" * 50)
    
    try:
        vector_store = VectorStore(
            collection_name=Config.COLLECTION_NAME,
            persist_directory=str(Config.VECTORDB_DIR)
        )
        
        # 문서 추가
        vector_store.add_documents(split_docs)
        
        # 최종 확인
        count = vector_store.get_collection_count()
        print(f"\n벡터 DB 적재 완료!")
        print(f"총 {count}개의 문서가 벡터 스토어에 저장되었습니다.")
        print(f"저장 위치: {Config.VECTORDB_DIR}")
        print(f"컬렉션 이름: {Config.COLLECTION_NAME}")
    except Exception as e:
        print(f"\n벡터 스토어 적재 실패: {e}")
        print("chromadb가 설치되지 않았습니다.")
        print("BM25 검색만 사용 가능합니다.")
        print(f"문서는 메모리에 로드되어 BM25 검색에 사용됩니다.")
        print(f"총 {len(split_docs)}개의 문서 청크가 준비되었습니다.")


if __name__ == "__main__":
    main()

