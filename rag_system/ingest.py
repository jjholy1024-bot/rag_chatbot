"""
데이터 적재 스크립트
PDF/CSV → 텍스트 분할 → 임베딩 → ChromaDB 적재 (로컬 PersistentClient)
- 호환 스택: chromadb==0.4.15, langchain-chroma==0.1.x, Python 3.11 (Windows OK)
"""

import os
import sys
import shutil
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import set_seed

# ---- LangChain / Chroma (권장 최신 스택) ----
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb  # PersistentClient
from langchain.schema import Document
# --------------------------------------------

# ---- 프로젝트 내부 모듈 ----
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag_system.config import Config
from rag_system.data_parser import parse_pdf_files, get_file_paths
from rag_system.text_splitter import split_documents, add_title_to_document
from rag_system.csv_loader import load_documents_from_csv, validate_csv_format


# ===========================
# 유틸: 시드 및 디렉터리 보장
# ===========================
def set_all_seeds(seed: int):
    """torch/numpy/random 재현성 보장"""
    torch.manual_seed(seed)
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def ensure_dirs():
    """Config 상의 주요 디렉터리 생성"""
    dirs = [
        getattr(Config, "DATA_DIR", None),
        getattr(Config, "PARSED_DIR", None),
        getattr(Config, "VECTORDB_DIR", None),
        getattr(Config, "OUTPUT_DIR", None),
    ]
    for p in dirs:
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


# ===========================
# Chroma 초기화 (스키마 충돌 자동복구)
# ===========================
def init_chroma_client(db_path: str, collection_name: str) -> chromadb.Client:
    """chromadb PersistentClient 생성 + 컬렉션 존재 보장"""
    def _new_client():
        return chromadb.PersistentClient(path=db_path)

    client = _new_client()

    try:
        try:
            client.get_collection(collection_name)
        except Exception:
            client.create_collection(collection_name)
        return client
    except Exception as e:
        msg = str(e).lower()
        if "no such column" in msg or "schema" in msg or "topic" in msg:
            print("[경고] ChromaDB 스키마 충돌 감지 → 벡터DB 폴더 재생성")
            shutil.rmtree(db_path, ignore_errors=True)
            client = _new_client()
            client.create_collection(collection_name)
            return client
        raise


# ===========================
# 메인 실행
# ===========================
def main():
    # 0) 시드 고정
    seed = getattr(Config, "RANDOM_SEED", 42)
    set_all_seeds(seed)

    # 1) 폴더 보장 + 자동 생성
    ensure_dirs()

    data_dir = Path(Config.DATA_DIR)
    if not data_dir.exists():
        print(f"[자동 생성] 데이터 폴더 생성: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)

    # 샘플 CSV 자동 생성 (PDF/CSV 모두 없을 경우)
    has_pdf = any(data_dir.glob("*.pdf"))
    has_csv = any(data_dir.glob("*.csv"))
    if not (has_pdf or has_csv):
        sample_csv = data_dir / "sample.csv"
        sample_csv.write_text(
            "title,text,prep_text,source\n"
            "예시 제목,예시 내용,예시 내용,샘플 파일\n",
            encoding="utf-8"
        )
        print(f"[참고] PDF/CSV 없음 → 샘플 CSV 생성: {sample_csv}")

    # 2) PDF 파싱
    print("=" * 50)
    print("1단계: PDF 파일 파싱")
    print("=" * 50)
    pdf_files = list(data_dir.glob("*.pdf"))
    if pdf_files:
        try:
            parsed_files = parse_pdf_files(
                pdf_folder_path=str(data_dir),
                output_dir=str(Config.PARSED_DIR),
                max_len=getattr(Config, "PDF_MAX_LEN", 500000),
                max_lvl=getattr(Config, "PDF_MAX_LVL", 4),
            )
            print(f"PDF 파싱 완료: {len(parsed_files)}개 파일")
        except Exception as e:
            print(f"[경고] PDF 파싱 중 오류: {e} (CSV만 사용)")
            parsed_files = []
    else:
        print("PDF 파일이 없습니다. CSV만 사용합니다.")
        parsed_files = []

    # 3) CSV 로드
    print("\n" + "=" * 50)
    print("2단계: 문서 로드 (CSV 포함)")
    print("=" * 50)
    file_paths: List[str] = []
    if Config.PARSED_DIR.exists():
        file_paths.extend(get_file_paths(str(Config.PARSED_DIR)))
    file_paths.extend([str(f) for f in data_dir.glob("*.csv")])

    if not file_paths and not parsed_files:
        print("[오류] 로드할 CSV 또는 PDF 데이터가 없습니다.")
        return

    all_documents: List[Document] = []
    for file_path in tqdm(file_paths, desc="문서 로딩"):
        if file_path.lower().endswith(".csv"):
            validation = validate_csv_format(file_path)
            if not validation.get("valid", False):
                print(f"[경고] {file_path} - {validation.get('message', '형식 오류')}")
                continue
            try:
                docs = load_documents_from_csv(file_path)
                all_documents.extend(docs)
                print(f"  ✓ {file_path}: {len(docs)}개 문서 로드")
            except Exception as e:
                print(f"  ✗ {file_path} 로드 실패: {e}")
                continue

    if not all_documents:
        print("[오류] 문서를 하나도 로드하지 못했습니다.")
        return

    # 4) 텍스트 분할
    print("\n" + "=" * 50)
    print("3단계: 텍스트 분할")
    print("=" * 50)
    split_docs = split_documents(
        all_documents,
        chunk_size=getattr(Config, "CHUNK_SIZE", 800),
        chunk_overlap=getattr(Config, "CHUNK_OVERLAP", 100),
    )
    split_docs = [add_title_to_document(doc) for doc in split_docs]
    print(f"총 {len(split_docs)}개의 청크 생성 완료")

    # 5) 임베딩 + Chroma 적재
    print("\n" + "=" * 50)
    print("4단계: ChromaDB 벡터 스토어 적재")
    print("=" * 50)

    try:
        model_name = getattr(Config, "EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
        device = getattr(Config, "DEVICE", "cpu")

        print(f"임베딩 모델 로드 중: {model_name}")
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
        print(f"임베딩 모델 로드 완료. (사용 장치: {device})")

        db_path = str(Config.VECTORDB_DIR)
        Path(db_path).mkdir(parents=True, exist_ok=True)
        collection_name = Config.COLLECTION_NAME

        print(f"ChromaDB 로컬 클라이언트 생성 (저장 위치: {db_path})")
        client = init_chroma_client(db_path, collection_name)

        # 기존 컬렉션 삭제 후 재생성
        try:
            client.delete_collection(collection_name)
            print(f"기존 컬렉션 '{collection_name}' 삭제 완료 (덮어쓰기)")
        except Exception:
            print(f"기존 컬렉션 '{collection_name}' 없음 → 새로 생성 예정")

        client.create_collection(collection_name)
        print(f"컬렉션 '{collection_name}' 새로 생성 완료")

        # Chroma 인스턴스 생성
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=db_path,
        )

        print(f"{len(split_docs)}개의 청크를 ChromaDB에 추가 중...")
        vector_store.add_documents(split_docs)

        count = vector_store._collection.count()
        print(f"\n✅ 벡터 스토어 적재 완료! (총 {count}개 청크 저장)")
        print(f"저장 위치: {db_path}")
        print(f"컬렉션 이름: {collection_name}")

    except ImportError as ie:
        print("[오류] 필요한 패키지가 없습니다. requirements를 확인하세요.")
        print(f"상세: {ie}")
    except Exception as e:
        print(f"[오류] 벡터 스토어 적재 실패: {e}")


if __name__ == "__main__":
    main()

# 버전 설명
# 📁 폴더 자동 생성	data, output, parsed, vectordb 자동 생성
# 🧾 샘플 CSV 자동 생성	PDF/CSV 없을 경우 data/sample.csv 자동 생성
# 💾 persist() 제거	langchain_chroma 호환 완료 (에러 제거됨)
# ⚡ 스키마 자동 복구	“no such column…” 오류 시 폴더 자동 리셋
# 🌐 기본 임베딩	intfloat/multilingual-e5-small (빠르고 가벼움)
# 🧠 완전 독립 실행 가능	첫 실행 시 바로 벡터스토어 생성 테스트 가능

#실행순서
# cd C:\rag_chatbot
# .\.venv\Scripts\activate
# python rag_system\ingest.py