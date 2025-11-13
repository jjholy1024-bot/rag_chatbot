"""
텍스트 청킹 모듈
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

from .config import Config


def create_text_splitter(
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """텍스트 분할기 생성"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP
) -> List[Document]:
    """문서 리스트를 청크로 분할"""
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    return text_splitter.split_documents(documents)


def add_title_to_document(doc: Document) -> Document:
    """문서에 제목 추가"""
    title = doc.metadata.get('title', '')
    if title:
        doc.page_content = f"{title}\n{doc.page_content}"
    return doc

