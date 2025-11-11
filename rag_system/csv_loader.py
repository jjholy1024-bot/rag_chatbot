"""
CSV 파일 로더 모듈
다양한 형식의 CSV 파일을 처리할 수 있도록 확장
"""
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from langchain.schema import Document
from langchain.document_loaders import DataFrameLoader

from .config import Config


def load_documents_from_csv(
    csv_path: str,
    text_column: str = "text",
    title_column: Optional[str] = "title",
    source_column: Optional[str] = "source",
    prep_text_column: str = "prep_text"
) -> List[Document]:
    """
    CSV 파일에서 Document 리스트 로드
    
    Args:
        csv_path: CSV 파일 경로
        text_column: 텍스트 컬럼 이름 (기본값: "text")
        title_column: 제목 컬럼 이름 (기본값: "title", None이면 자동 생성)
        source_column: 출처 컬럼 이름 (기본값: "source")
        prep_text_column: 전처리된 텍스트 컬럼 (기본값: "prep_text", 없으면 text_column 사용)
    
    Returns:
        Document 리스트
    """
    data = pd.read_csv(csv_path)
    
    # prep_text가 있으면 사용, 없으면 text_column 사용
    if prep_text_column in data.columns:
        text_col = prep_text_column
    elif text_column in data.columns:
        text_col = text_column
    else:
        raise ValueError(f"CSV 파일에 '{text_column}' 또는 '{prep_text_column}' 컬럼이 없습니다.")
    
    # NaN 값 처리
    data = data.replace(np.nan, None).dropna(subset=[text_col])
    
    # set_form 함수 적용 (제목 생성)
    if title_column and title_column not in data.columns:
        # 제목 자동 생성
        data = _set_form_auto(data, text_col)
    elif title_column:
        # 제목 컬럼이 있으면 그대로 사용
        pass
    else:
        # 제목 없이 처리
        if 'title' not in data.columns:
            data['title'] = ''
    
    # 최소 길이 필터링
    data = data.loc[data[text_col].str.len() > 10].reset_index(drop=True)
    
    if len(data) == 0:
        print(f"경고: {csv_path}에서 유효한 문서가 없습니다.")
        return []
    
    # Document 로드
    columns_to_use = []
    if title_column and title_column in data.columns:
        columns_to_use.append(title_column)
    if source_column and source_column in data.columns:
        columns_to_use.append(source_column)
    columns_to_use.append(text_col)
    
    loader = DataFrameLoader(
        data[columns_to_use],
        page_content_column=text_col
    )
    documents = loader.load()
    
    return documents


def _set_form_auto(data: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """DataFrame에 제목 자동 생성"""
    def set_form(series):
        """DataFrame 행을 Document 형식으로 변환"""
        if not series.get(text_col):
            series['title'] = ''
            return series
        
        title = "Title : "
        
        # level1_title, level2_title이 있으면 사용
        if 'level1_title' in series:
            if series['level1_title']:
                title += series['level1_title']
                if series.get('level2_title'):
                    title += f" > {series['level2_title']}"
            else:
                title = ""
        elif 'title' in series and series['title']:
            title = series['title']
        else:
            # 제목이 없으면 파일명이나 첫 50자 사용
            title = series.get('source', '')[:50] if 'source' in series else ''
        
        series['title'] = title.strip()
        return series
    
    data = data.apply(set_form, axis=1)
    return data


def validate_csv_format(csv_path: str) -> dict:
    """
    CSV 파일 형식 검증
    
    Returns:
        검증 결과 딕셔너리
    """
    try:
        data = pd.read_csv(csv_path)
        
        required_columns = []
        optional_columns = ['title', 'source', 'prep_text', 'text', 
                          'level1_title', 'level2_title', 'level3_title']
        
        # 최소한 하나의 텍스트 컬럼은 필요
        has_text = any(col in data.columns for col in ['text', 'prep_text'])
        
        result = {
            'valid': has_text,
            'columns': list(data.columns),
            'row_count': len(data),
            'has_text': has_text,
            'has_title': 'title' in data.columns,
            'has_source': 'source' in data.columns,
            'message': ''
        }
        
        if not has_text:
            result['message'] = "CSV 파일에 'text' 또는 'prep_text' 컬럼이 필요합니다."
        else:
            result['message'] = "CSV 파일 형식이 올바릅니다."
        
        return result
    except Exception as e:
        return {
            'valid': False,
            'message': f"CSV 파일 읽기 오류: {str(e)}"
        }


def get_csv_info(csv_path: str) -> dict:
    """CSV 파일 정보 반환"""
    try:
        data = pd.read_csv(csv_path)
        return {
            'file': csv_path,
            'rows': len(data),
            'columns': list(data.columns),
            'sample': data.head(1).to_dict('records')[0] if len(data) > 0 else {}
        }
    except Exception as e:
        return {
            'file': csv_path,
            'error': str(e)
        }


