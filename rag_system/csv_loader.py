"""
CSV 파일 로더 모듈
다양한 형식의 CSV 파일을 처리할 수 있도록 확장
컬럼 구조가 없어도 자동으로 감지하여 처리
"""
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import re
from langchain.schema import Document
from langchain.document_loaders import DataFrameLoader

from .config import Config


def _detect_date_pattern(text: str) -> bool:
    """날짜 패턴 감지 (예: 2025-01.20, 2025-01-20 등)"""
    if not isinstance(text, str):
        return False
    # 다양한 날짜 패턴 감지
    date_patterns = [
        r'\d{4}[-.]\d{1,2}[-.]\d{1,2}',  # 2025-01.20, 2025-01-20
        r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',  # 2025년 1월 20일
    ]
    for pattern in date_patterns:
        if re.search(pattern, text):
            return True
    return False


def _auto_detect_columns(data: pd.DataFrame) -> Dict[str, Any]:
    """
    CSV 컬럼 구조 자동 감지
    
    Returns:
        감지된 컬럼 정보 딕셔너리
    """
    columns = list(data.columns)
    result = {
        'title_col': None,
        'text_col': None,
        'prep_text_col': None,
        'source_col': None,
        'date_col': None,
        'auto_detected': False
    }
    
    # 표준 컬럼명이 있는지 확인
    if 'title' in columns:
        result['title_col'] = 'title'
    if 'text' in columns:
        result['text_col'] = 'text'
    if 'prep_text' in columns:
        result['prep_text_col'] = 'prep_text'
    if 'source' in columns:
        result['source_col'] = 'source'
    
    # 표준 컬럼이 없으면 자동 감지
    if not result['text_col'] and not result['prep_text_col']:
        result['auto_detected'] = True
        
        # 첫 번째 컬럼이 날짜 형식이면 title로, 두 번째를 text로
        if len(columns) >= 2:
            first_col = columns[0]
            second_col = columns[1]
            
            # 첫 번째 컬럼 샘플 확인
            sample_first = data[first_col].dropna().iloc[0] if len(data) > 0 else None
            sample_second = data[second_col].dropna().iloc[0] if len(data) > 0 else None
            
            if sample_first and _detect_date_pattern(str(sample_first)):
                result['title_col'] = first_col
                result['date_col'] = first_col
                result['text_col'] = second_col
                
                # 세 번째 컬럼이 있으면 prep_text로 (또는 source로)
                if len(columns) >= 3:
                    third_col = columns[2]
                    if sample_second and str(sample_second) == str(sample_first):
                        # 두 번째와 세 번째가 같으면 prep_text
                        result['prep_text_col'] = third_col
                    else:
                        # 다르면 세 번째를 source로
                        result['source_col'] = third_col
                
                # 네 번째 컬럼이 있으면 source로
                if len(columns) >= 4:
                    result['source_col'] = columns[3]
            else:
                # 첫 번째가 날짜가 아니면 첫 번째를 text로
                result['text_col'] = first_col
                if len(columns) >= 2:
                    result['title_col'] = second_col
    
    # text_col이 없으면 첫 번째 비어있지 않은 컬럼 사용
    if not result['text_col']:
        for col in columns:
            if col not in [result['title_col'], result['source_col']]:
                non_null_count = data[col].notna().sum()
                if non_null_count > 0:
                    result['text_col'] = col
                    break
    
    # prep_text_col이 없으면 text_col 사용
    if not result['prep_text_col']:
        result['prep_text_col'] = result['text_col']
    
    return result


def _extract_date_info(date_str: str) -> Dict[str, Any]:
    """날짜 문자열에서 시작일, 종료일 추출"""
    if not isinstance(date_str, str):
        return {'start_date': None, 'end_date': None, 'date_range': None}
    
    # 날짜 범위 패턴 (예: "2025-01.20 (월) ~ 2025-01.31 (금)")
    range_pattern = r'(\d{4}[-.]\d{1,2}[-.]\d{1,2})\s*[~-]\s*(\d{4}[-.]\d{1,2}[-.]\d{1,2})'
    match = re.search(range_pattern, date_str)
    
    if match:
        start_date = match.group(1).replace('.', '-')
        end_date = match.group(2).replace('.', '-')
        return {
            'start_date': start_date,
            'end_date': end_date,
            'date_range': f"{start_date} ~ {end_date}"
        }
    
    # 단일 날짜 패턴
    single_pattern = r'(\d{4}[-.]\d{1,2}[-.]\d{1,2})'
    match = re.search(single_pattern, date_str)
    if match:
        date = match.group(1).replace('.', '-')
        return {
            'start_date': date,
            'end_date': date,
            'date_range': date
        }
    
    return {'start_date': None, 'end_date': None, 'date_range': date_str}


def load_documents_from_csv(
    csv_path: str,
    text_column: Optional[str] = None,
    title_column: Optional[str] = None,
    source_column: Optional[str] = None,
    prep_text_column: Optional[str] = None
) -> List[Document]:
    """
    CSV 파일에서 Document 리스트 로드
    컬럼 구조가 없어도 자동으로 감지하여 처리
    
    Args:
        csv_path: CSV 파일 경로
        text_column: 텍스트 컬럼 이름 (None이면 자동 감지)
        title_column: 제목 컬럼 이름 (None이면 자동 감지)
        source_column: 출처 컬럼 이름 (None이면 자동 감지)
        prep_text_column: 전처리된 텍스트 컬럼 (None이면 자동 감지)
    
    Returns:
        Document 리스트 (각 행이 독립적인 문서)
    """
    try:
        data = pd.read_csv(csv_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(csv_path, encoding='cp949')
        except:
            data = pd.read_csv(csv_path, encoding='latin-1')
    
    if len(data) == 0:
        print(f"경고: {csv_path}가 비어있습니다.")
        return []
    
    # 컬럼 자동 감지
    detected = _auto_detect_columns(data)
    
    # 사용자 지정 컬럼이 있으면 우선 사용
    text_col = text_column or detected['text_col'] or detected['prep_text_col']
    title_col = title_column or detected['title_col']
    source_col = source_column or detected['source_col']
    prep_text_col = prep_text_column or detected['prep_text_col'] or text_col
    date_col = detected['date_col']
    
    if not text_col:
        raise ValueError(f"CSV 파일에서 텍스트 컬럼을 찾을 수 없습니다. 컬럼: {list(data.columns)}")
    
    # NaN 값 처리
    data = data.replace(np.nan, None)
    
    # text_col이 비어있는 행 제거
    data = data[data[text_col].notna() & (data[text_col].astype(str).str.strip() != '')].reset_index(drop=True)
    
    if len(data) == 0:
        print(f"경고: {csv_path}에서 유효한 문서가 없습니다.")
        return []
    
    # 각 행을 독립적인 Document로 변환
    documents = []
    for idx, row in data.iterrows():
        # 텍스트 컨텐츠 결정 (prep_text 우선, 없으면 text)
        if prep_text_col and prep_text_col in data.columns and pd.notna(row.get(prep_text_col)):
            page_content = str(row[prep_text_col]).strip()
        elif text_col and pd.notna(row.get(text_col)):
            page_content = str(row[text_col]).strip()
        else:
            continue  # 텍스트가 없으면 스킵
        
        # 최소 길이 체크
        if len(page_content) <= 10:
            continue
        
        # 메타데이터 구성
        metadata = {}
        
        # 제목 설정
        if title_col and title_col in data.columns and pd.notna(row.get(title_col)):
            title = str(row[title_col]).strip()
            metadata['title'] = title
            
            # 날짜 정보 추출 및 메타데이터에 추가
            if date_col and date_col == title_col:
                date_info = _extract_date_info(title)
                if date_info['start_date']:
                    metadata['start_date'] = date_info['start_date']
                    metadata['end_date'] = date_info['end_date']
                    metadata['date_range'] = date_info['date_range']
        else:
            # 제목이 없으면 첫 번째 컬럼이나 파일명 사용
            if date_col:
                metadata['title'] = str(row[date_col]).strip() if pd.notna(row.get(date_col)) else ''
            else:
                metadata['title'] = page_content[:50] if len(page_content) > 50 else page_content
        
        # 출처 설정
        if source_col and source_col in data.columns and pd.notna(row.get(source_col)):
            metadata['source'] = str(row[source_col]).strip()
        else:
            # 파일명을 source로 사용
            metadata['source'] = Path(csv_path).stem
        
        # 행 번호 추가 (겹치는 날짜 구분용)
        metadata['row_index'] = idx
        
        # 모든 컬럼을 메타데이터에 추가 (검색 시 활용)
        for col in data.columns:
            if col not in [text_col, prep_text_col] and pd.notna(row.get(col)):
                metadata[f'col_{col}'] = str(row[col]).strip()
        
        # Document 생성
        doc = Document(
            page_content=page_content,
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"  ✓ {csv_path}: {len(documents)}개 문서 로드 (자동 감지: {detected['auto_detected']})")
    
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
    CSV 파일 형식 검증 (자동 감지 지원)
    
    Returns:
        검증 결과 딕셔너리
    """
    try:
        # 다양한 인코딩 시도
        try:
            data = pd.read_csv(csv_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            try:
                data = pd.read_csv(csv_path, encoding='cp949')
            except:
                data = pd.read_csv(csv_path, encoding='latin-1')
        
        if len(data) == 0:
            return {
                'valid': False,
                'columns': list(data.columns),
                'row_count': 0,
                'message': "CSV 파일이 비어있습니다."
            }
        
        # 컬럼 자동 감지
        detected = _auto_detect_columns(data)
        
        # 표준 컬럼이 있거나 자동 감지가 성공하면 유효
        has_standard_text = any(col in data.columns for col in ['text', 'prep_text'])
        has_detected_text = detected['text_col'] is not None
        
        result = {
            'valid': has_standard_text or has_detected_text,
            'columns': list(data.columns),
            'row_count': len(data),
            'has_text': has_standard_text or has_detected_text,
            'has_title': 'title' in data.columns or detected['title_col'] is not None,
            'has_source': 'source' in data.columns or detected['source_col'] is not None,
            'auto_detected': detected['auto_detected'],
            'detected_columns': {
                'text_col': detected['text_col'],
                'title_col': detected['title_col'],
                'source_col': detected['source_col'],
                'date_col': detected['date_col']
            },
            'message': ''
        }
        
        if not result['valid']:
            result['message'] = "CSV 파일에서 텍스트 컬럼을 찾을 수 없습니다."
        else:
            if detected['auto_detected']:
                result['message'] = f"CSV 파일 형식이 자동으로 감지되었습니다. (text: {detected['text_col']})"
            else:
                result['message'] = "CSV 파일 형식이 올바릅니다."
        
        return result
    except Exception as e:
        return {
            'valid': False,
            'columns': [],
            'row_count': 0,
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


