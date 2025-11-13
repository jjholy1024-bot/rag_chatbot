"""
CSV ?뚯씪 濡쒕뜑 紐⑤뱢
?ㅼ뼇???뺤떇??CSV ?뚯씪??泥섎━?????덈룄濡??뺤옣
而щ읆 援ъ“媛 ?놁뼱???먮룞?쇰줈 媛먯??섏뿬 泥섎━
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
    """?좎쭨 ?⑦꽩 媛먯? (?? 2025-01.20, 2025-01-20 ??"""
    if not isinstance(text, str):
        return False
    # ?ㅼ뼇???좎쭨 ?⑦꽩 媛먯?
    date_patterns = [
        r'\d{4}[-.]\d{1,2}[-.]\d{1,2}',  # 2025-01.20, 2025-01-20
        r'\d{4}??s*\d{1,2}??s*\d{1,2}??,  # 2025??1??20??    ]
    for pattern in date_patterns:
        if re.search(pattern, text):
            return True
    return False


def _auto_detect_columns(data: pd.DataFrame) -> Dict[str, Any]:
    """
    CSV 而щ읆 援ъ“ ?먮룞 媛먯?
    
    Returns:
        媛먯???而щ읆 ?뺣낫 ?뺤뀛?덈━
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
    
    # ?쒖? 而щ읆紐낆씠 ?덈뒗吏 ?뺤씤
    if 'title' in columns:
        result['title_col'] = 'title'
    if 'text' in columns:
        result['text_col'] = 'text'
    if 'prep_text' in columns:
        result['prep_text_col'] = 'prep_text'
    if 'source' in columns:
        result['source_col'] = 'source'
    
    # ?쒖? 而щ읆???놁쑝硫??먮룞 媛먯?
    if not result['text_col'] and not result['prep_text_col']:
        result['auto_detected'] = True
        
        # 泥?踰덉㎏ 而щ읆???좎쭨 ?뺤떇?대㈃ title濡? ??踰덉㎏瑜?text濡?        if len(columns) >= 2:
            first_col = columns[0]
            second_col = columns[1]
            
            # 泥?踰덉㎏ 而щ읆 ?섑뵆 ?뺤씤
            sample_first = data[first_col].dropna().iloc[0] if len(data) > 0 else None
            sample_second = data[second_col].dropna().iloc[0] if len(data) > 0 else None
            
            if sample_first and _detect_date_pattern(str(sample_first)):
                result['title_col'] = first_col
                result['date_col'] = first_col
                result['text_col'] = second_col
                
                # ??踰덉㎏ 而щ읆???덉쑝硫?prep_text濡?(?먮뒗 source濡?
                if len(columns) >= 3:
                    third_col = columns[2]
                    if sample_second and str(sample_second) == str(sample_first):
                        # ??踰덉㎏? ??踰덉㎏媛 媛숈쑝硫?prep_text
                        result['prep_text_col'] = third_col
                    else:
                        # ?ㅻⅤ硫???踰덉㎏瑜?source濡?                        result['source_col'] = third_col
                
                # ??踰덉㎏ 而щ읆???덉쑝硫?source濡?                if len(columns) >= 4:
                    result['source_col'] = columns[3]
            else:
                # 泥?踰덉㎏媛 ?좎쭨媛 ?꾨땲硫?泥?踰덉㎏瑜?text濡?                result['text_col'] = first_col
                if len(columns) >= 2:
                    result['title_col'] = second_col
    
    # text_col???놁쑝硫?泥?踰덉㎏ 鍮꾩뼱?덉? ?딆? 而щ읆 ?ъ슜
    if not result['text_col']:
        for col in columns:
            if col not in [result['title_col'], result['source_col']]:
                non_null_count = data[col].notna().sum()
                if non_null_count > 0:
                    result['text_col'] = col
                    break
    
    # prep_text_col???놁쑝硫?text_col ?ъ슜
    if not result['prep_text_col']:
        result['prep_text_col'] = result['text_col']
    
    return result


def _extract_date_info(date_str: str) -> Dict[str, Any]:
    """?좎쭨 臾몄옄?댁뿉???쒖옉?? 醫낅즺??異붿텧"""
    if not isinstance(date_str, str):
        return {'start_date': None, 'end_date': None, 'date_range': None}
    
    # ?좎쭨 踰붿쐞 ?⑦꽩 (?? "2025-01.20 (?? ~ 2025-01.31 (湲?")
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
    
    # ?⑥씪 ?좎쭨 ?⑦꽩
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
    CSV ?뚯씪?먯꽌 Document 由ъ뒪??濡쒕뱶
    而щ읆 援ъ“媛 ?놁뼱???먮룞?쇰줈 媛먯??섏뿬 泥섎━
    
    Args:
        csv_path: CSV ?뚯씪 寃쎈줈
        text_column: ?띿뒪??而щ읆 ?대쫫 (None?대㈃ ?먮룞 媛먯?)
        title_column: ?쒕ぉ 而щ읆 ?대쫫 (None?대㈃ ?먮룞 媛먯?)
        source_column: 異쒖쿂 而щ읆 ?대쫫 (None?대㈃ ?먮룞 媛먯?)
        prep_text_column: ?꾩쿂由щ맂 ?띿뒪??而щ읆 (None?대㈃ ?먮룞 媛먯?)
    
    Returns:
        Document 由ъ뒪??(媛??됱씠 ?낅┰?곸씤 臾몄꽌)
    """
    try:
        data = pd.read_csv(csv_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(csv_path, encoding='cp949')
        except:
            data = pd.read_csv(csv_path, encoding='latin-1')
    
    if len(data) == 0:
        print(f"寃쎄퀬: {csv_path}媛 鍮꾩뼱?덉뒿?덈떎.")
        return []
    
    # 而щ읆 ?먮룞 媛먯?
    detected = _auto_detect_columns(data)
    
    # ?ъ슜??吏??而щ읆???덉쑝硫??곗꽑 ?ъ슜
    text_col = text_column or detected['text_col'] or detected['prep_text_col']
    title_col = title_column or detected['title_col']
    source_col = source_column or detected['source_col']
    prep_text_col = prep_text_column or detected['prep_text_col'] or text_col
    date_col = detected['date_col']
    
    if not text_col:
        raise ValueError(f"CSV ?뚯씪?먯꽌 ?띿뒪??而щ읆??李얠쓣 ???놁뒿?덈떎. 而щ읆: {list(data.columns)}")
    
    # NaN 媛?泥섎━
    data = data.replace(np.nan, None)
    
    # text_col??鍮꾩뼱?덈뒗 ???쒓굅
    data = data[data[text_col].notna() & (data[text_col].astype(str).str.strip() != '')].reset_index(drop=True)
    
    if len(data) == 0:
        print(f"寃쎄퀬: {csv_path}?먯꽌 ?좏슚??臾몄꽌媛 ?놁뒿?덈떎.")
        return []
    
    # 媛??됱쓣 ?낅┰?곸씤 Document濡?蹂??    documents = []
    for idx, row in data.iterrows():
        # ?띿뒪??而⑦뀗痢?寃곗젙 (prep_text ?곗꽑, ?놁쑝硫?text)
        if prep_text_col and prep_text_col in data.columns and pd.notna(row.get(prep_text_col)):
            page_content = str(row[prep_text_col]).strip()
        elif text_col and pd.notna(row.get(text_col)):
            page_content = str(row[text_col]).strip()
        else:
            continue  # ?띿뒪?멸? ?놁쑝硫??ㅽ궢
        
        # 理쒖냼 湲몄씠 泥댄겕
        if len(page_content) <= 10:
            continue
        
        # 硫뷀??곗씠??援ъ꽦
        metadata = {}
        
        # ?쒕ぉ ?ㅼ젙
        if title_col and title_col in data.columns and pd.notna(row.get(title_col)):
            title = str(row[title_col]).strip()
            metadata['title'] = title
            
            # ?좎쭨 ?뺣낫 異붿텧 諛?硫뷀??곗씠?곗뿉 異붽?
            if date_col and date_col == title_col:
                date_info = _extract_date_info(title)
                if date_info['start_date']:
                    metadata['start_date'] = date_info['start_date']
                    metadata['end_date'] = date_info['end_date']
                    metadata['date_range'] = date_info['date_range']
        else:
            # ?쒕ぉ???놁쑝硫?泥?踰덉㎏ 而щ읆?대굹 ?뚯씪紐??ъ슜
            if date_col:
                metadata['title'] = str(row[date_col]).strip() if pd.notna(row.get(date_col)) else ''
            else:
                metadata['title'] = page_content[:50] if len(page_content) > 50 else page_content
        
        # 異쒖쿂 ?ㅼ젙
        if source_col and source_col in data.columns and pd.notna(row.get(source_col)):
            metadata['source'] = str(row[source_col]).strip()
        else:
            # ?뚯씪紐낆쓣 source濡??ъ슜
            metadata['source'] = Path(csv_path).stem
        
        # ??踰덊샇 異붽? (寃뱀튂???좎쭨 援щ텇??
        metadata['row_index'] = idx
        
        # 紐⑤뱺 而щ읆??硫뷀??곗씠?곗뿉 異붽? (寃?????쒖슜)
        for col in data.columns:
            if col not in [text_col, prep_text_col] and pd.notna(row.get(col)):
                metadata[f'col_{col}'] = str(row[col]).strip()
        
        # Document ?앹꽦
        doc = Document(
            page_content=page_content,
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"  ??{csv_path}: {len(documents)}媛?臾몄꽌 濡쒕뱶 (?먮룞 媛먯?: {detected['auto_detected']})")
    
    return documents


def _set_form_auto(data: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """DataFrame???쒕ぉ ?먮룞 ?앹꽦"""
    def set_form(series):
        """DataFrame ?됱쓣 Document ?뺤떇?쇰줈 蹂??""
        if not series.get(text_col):
            series['title'] = ''
            return series
        
        title = "Title : "
        
        # level1_title, level2_title???덉쑝硫??ъ슜
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
            # ?쒕ぉ???놁쑝硫??뚯씪紐낆씠??泥?50???ъ슜
            title = series.get('source', '')[:50] if 'source' in series else ''
        
        series['title'] = title.strip()
        return series
    
    data = data.apply(set_form, axis=1)
    return data


def validate_csv_format(csv_path: str) -> dict:
    """
    CSV ?뚯씪 ?뺤떇 寃利?(?먮룞 媛먯? 吏??
    
    Returns:
        寃利?寃곌낵 ?뺤뀛?덈━
    """
    try:
        # ?ㅼ뼇???몄퐫???쒕룄
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
                'message': "CSV ?뚯씪??鍮꾩뼱?덉뒿?덈떎."
            }
        
        # 而щ읆 ?먮룞 媛먯?
        detected = _auto_detect_columns(data)
        
        # ?쒖? 而щ읆???덇굅???먮룞 媛먯?媛 ?깃났?섎㈃ ?좏슚
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
            result['message'] = "CSV ?뚯씪?먯꽌 ?띿뒪??而щ읆??李얠쓣 ???놁뒿?덈떎."
        else:
            if detected['auto_detected']:
                result['message'] = f"CSV ?뚯씪 ?뺤떇???먮룞?쇰줈 媛먯??섏뿀?듬땲?? (text: {detected['text_col']})"
            else:
                result['message'] = "CSV ?뚯씪 ?뺤떇???щ컮由낅땲??"
        
        return result
    except Exception as e:
        return {
            'valid': False,
            'columns': [],
            'row_count': 0,
            'message': f"CSV ?뚯씪 ?쎄린 ?ㅻ쪟: {str(e)}"
        }


def get_csv_info(csv_path: str) -> dict:
    """CSV ?뚯씪 ?뺣낫 諛섑솚"""
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


