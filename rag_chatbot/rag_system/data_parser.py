"""
PDF 문서 파싱 및 전처리 모듈
"""
import os
import re
import traceback
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

from .config import Config


def clean_illegal_char(s: str) -> str:
    """불법적인 문자 제거"""
    if not isinstance(s, str):
        return s
    illegal_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
    s = illegal_pattern.sub('', s)
    return re.sub(r'[\u0000-\u001F\u007F-\u009F\uFFFC]', '  ', s)


def clean_text(text: str, title: str) -> str:
    """텍스트 정제"""
    # 허용 문자: 한글, 숫자, 영문, 공백, 특수문자
    pattern = r"[^가-힣a-zA-Z0-9\s~!@#$%^&*()_\-+=\[\]{}|\\;:'\",.<>/?`]"
    cleaned = re.sub(pattern, "-", text)
    cleaned = re.sub(r'\n+', '\n', cleaned)
    cleaned = re.sub(r'[ ]{3,}', '  ', cleaned)
    cleaned = re.sub(r'^(|||)(\s)*', '', cleaned)
    cleaned = re.sub(r'[.]{4,}', '...', cleaned)  # escape sequence 수정
    cleaned = re.sub(r'(법제처)( )*[0-9]+( )*(국가법령정보센터)', '', cleaned).strip()
    cleaned = re.sub(f'^({re.escape(title)})', '', cleaned)
    return cleaned


def get_info(doc: fitz.Document, max_lvl: int) -> List[Dict[str, Any]]:
    """PDF 목차 정보 추출"""
    toc_with_locations = []
    toc_data = doc.get_toc(simple=False)
    
    if len(toc_data) == 0:
        print("목차 정보가 없습니다.")
        return toc_with_locations
    
    lvl_list = [lvl[0] for lvl in toc_data]
    
    try:
        assert max(lvl_list) <= max_lvl
    except AssertionError:
        print(f'목차 레벨이 {max_lvl} 이상입니다.')
        print(f'max_level : {max(lvl_list)}')
        outlier_list = [(lvl[1], lvl[0]) for lvl in toc_data if lvl[0] > max_lvl]
        print(f'outlier : {outlier_list}')
    
    for level, title, page_num, dest in toc_data:
        title = title.strip()
        location_info = {
            "level": level,
            "title": title,
            "page_num": page_num,
            "y": 0.0
        }
        
        if 'to' in dest and isinstance(dest.get('to'), fitz.Point):
            location_info['y'] = dest['to'].y
        else:
            page = doc.load_page(page_num - 1)
            search_results = page.search_for(clean_illegal_char(title).strip())
            if search_results:
                location_info['y'] = search_results[0].y0
            else:
                page = doc.load_page(page_num)
                search_results = page.search_for(clean_illegal_char(title).strip())
                
                if search_results:
                    location_info['y'] = search_results[0].y0
                else:
                    if page_num - 2 > 0:
                        page = doc.load_page(page_num - 2)
                        search_results = page.search_for(clean_illegal_char(title).strip())
                        if search_results:
                            location_info['y'] = search_results[0].y0
                        else:
                            print(f"### ({title}, page: {page_num}) 텍스트를 찾지 못해 y좌표를 0으로 설정합니다.")
                    else:
                        print(f"### ({title}, page: {page_num}) 텍스트를 찾지 못해 y좌표를 0으로 설정합니다.")
        
        toc_with_locations.append(location_info)
    
    print("목차 분석 완료")
    return toc_with_locations


def extract_section(
    pdf_path: str,
    max_len: int = Config.PDF_MAX_LEN,
    max_lvl: int = Config.PDF_MAX_LVL,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """PDF에서 섹션별로 텍스트 추출"""
    doc = fitz.open(pdf_path)
    source = os.path.basename(pdf_path)
    results = []
    
    master_toc = get_info(doc, max_lvl)
    
    if not master_toc:
        print(f"{source}에는 유효한 목차 정보가 없습니다.")
        doc.close()
        return results
    
    print("section별로 텍스트 추출 시작")
    
    level1_title = None
    level2_title = None
    level3_title = None
    
    for i, current_section in enumerate(master_toc):
        current_level = current_section['level']
        current_title = current_section['title']
        current_title = clean_illegal_char(current_title)
        start_page_num = current_section['page_num']
        start_y = current_section['y']
        
        if current_level == 1:
            level1_title = current_title
            level2_title = None
            level3_title = None
        elif current_level == 2:
            level2_title = current_title
            level3_title = None
        elif current_level == 3:
            level3_title = current_title
        
        if i < len(master_toc) - 1:
            next_section = master_toc[i + 1]
            end_page_num = next_section['page_num']
            end_y = next_section['y']
        else:
            end_page_num = doc.page_count
            end_y = doc[end_page_num - 1].rect.height
        
        if verbose:
            print(f" - 처리중 : {current_title} (페이지 {start_page_num}(y:{start_y:.0f}))부터 페이지 {end_page_num}(y:{end_y:.0f} 페이지 직전까지)")
        
        section_text_parts = []
        
        for page_index in range(start_page_num - 1, end_page_num):
            page = doc.load_page(page_index)
            clip_rect = page.rect
            
            if page_index == start_page_num - 1:
                clip_rect.y0 = start_y
            if page_index == end_page_num - 1:
                clip_rect.y1 = end_y
            
            try:
                section_text_parts.append(page.get_text("text", clip=clip_rect))
            except Exception as e:
                print(f"Error: {traceback.format_exc()}")
        
        doc_text = "\n\n".join(section_text_parts).strip()
        final_text = clean_illegal_char(doc_text)
        final_text = clean_text(final_text, current_title)
        
        section_data = {
            'level': current_level,
            'title': current_title,
            'raw_text': final_text,
            'source': source,
            'prep_text': final_text,
            'level1_title': level1_title,
            'level2_title': level2_title,
            'len': len(final_text)
        }
        
        if max_lvl > 2:
            section_data['level3_title'] = level3_title
        
        if current_section['level'] == 1 and len(final_text) > max_len:
            print(" ### 특이사항 발생 - lvl 1")
        elif current_section['level'] != 1 and len(final_text) > max_len:
            print(" ### 특이사항 발생 - lvl 2 이상")
        
        results.append(section_data)
    
    doc.close()
    print('추출 완료')
    return results


def parse_pdf_files(
    pdf_folder_path: str,
    output_dir: Optional[str] = None,
    max_len: int = Config.PDF_MAX_LEN,
    max_lvl: int = Config.PDF_MAX_LVL
) -> List[str]:
    """PDF 폴더의 모든 파일을 파싱하여 CSV로 저장"""
    if output_dir is None:
        output_dir = Config.PARSED_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_folder_path = Path(pdf_folder_path)
    file_path_list = []
    
    for file_path in pdf_folder_path.iterdir():
        if file_path.suffix.lower() == '.pdf':
            file_path_list.append(str(file_path))
    
    parsed_files = []
    
    for pdf_path in file_path_list:
        print(f"### {pdf_path} ###")
        source = os.path.basename(pdf_path)
        
        content_list = extract_section(
            pdf_path,
            verbose=False,
            max_len=max_len,
            max_lvl=max_lvl
        )
        
        if content_list:
            csv_path = output_dir / f'parsed_type1_{source}.csv'
            pd.DataFrame.from_dict(content_list).to_csv(csv_path, index=False)
            parsed_files.append(str(csv_path))
            print(f"파싱 완료: {csv_path}")
        else:
            print('목차가 존재하지 않는 파일이 존재합니다.')
    
    return parsed_files


def get_file_paths(directory: str) -> List[str]:
    """디렉토리에서 PDF와 CSV 파일 경로 수집"""
    file_paths = []
    directory = Path(directory)
    
    for file_path in directory.rglob('*.pdf'):
        file_paths.append(str(file_path))
    for file_path in directory.rglob('*.csv'):
        file_paths.append(str(file_path))
    
    return file_paths

