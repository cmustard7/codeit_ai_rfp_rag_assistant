import os
import pdfplumber
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import olefile, zlib, struct, re

# ---------------------------------------------------------
# HWP 텍스트 파서
# ---------------------------------------------------------
def get_hwp_text(filename):
    f = olefile.OleFileIO(filename)
    dirs = f.listdir()

    if ["FileHeader"] not in dirs or ["\x05HwpSummaryInformation"] not in dirs:
        raise Exception("Not valid HWP file.")

    header = f.openstream('FileHeader')
    header_data = header.read()
    is_compressed = (header_data[36] & 1) == 1

    nums = [int(d[1][len('Section'):]) for d in dirs if d[0] == 'BodyText']
    sections = [f'BodyText/Section{x}' for x in sorted(nums)]

    text = ""
    for section in sections:
        bodytext = f.openstream(section)
        data = bodytext.read()
        unpacked = zlib.decompress(data, -15) if is_compressed else data

        i, size = 0, len(unpacked)
        while i < size:
            header = struct.unpack_from("<I", unpacked, i)[0]
            rec_type = header & 0x3ff
            rec_len = (header >> 20) & 0xfff
            if rec_type == 67:
                rec_data = unpacked[i + 4:i + 4 + rec_len]
                try:
                    text += rec_data.decode('utf-16')
                except:
                    pass
                text += "\n"
            i += 4 + rec_len
        text += "\n"
    cleaned = re.sub(r'[^\uAC00-\uD7A3\u3131-\u318F\u1100-\u11FF\s0-9A-Za-z.,()「」<>·-]', '', text)
    return cleaned.strip()

# ---------------------------------------------------------
# CSV 로드 및 매핑
# ---------------------------------------------------------
def clean_filename(name):
    import unicodedata
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFC", name)   # ← 이 한 줄 추가
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^\w가-힣\- ]", "", name)
    name = name.strip()
    return name

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["파일명_clean"] = df["파일명"].apply(clean_filename)
    return df

def get_metadata_from_csv(filename, meta_df):
    base = os.path.basename(filename)
    base_clean = clean_filename(base)

    row = meta_df[meta_df["파일명_clean"] == base_clean]
    if row.empty:
        return {
            "source": base,
            "org": "미상",
            "category": "미상",
            "doc_type": "미상"
        }

    row = meta_df[meta_df["파일명_clean"] == base_clean].iloc[0]
    doc_type = "제안요청서" if "제안" in str(row.get("텍스트", "")) else "일반문서"
    
    return {
        "source": base,
        "org": str(row.get("발주 기관", "미상")).strip(),
        "category": str(row.get("사업명", "미상")).strip(),
        "doc_type": doc_type,
        "budget": str(row.get("사업 금액", "")),
        "open_date": str(row.get("공개 일자", "")),
        "start_date": str(row.get("입찰 참여 시작일", "")),
        "end_date": str(row.get("입찰 참여 마감일", "")),
    }

def clean_text_for_rag(text):
    # 1) CRLF → LF 통일
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 2) 제어문자 삭제 (단, \n은 남김)
    # \x00-\x08, \x0B-\x0C, \x0E-\x1F, \x7F 만 제거
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)

    # 3) 각 줄 오른쪽 공백만 정리 (왼쪽은 유지해도 됨, 제목 들여쓰기 살릴거면)
    lines = [line.rstrip() for line in text.split('\n')]
    text = "\n".join(lines)

    # 4) 3줄 이상 빈 줄 → 2줄로 축소
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# ---------------------------------------------------------
# 파일 로더
# ---------------------------------------------------------
def load_hwp(filepath, meta_df):
    text = get_hwp_text(filepath)
    text = clean_text_for_rag(text)
    meta = get_metadata_from_csv(os.path.basename(filepath), meta_df)
    return [Document(page_content=text, metadata=meta)]

def load_pdf(filepath, meta_df):
    documents = []
    meta = get_metadata_from_csv(os.path.basename(filepath), meta_df)

    with pdfplumber.open(filepath) as pdf:
        for idx, page in enumerate(pdf.pages):
            page_text = page.extract_text(x_tolerance=1, y_tolerance=2) or ""
            page_text = clean_text_for_rag(page_text)

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        **meta,
                        "page": idx + 1   # ← 페이지 정보
                    }
                )
            )
    return documents

def load_document(filepath, meta_df):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return load_pdf(filepath, meta_df)
    elif ext == ".hwp":
        return load_hwp(filepath, meta_df)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")