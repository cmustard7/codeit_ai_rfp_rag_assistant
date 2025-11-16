"""Utility helpers for reading data_list.* rows into normalized entries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .document_parser import extract_text

DEFAULT_CSV = Path("data/data_list.csv")
DEFAULT_XLSX = Path("data/data_list.xlsx")

TARGET_COLUMNS = {
    "agency": ["발주 기관", "발주기관", "agency"],
    "project": ["사업명", "project_name", "title"],
    "summary": ["사업 요약", "요약", "summary"],
}
TEXT_COLUMNS = ["텍스트", "본문", "text", "내용"]
FILE_COLUMNS = ["파일명", "file_name", "원본파일"]


def load_dataframe(csv_path: Path = DEFAULT_CSV, xlsx_path: Path = DEFAULT_XLSX) -> pd.DataFrame:
    if csv_path.exists():
        for enc in ("utf-8", "cp949", "euc-kr"):
            try:
                return pd.read_csv(csv_path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(
            "combined", b"", 0, 0, "지원 가능한 인코딩(utf-8/cp949/euc-kr)으로 CSV를 읽을 수 없습니다."
        )
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    raise FileNotFoundError("data_list.csv 또는 data_list.xlsx를 찾을 수 없습니다.")


def pick_value(row: pd.Series, keys: List[str]) -> str:
    for key in keys:
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value:
                return value
    return ""


def load_project_entries(csv_path: Path = DEFAULT_CSV, xlsx_path: Path = DEFAULT_XLSX) -> List[Dict[str, str]]:
    """Return project dictionaries enriched with parsed document text."""
    df = load_dataframe(csv_path, xlsx_path)
    entries: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        agency = pick_value(row, TARGET_COLUMNS["agency"])
        project = pick_value(row, TARGET_COLUMNS["project"])
        summary = pick_value(row, TARGET_COLUMNS["summary"])
        file_name = pick_value(row, FILE_COLUMNS)
        full_text = pick_value(row, TEXT_COLUMNS)
        if not full_text:
            full_text = extract_text(file_name)
        text_blob = "\n".join(
            f"{col}: {value}"
            for col, value in row_dict.items()
            if pd.notna(value) and str(value).strip()
        )
        entries.append(
            {
                "agency": agency,
                "project": project,
                "summary": summary,
                "full_text": full_text,
                "file_name": file_name,
                "text_blob": text_blob,
            }
        )
    return entries
