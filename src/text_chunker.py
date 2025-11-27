"""텍스트를 일정한 크기의 청크로 분할하는 도우미."""

from __future__ import annotations

import re
from typing import List


HEADING_PATTERNS = [
    r"^\\s*제?\\d+장",          # 제1장, 1장
    r"^\\s*제?\\d+절",          # 제1절, 1절
    r"^\\s*\\d+\\.\\s",         # 1. 개요
    r"^\\s*\\d+\\.\\d+\\s",     # 1.1 세부
    r"^\\s*[가-힣A-Z]\\.\\s",    # 가. 또는 A.
]


def _split_by_heading(text: str) -> List[str]:
    """목차/섹션 헤더를 감지해 의미 단위로 1차 분리."""
    if not text:
        return []
    lines = text.splitlines()
    sections: List[str] = []
    buf: List[str] = []
    heading_regex = re.compile("|".join(HEADING_PATTERNS))
    for line in lines:
        if heading_regex.match(line):
            if buf:
                sections.append("\n".join(buf).strip())
                buf = []
        buf.append(line)
    if buf:
        sections.append("\n".join(buf).strip())
    return sections


def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    긴 텍스트를 의미 단위(헤더)로 1차 분리한 뒤, 길이에 따라 적응형 chunk_size로 슬라이싱.
    """
    if not text:
        return []
    # 1차: 의미 단위 분리
    sections = _split_by_heading(text) or [text]

    chunks: List[str] = []
    for sec in sections:
        if not sec:
            continue
        normalized = " ".join(sec.split())
        length = len(normalized)

        # 적응형 chunk 크기: 섹션이 길면 1.5배로 확장
        adaptive_chunk = chunk_size
        adaptive_overlap = overlap
        if length > chunk_size * 3:
            adaptive_chunk = int(chunk_size * 1.5)
            adaptive_overlap = int(overlap * 1.5)

        if length <= adaptive_chunk:
            chunks.append(normalized)
            continue

        start = 0
        while start < length:
            end = min(length, start + adaptive_chunk)
            chunks.append(normalized[start:end])
            if end == length:
                break
            start = max(0, end - adaptive_overlap)
    return chunks
