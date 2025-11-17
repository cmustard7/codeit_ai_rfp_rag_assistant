"""텍스트를 일정한 크기의 청크로 분할하는 도우미."""

from __future__ import annotations

from typing import List


def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """긴 텍스트를 chunk_size 기준으로 겹치게 나눠 반환한다."""
    if not text:
        return []
    normalized = " ".join(text.split())
    length = len(normalized)
    if length <= chunk_size:
        return [normalized]

    chunks: List[str] = []
    start = 0
    while start < length:
        end = min(length, start + chunk_size)
        chunks.append(normalized[start:end])
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks
