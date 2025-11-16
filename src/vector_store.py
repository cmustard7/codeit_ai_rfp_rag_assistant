"""OpenAI 임베딩을 사용해 JSON 파일로 관리하는 간단한 벡터스토어."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from .data_loader import load_project_entries

DEFAULT_VECTORSTORE_PATH = Path("data/vectorstore.json")
EMBED_MODEL = os.environ.get("LANGGRAPH_EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


@dataclass
class VectorChunk:
    id: str
    text: str
    metadata: Dict[str, str]
    embedding: List[float]


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterable[str]:
    """긴 문서를 겹치는 청크로 잘라 임베딩 품질을 유지한다."""
    if not text:
        return []
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + size)
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def _ensure_embeddings() -> OpenAIEmbeddings:
    load_dotenv(override=True)
    return OpenAIEmbeddings(model=EMBED_MODEL)


def build_vector_chunks() -> List[VectorChunk]:
    """각 사업 엔트리를 VectorChunk 목록으로 나눈다."""
    entries = load_project_entries()
    chunks: List[VectorChunk] = []
    for idx, entry in enumerate(entries):
        base_texts: List[str] = []
        summary = entry.get("summary")
        if summary:
            base_texts.append(summary.strip())
        full_text = entry.get("full_text") or entry.get("text_blob") or ""
        if full_text:
            base_texts.extend(_chunk_text(full_text))
        if not base_texts:
            continue
        for part_idx, text in enumerate(base_texts):
            chunk_id = f"{idx:04d}-{part_idx:03d}"
            metadata = {
                "agency": entry.get("agency") or "",
                "project": entry.get("project") or "",
                "file_name": entry.get("file_name") or "",
                "source_index": str(idx),
            }
            chunks.append(VectorChunk(id=chunk_id, text=text, metadata=metadata, embedding=[]))
    return chunks


def create_vectorstore(output_path: Path = DEFAULT_VECTORSTORE_PATH) -> Path:
    """임베딩을 생성해 JSON 벡터스토어로 저장한다."""
    chunks = build_vector_chunks()
    if not chunks:
        raise ValueError("생성할 텍스트 chunk가 없습니다.")

    embeddings = _ensure_embeddings()
    texts = [chunk.text for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec

    payload = {
        "model": EMBED_MODEL,
        "dimension": len(chunks[0].embedding),
        "chunks": [
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": chunk.embedding,
            }
            for chunk in chunks
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_vectorstore(path: Path = DEFAULT_VECTORSTORE_PATH):
    """JSON 벡터스토어를 읽어 numpy 기반 검색용 구조로 반환한다."""
    if not path.exists():
        raise FileNotFoundError(f"{path} 벡터스토어 파일이 없습니다. 먼저 build_vectorstore.py를 실행하세요.")
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    vectors = []
    texts = []
    metadata = []
    ids = []
    for chunk in chunks:
        ids.append(chunk.get("id"))
        texts.append(chunk.get("text", ""))
        metadata.append(chunk.get("metadata", {}))
        vec = chunk.get("embedding", [])
        vectors.append(vec)
    matrix = np.array(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return {
        "ids": ids,
        "texts": texts,
        "metadata": metadata,
        "normalized": normalized,
    }


def _cosine_sim(query_vec: Sequence[float], doc_matrix: np.ndarray) -> np.ndarray:
    query = np.array(query_vec, dtype=float)
    norm = np.linalg.norm(query)
    if norm == 0:
        return np.zeros(doc_matrix.shape[0])
    query /= norm
    return doc_matrix @ query


def search_vectorstore(question: str, store: dict, top_k: int = 3) -> List[Dict[str, str]]:
    """코사인 유사도를 기준으로 상위 top_k 청크를 반환한다."""
    embeddings = _ensure_embeddings()
    query_vec = embeddings.embed_query(question)
    scores = _cosine_sim(query_vec, store["normalized"])
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append(
            {
                "id": store["ids"][idx],
                "text": store["texts"][idx],
                "metadata": store["metadata"][idx],
                "score": float(scores[idx]),
            }
        )
    return results
