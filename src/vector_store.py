"""OpenAI 임베딩을 사용해 JSON 파일로 관리하는 간단한 벡터스토어."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

try:  # pragma: no cover
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    HuggingFaceEmbeddings = None  # type: ignore
from .data_loader import load_project_entries
from .text_chunker import split_into_chunks

DEFAULT_VECTORSTORE_PATH = Path("data/vectorstore.json")
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "openai").lower()  # openai | hf
EMBED_MODEL = os.environ.get("LANGGRAPH_EMBED_MODEL", "text-embedding-3-small")
HF_EMBED_MODEL = os.environ.get("HF_EMBED_MODEL", "BAAI/bge-m3")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "300"))
CHROMA_DIR = (Path("store/chroma")).resolve()

@dataclass
class VectorChunk:
    id: str
    text: str
    metadata: Dict[str, str]
    embedding: List[float]


def _ensure_embeddings():
    load_dotenv(override=True)
    if EMBED_PROVIDER == "hf" and HuggingFaceEmbeddings is not None:
        return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    return OpenAIEmbeddings(model=EMBED_MODEL)

def _embed_in_batches(embeddings: OpenAIEmbeddings, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    OpenAI 임베딩의 요청당 토큰 수 제한(300k)을 피하기 위해
    texts 를 여러 번 나누어 embed_documents 를 호출하는 헬퍼 함수
    """
    all_vectors: List[List[float]] = []
    n = len(texts)

    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        # 필요하면 디버깅 로그
        # print(f"Embedding batch {i} ~ {i + len(batch) - 1} / {n}")
        batch_vectors = embeddings.embed_documents(batch)
        all_vectors.extend(batch_vectors)

    return all_vectors

def build_chroma_store(chunks: List[VectorChunk]):
    """기존 JSON 벡터스토어와 별개로 Chroma에도 저장."""
    # 매번 새로 구축할 때 이전 데이터가 섞이지 않도록 정리
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # HF / OpenAI 어떤 임베딩이든, 이미 chunk.embedding 에 들어있다고 가정
    client = Chroma(
        collection_name="rfp_chunks",
        persist_directory=str(CHROMA_DIR),
        embedding_function=None,  # 우리는 직접 임베딩해서 넣을거라 None
    )

    # langchain-chroma 최신 버전에서는 add 대신 내부 컬렉션에 직접 추가
    client._collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=[c.embedding for c in chunks],
    )
    # Chroma 버전에 따라 persist 위치가 다를 수 있어 안전하게 처리
    if hasattr(client, "persist"):
        client.persist()
    elif hasattr(client, "_client") and hasattr(client._client, "persist"):
        client._client.persist()
    return client

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
            base_texts.extend(split_into_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP))
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
    # 기존 산출물 정리: JSON 파일 및 Chroma 디렉터리
    if output_path.exists():
        output_path.unlink()
    chunks = build_vector_chunks()
    if not chunks:
        raise ValueError("생성할 텍스트 chunk가 없습니다.")

    embeddings = _ensure_embeddings()
    texts = [chunk.text for chunk in chunks]
    vectors = _embed_in_batches(embeddings, texts, batch_size=64)

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

    # 🔥 선택: Chroma도 함께 구축
    try:
        build_chroma_store(chunks)
    except Exception as e:
        print(f"[WARN] Chroma 벡터스토어 구축 중 에러 발생: {e}")

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


def _mmr(
    query_vec: Sequence[float],
    doc_matrix: np.ndarray,
    top_k: int,
    lambda_diversity: float = 0.5,
) -> List[int]:
    """간단한 MMR로 다양한 후보 선택 (doc_matrix는 정규화된 벡터)."""
    if doc_matrix.size == 0:
        return []
    sims = _cosine_sim(query_vec, doc_matrix)
    selected = []
    candidates = set(range(doc_matrix.shape[0]))
    while candidates and len(selected) < top_k:
        if not selected:
            idx = int(np.argmax(sims[list(candidates)]))
            idx = list(candidates)[idx]
            selected.append(idx)
            candidates.remove(idx)
            continue
        mmr_scores = {}
        for idx in candidates:
            diversity = max(
                np.dot(doc_matrix[idx], doc_matrix[j]) for j in selected
            )
            mmr_scores[idx] = lambda_diversity * sims[idx] - (1 - lambda_diversity) * diversity
        best = max(mmr_scores, key=mmr_scores.get)
        selected.append(best)
        candidates.remove(best)
    return selected

def search_chroma(question: str, top_k: int = 3):
    """Chroma에 저장된 벡터스토어에서 검색."""
    embeddings = _ensure_embeddings()
    query_vec = embeddings.embed_query(question)

    client = Chroma(
        collection_name="rfp_chunks",
        persist_directory=str(CHROMA_DIR),
        embedding_function=None,
    )
    res = client._collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
    )

    results = []
    for i in range(len(res["ids"][0])):
        results.append(
            {
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score": float(res["distances"][0][i]) if "distances" in res else 0.0,
            }
        )
    return results



def search_vectorstore(question: str, store: dict, top_k: int = 3) -> List[Dict[str, str]]:
    """코사인 유사도를 기준으로 상위 top_k 청크를 반환한다."""
    enable_mmr = os.environ.get("ENABLE_MMR", "0").lower() not in {"0", "false", "no"}
    mmr_lambda = float(os.environ.get("MMR_LAMBDA", "0.5"))
    embeddings = _ensure_embeddings()
    query_vec = embeddings.embed_query(question)
    scores = _cosine_sim(query_vec, store["normalized"])
    if enable_mmr:
        top_indices = _mmr(query_vec, store["normalized"], top_k=top_k, lambda_diversity=mmr_lambda)
    else:
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
