# langgraph_multisearch/adaptive_retreival.py (기존 코드 유지하면서 로깅만 추가)
import unicodedata
import time
import wandb

from langchain_core.documents import Document

from langgraph_multisearch.reranker import rerank_scores
from langgraph_multisearch.recall import metadata_recall
from langgraph_multisearch.rag_state import RAGState

import rag_logger as logger

def _norm(s):
    return unicodedata.normalize("NFC", str(s or "")).strip()


def ensure_document(obj):
    if isinstance(obj, Document):
        return obj
    if isinstance(obj, dict):
        return Document(
            page_content=_norm(obj.get("page_content", "")),
            metadata={k: _norm(v) if isinstance(v, str) else v 
                      for k, v in obj.get("metadata", {}).items()}
        )
    raise TypeError(f"Unsupported doc: {type(obj)}")


def _build_rerank_text(doc):
    org = str(doc.metadata.get("org", ""))
    cat = str(doc.metadata.get("category", ""))
    source = str(doc.metadata.get("source", ""))
    page = str(doc.page_content)
    return f"{org} {cat} {source}\n{page}"


def single_retrieve(retriever, question: str, top_k: int = 3, full_vs=None):
    """단일 문서 검색 + 로깅"""
    retrieve_start = time.time()
    q_norm = _norm(question)

    # 1) Semantic retrieval
    semantic_start = time.time()
    raw = retriever.invoke(q_norm)
    semantic_docs = [ensure_document(d) for d in raw]
    semantic_time = time.time() - semantic_start

    # 2) Metadata recall
    meta_start = time.time()
    meta_docs = []
    if full_vs is not None:
        meta_docs = metadata_recall(full_vs, q_norm)
        meta_docs = [ensure_document(d) for d in meta_docs]
    meta_time = time.time() - meta_start

    # 3) Merge
    merged = semantic_docs + meta_docs
    merged = merged[:10]

    # 4) Dedupe
    seen = set()
    uniq = []
    for d in merged:
        src = _norm(d.metadata.get("source", ""))
        prev = _norm(d.page_content[:50])
        key = (src, prev)
        if key not in seen:
            seen.add(key)
            uniq.append(d)

    # 5) Rerank
    rerank_start = time.time()
    pairs = [(q_norm, _build_rerank_text(doc)) for doc in uniq]
    scores = rerank_scores(pairs)
    rerank_time = time.time() - rerank_start

    ranked = sorted(zip(uniq, scores), key=lambda x: x[1], reverse=True)
    final_docs = [d for d, _ in ranked[:top_k]]
    final_scores = [s for _, s in ranked[:top_k]]

    total_time = time.time() - retrieve_start

    # 🔥 검색 단계별 시간 로깅
    logger.log({
        "retrieval/single_total_time_sec": total_time,
        "retrieval/semantic_time_sec": semantic_time,
        "retrieval/metadata_recall_time_sec": meta_time,
        "retrieval/rerank_time_sec": rerank_time,
        "retrieval/num_semantic_docs": len(semantic_docs),
        "retrieval/num_metadata_docs": len(meta_docs),
        "retrieval/num_merged_docs": len(merged),
        "retrieval/num_unique_docs": len(uniq),
        "retrieval/num_final_docs": len(final_docs)
    })

    return final_docs, final_scores


def multi_retrieve_for_compare(retriever, compare_keys, top_k=5, full_vs=None):
    """비교 문서 검색 + 로깅"""
    compare_start = time.time()
    
    final_docs = []
    final_scores = []

    for i, key in enumerate(compare_keys):
        key_start = time.time()
        docs, scores = single_retrieve(retriever, key, top_k, full_vs)
        key_time = time.time() - key_start
        
        final_docs.extend(docs)
        final_scores.extend(scores)
        
        # 🔥 각 비교 키별 로깅
        logger.log({
            f"retrieval/compare_key_{i}_time_sec": key_time,
            f"retrieval/compare_key_{i}_query": key,
            f"retrieval/compare_key_{i}_num_docs": len(docs),
            f"retrieval/compare_key_{i}_avg_score": sum(scores) / len(scores) if scores else 0
        })

    total_time = time.time() - compare_start
    
    logger.log({
        "retrieval/compare_total_time_sec": total_time,
        "retrieval/num_compare_keys": len(compare_keys),
        "retrieval/total_docs_retrieved": len(final_docs)
    })

    return final_docs, final_scores


def multi_retrieve(
    state: RAGState,
    question: str,
    vector_store,
    retriever,
    is_compare=False,
    compare_keys=None,
    top_k=5,
    return_scores=False
):
    """멀티 검색 메인 로직"""
    
    if not is_compare or not compare_keys:
        docs, scores = single_retrieve(
            retriever=retriever,
            question=question,
            top_k=top_k,
            full_vs=state["full_vs"]
        )
        strategy = "single"
    else:
        docs, scores = multi_retrieve_for_compare(
            retriever=retriever,
            compare_keys=compare_keys,
            top_k=top_k,
            full_vs=state["full_vs"]
        )
        strategy = "compare"
    
    logger.log({"retrieval/strategy": strategy})

    if return_scores:
        return docs, scores
    return docs


def node_retrieve(state: RAGState):
    """검색 노드 + 상세 로깅"""
    node_start = time.time()
    
    question = state["question"]
    retriever = state["retriever"]
    vector_store = state["vector_store"]

    is_compare = state.get("is_compare", False)
    compare_keys = state.get("compare_keys", [])

    print("\n[RETRIEVE] 멀티 리트리버 실행")

    docs, scores = multi_retrieve(
        state=state,
        question=question,
        vector_store=vector_store,
        retriever=retriever,
        is_compare=is_compare,
        compare_keys=compare_keys,
        top_k=5,
        return_scores=True
    )

    node_time = time.time() - node_start

    # 🔥 검색 결과 상세 로깅
    print("\n[RERANK RESULTS]")
    for i, (doc, score) in enumerate(zip(docs, scores), 1):
        preview = doc.page_content[:80].replace("\n", " ")
        org = doc.metadata.get("org", "???")
        category = doc.metadata.get("category", "???")
        source = doc.metadata.get("source", "???")
        print(f" {i}. score={score:.4f} | {org} | {category} | {source} | {preview}...")
        
        logger.log({
            f"retrieval/doc_{i}_score": float(score),
            f"retrieval/doc_{i}_org": org,
            f"retrieval/doc_{i}_category": category,
            f"retrieval/doc_{i}_source": source[:50],
            f"retrieval/doc_{i}_length": len(doc.page_content)
        })

    # 전체 검색 통계
    logger.log({
        "retrieval/node_time_sec": node_time,
        "retrieval/num_docs": len(docs),
        "retrieval/avg_rerank_score": sum(scores) / len(scores) if scores else 0,
        "retrieval/max_rerank_score": max(scores) if scores else 0,
        "retrieval/min_rerank_score": min(scores) if scores else 0,
        "retrieval/is_compare": is_compare,
        "retrieval/num_compare_keys": len(compare_keys)
    })
    
    return {
        "docs": docs,
        "rerank_scores": scores
    }