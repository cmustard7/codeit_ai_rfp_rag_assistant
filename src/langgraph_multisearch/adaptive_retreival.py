import unicodedata
import rag_logger as rl

from langchain_core.documents import Document

from langgraph_multisearch.reranker import rerank_scores
from langgraph_multisearch.recall import metadata_recall
from langgraph_multisearch.rag_state import RAGState


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


# ---------------------------------------------------------
# 1) 단일 문서 retrieval (기존 로직 기반 + 최소 조정)
# ---------------------------------------------------------
def single_retrieve(retriever, question: str, top_k: int = 3, full_vs=None):
    q_norm = _norm(question)

    # ------------------------------------------------------------
    # 1) Base semantic retrieval
    # ------------------------------------------------------------
    raw = retriever.invoke(q_norm)
    semantic_docs = [ensure_document(d) for d in raw]

    # ------------------------------------------------------------
    # 2) Metadata recall (문서 기반 recall)
    # ------------------------------------------------------------
    meta_docs = []
    if full_vs is not None:
        # metadata_recall은 vector_store의 metadata 기반 recall
        meta_docs = metadata_recall(full_vs, q_norm)
        meta_docs = [ensure_document(d) for d in meta_docs]

    # ------------------------------------------------------------
    # 3) Merge semantic + metadata recall
    # ------------------------------------------------------------
    merged = semantic_docs + meta_docs

    # 제한 (너무 많으면 rerank cost 증가)
    merged = merged[:10]

    # ------------------------------------------------------------
    # 4) dedupe by (source, preview)
    # ------------------------------------------------------------
    seen = set()
    uniq = []
    for d in merged:
        src = _norm(d.metadata.get("source", ""))
        prev = _norm(d.page_content[:50])
        key = (src, prev)
        if key not in seen:
            seen.add(key)
            uniq.append(d)

    # ------------------------------------------------------------
    # 5) rerank
    # ------------------------------------------------------------
    pairs = [(q_norm, _build_rerank_text(doc)) for doc in uniq]
    scores = rerank_scores(pairs)

    ranked = sorted(zip(uniq, scores), key=lambda x: x[1], reverse=True)
    final_docs = [d for d, _ in ranked[:top_k]]
    final_scores = [s for _, s in ranked[:top_k]]

    return final_docs, final_scores


# ---------------------------------------------------------
# 2) 비교 문서 retrieval (compare_keys 개수만큼 단일로 돌림)
# ---------------------------------------------------------
def multi_retrieve_for_compare(retriever, compare_keys, top_k=5, full_vs=None):
    final_docs = []
    final_scores = []

    for key in compare_keys:
        docs, scores = single_retrieve(retriever, key, top_k, full_vs)
        final_docs.extend(docs)
        final_scores.extend(scores)

    return final_docs, final_scores


# ---------------------------------------------------------
# 3) 최종 multi_retrieve (단일 / 비교 자동 분기)
# ---------------------------------------------------------
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

    # 비교가 아니면 단일 retrieval
    if not is_compare or not compare_keys:
        docs, scores = single_retrieve(
            retriever=retriever,
            question=question,
            top_k=top_k,
            full_vs=state["full_vs"]
        )
    else:
        # 비교면 compare_keys 개수만큼 단일 검색
        docs, scores = multi_retrieve_for_compare(
            retriever=retriever,
            compare_keys=compare_keys,
            top_k=top_k,
            full_vs=state["full_vs"]
        )

    if return_scores:
        return docs, scores
    return docs


# ---------------------------------------------------------
# 4) Node(wrapper)
# ---------------------------------------------------------
def node_retrieve(state: RAGState):
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

    print("\n[RERANK RESULTS]")
    for i, (doc, score) in enumerate(zip(docs, scores), 1):
        preview = doc.page_content[:80].replace("\n", " ")
        org = doc.metadata.get("org", "???")
        category = doc.metadata.get("category", "???")
        source = doc.metadata.get("source", "???")
        print(f" {i}. score={score:.4f} | {org} | {category} | {source} | {preview}...")
        rl.log({
            f"retrieved_{i}": {
                "score": float(score),
                "org": org,
                "category": category,
                "source": source,
                "preview": preview
            }
        })

    rl.log_retrieval({
        "retrieval/num_docs": len(docs),
        "retrieval/avg_score": sum(scores) / len(scores) if scores else 0,
        "retrieval/min_score": min(scores) if scores else 0,
        "retrieval/max_score": max(scores) if scores else 0,
    })
    
    return {
        "docs": docs,
        "rerank_scores": scores
    }