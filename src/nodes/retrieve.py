"""Hybrid metadata + vectorstore retriever node."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..data_loader import DEFAULT_CSV, DEFAULT_XLSX, load_project_entries
from ..graph_state import GraphState
from ..vector_store import DEFAULT_VECTORSTORE_PATH, load_vectorstore, search_vectorstore

DATA_CSV_PATH = DEFAULT_CSV
DATA_XLSX_PATH = DEFAULT_XLSX
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "3"))
SNIPPET_MAX_CHARS = 600

ENABLE_RERANK = os.getenv("ENABLE_RERANK", "0").lower() not in {"0", "false", "no"}
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-large")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "12"))  # 후보 상한
RERANK_SCORE_FLOOR = float(os.getenv("RERANK_SCORE_FLOOR", "0.0"))  # rerank 최고점이 너무 낮으면 보강 검색

# 멀티쿼리 고도화: 간단 paraphrase 추가 (옵션)
ENABLE_PARAPHRASE = os.getenv("ENABLE_PARAPHRASE", "0").lower() not in {"0", "false", "no"}
PARAPHRASE_N = int(os.getenv("PARAPHRASE_N", "2"))

# BM25 하이브리드 검색 (옵션)
ENABLE_BM25 = os.getenv("ENABLE_BM25", "0").lower() not in {"0", "false", "no"}
BM25_TOP_K = int(os.getenv("BM25_TOP_K", str(RETRIEVAL_TOP_K)))

_PROJECT_ENTRIES: List[Dict[str, str]] = []
_VECTOR_STORE = None
_RERANKER = None
_PARA_LLM = None
_BM25 = None
_BM25 = None


def _ensure_entries() -> List[Dict[str, str]]:
    global _PROJECT_ENTRIES
    if not _PROJECT_ENTRIES:
        _PROJECT_ENTRIES = load_project_entries(DATA_CSV_PATH, DATA_XLSX_PATH)
    return _PROJECT_ENTRIES


def _ensure_vector_store():
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = load_vectorstore(DEFAULT_VECTORSTORE_PATH)
    return _VECTOR_STORE


def _ensure_reranker():
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    if not ENABLE_RERANK:
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError:
        return None
    try:
        _RERANKER = CrossEncoder(RERANK_MODEL)
    except Exception:
        _RERANKER = None
    return _RERANKER


def _ensure_para_llm():
    """멀티쿼리 paraphrase용 소형 LLM. 설치/설정 실패 시 None."""
    global _PARA_LLM
    if _PARA_LLM is not None:
        return _PARA_LLM
    if not ENABLE_PARAPHRASE or PARAPHRASE_N <= 0:
        return None
    try:
        # 기존 providers의 기본 채팅 클라이언트 활용
        from ..providers import get_chat_client
        _PARA_LLM = get_chat_client()
    except Exception:
        _PARA_LLM = None
    return _PARA_LLM


def _generate_paraphrases(question: str) -> List[str]:
    """LLM으로 간단한 변형 질의 생성 (옵션). 실패 시 빈 리스트."""
    llm = _ensure_para_llm()
    if llm is None:
        return []
    prompt = (
        "아래 질문을 검색용 짧은 변형으로 {n}개만 만들어 주세요. 숫자/기관/사업명은 유지하고, "
        "불릿 없이 줄바꿈으로만 나열하세요.\n\n질문: {q}".format(n=PARAPHRASE_N, q=question)
    )
    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        lines = [line.strip("-• \n\t") for line in str(text).splitlines() if line.strip()]
        return lines[:PARAPHRASE_N]
    except Exception:
        return []


def _ensure_bm25():
    """BM25 검색기 초기화 (옵션)."""
    global _BM25
    if _BM25 is not None:
        return _BM25
    if not ENABLE_BM25:
        return None
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except ImportError:
        return None
    entries = _ensure_entries()
    corpus_tokens = []
    for entry in entries:
        text = entry.get("full_text") or entry.get("text_blob", "")
        corpus_tokens.append(_tokenize(text))
    try:
        bm25 = BM25Okapi(corpus_tokens)
    except Exception:
        return None
    _BM25 = (bm25, entries, corpus_tokens)
    return _BM25


def _score_entry(entry: Dict[str, str], question: str, agency: Optional[str], project: Optional[str], keywords: Sequence[str]) -> float:
    score = 0.0
    text_blob = entry.get("text_blob", "")
    summary = entry.get("summary", "")
    full_text = entry.get("full_text", "")
    entry_agency = entry.get("agency", "")
    entry_project = entry.get("project", "")
    lower_question = question.lower()

    if agency and entry_agency and agency in entry_agency:
        score += 2.5
    if project and entry_project and project.replace(" ", "") in entry_project.replace(" ", ""):
        score += 1.5
    if entry_project and entry_project in question:
        score += 1.0
    if summary and any(token for token in (agency or "", project or "") if token and token in summary):
        score += 0.5
    if lower_question and lower_question in text_blob.lower():
        score += 0.5
    if full_text and any(token in full_text for token in (agency or "", project or "")):
        score += 0.5
    if full_text:
        lower_text = full_text.lower()
        keyword_hits = sum(1 for token in keywords if token.lower() in lower_text)
        score += min(keyword_hits * 0.2, 1.0)
    return score


def _format_entry(entry: Dict[str, str], keywords: Sequence[str]) -> str:
    lines = []
    if entry.get("agency"):
        lines.append(f"[기관] {entry['agency']}")
    if entry.get("project"):
        lines.append(f"[사업] {entry['project']}")
    summary = entry.get("summary")
    if summary:
        lines.append(f"[요약] {summary}")
    snippet = _extract_snippet(entry.get("full_text") or entry.get("text_blob", ""), keywords)
    if snippet:
        lines.append(f"[본문 발췌]\n{snippet}")
    return "\n".join(lines)


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", text or "")
    return [token for token in tokens if len(token) > 1]


def _extract_snippet(full_text: str, keywords: Sequence[str]) -> str:
    if not full_text:
        return ""
    if not keywords:
        return full_text[:SNIPPET_MAX_CHARS]
    lower_text = full_text.lower()
    spans: List[Tuple[int, int]] = []
    for keyword in keywords:
        idx = lower_text.find(keyword.lower())
        if idx == -1:
            continue
        start = max(0, idx - 120)
        end = min(len(full_text), idx + 120)
        spans.append((start, end))
    if not spans:
        return full_text[:SNIPPET_MAX_CHARS]
    spans.sort()
    merged_start, merged_end = spans[0]
    for start, end in spans[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
        else:
            break
    snippet = full_text[merged_start:merged_end]
    snippet = snippet.strip()
    if len(snippet) > SNIPPET_MAX_CHARS:
        snippet = snippet[:SNIPPET_MAX_CHARS] + "..."
    return snippet


def _rerank_hits(question: str, hits: List[Dict[str, str]]) -> List[Dict[str, str]]:
    reranker = _ensure_reranker()
    if not reranker or not hits:
        return hits
    candidates = hits[: min(len(hits), RERANK_TOP_N)]
    pairs = [(question, h.get("text", "")) for h in candidates]
    try:
        scores = reranker.predict(pairs)
    except Exception:
        return hits
    reranked = []
    for hit, score in zip(candidates, scores):
        new_hit = dict(hit)
        new_hit["rerank_score"] = float(score)
        reranked.append(new_hit)
    reranked.sort(key=lambda h: h.get("rerank_score", 0), reverse=True)
    return reranked


def retrieve_context(state: GraphState) -> Dict[str, str]:
    question = state.get("current_question", "")
    agency = state.get("agency")
    project = state.get("project")
    entries = _ensure_entries()
    keywords = _tokenize(question)
    scored: List[Tuple[float, Dict[str, str]]] = []
    for entry in entries:
        score = _score_entry(entry, question, agency, project, keywords)
        if score > 0 or not agency:
            scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_entries = [entry for _, entry in scored[:RETRIEVAL_TOP_K]] or entries[:1]

    context_blocks = []
    retrieved_docs: List[Dict[str, str]] = []
    vector_hits: List[Dict[str, str]] = []
    bm25_hits: List[Dict[str, str]] = []
    try:
        store = _ensure_vector_store()
        queries = [question]
        # paraphrase 멀티쿼리 추가 (옵션)
        queries.extend(_generate_paraphrases(question))
        if agency:
            queries.append(f"{agency} {question}")
        if project:
            queries.append(f"{project} {question}")

        merged: Dict[str, Dict[str, str]] = {}
        for q in queries:
            hits = search_vectorstore(q, store, top_k=RETRIEVAL_TOP_K)
            for hit in hits:
                hid = hit.get("id")
                prev = merged.get(hid)
                if (not prev) or (hit.get("score", 0) > prev.get("score", 0)):
                    merged[hid] = hit

        raw_sorted = sorted(merged.values(), key=lambda h: h.get("score", 0), reverse=True)
        vector_hits = _rerank_hits(question, raw_sorted)[:RETRIEVAL_TOP_K]
        best_rerank = vector_hits[0].get("rerank_score", vector_hits[0].get("score", 0)) if vector_hits else 0.0
        low_confidence = ENABLE_RERANK and (RERANK_SCORE_FLOOR > 0) and (best_rerank < RERANK_SCORE_FLOOR)
        if low_confidence:
            # rerank 점수가 너무 낮으면 원본 score 기준 상위 결과를 보강으로 추가
            fallback_hits = raw_sorted[:RETRIEVAL_TOP_K]
            vector_hits = fallback_hits

        if vector_hits:
            vector_formatted = []
            for hit in vector_hits:
                meta = hit.get("metadata", {})
                header = f"[벡터] {meta.get('project') or ''} / {meta.get('agency') or ''} (score={hit.get('rerank_score', hit.get('score', 0)):.2f})"
                vector_formatted.append(f"{header}\n{hit['text']}")
                retrieved_docs.append(
                    {
                        "source": "vector",
                        "file_name": meta.get("file_name", ""),
                        "project": meta.get("project", ""),
                        "agency": meta.get("agency", ""),
                        "score": f"{hit.get('score', 0):.4f}",
                    }
                )
            context_blocks.append("\n\n".join(vector_formatted))
    except FileNotFoundError:
        pass

    seen_files = {doc.get("file_name") for doc in retrieved_docs if doc.get("file_name")}

    # BM25 하이브리드 (옵션)
    bm25_pack = _ensure_bm25()
    if bm25_pack:
        bm25, bm25_entries, _tokens = bm25_pack
        query_tokens = _tokenize(question)
        scores = bm25.get_scores(query_tokens)
        idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for idx in idx_sorted[:BM25_TOP_K]:
            entry = bm25_entries[idx]
            fname = entry.get("file_name", "")
            if fname in seen_files:
                continue
            seen_files.add(fname)
            bm25_hits.append(entry)
            context_blocks.append(_format_entry(entry, keywords))
            retrieved_docs.append(
                {
                    "source": "bm25",
                    "file_name": fname,
                    "project": entry.get("project", ""),
                    "agency": entry.get("agency", ""),
                    "score": f"{scores[idx]:.4f}",
                }
            )

    for entry in top_entries:
        if entry.get("file_name", "") in seen_files:
            continue
        context_blocks.append(_format_entry(entry, keywords))
        retrieved_docs.append(
            {
                "source": "metadata",
                "file_name": entry.get("file_name", ""),
                "project": entry.get("project", ""),
                "agency": entry.get("agency", ""),
                "score": "",
            }
        )
    context = "\n\n".join(block for block in context_blocks if block.strip())
    if history := state.get("history_summary"):
        context = f"[이전 요약] {history}\n\n{context}"

    retrieval_config = {
        "top_k": RETRIEVAL_TOP_K,
        "vector_results": len(vector_hits),
        "bm25_results": len(bm25_hits),
        "metadata_results": len(top_entries),
        "history_included": bool(state.get("history_summary")),
        "rerank_enabled": ENABLE_RERANK,
        "rerank_model": RERANK_MODEL if ENABLE_RERANK else "",
        "paraphrase_enabled": ENABLE_PARAPHRASE,
    }

    return {
        "context": context.strip(),
        "retrieved_docs": retrieved_docs,
        "retrieval_config": retrieval_config,
    }
