"""Hybrid metadata + vectorstore retriever node."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..data_loader import DEFAULT_CSV, DEFAULT_XLSX, load_project_entries
from ..graph_state import GraphState
from ..vector_store import DEFAULT_VECTORSTORE_PATH, load_vectorstore, search_vectorstore

DATA_CSV_PATH = DEFAULT_CSV
DATA_XLSX_PATH = DEFAULT_XLSX
RETRIEVAL_TOP_K = 2
SNIPPET_MAX_CHARS = 600

_PROJECT_ENTRIES: List[Dict[str, str]] = []
_VECTOR_STORE = None


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
    try:
        store = _ensure_vector_store()
        vector_hits = search_vectorstore(question, store, top_k=RETRIEVAL_TOP_K)
        if vector_hits:
            vector_formatted = []
            for hit in vector_hits:
                meta = hit.get("metadata", {})
                header = f"[벡터] {meta.get('project') or ''} / {meta.get('agency') or ''} (score={hit['score']:.2f})"
                vector_formatted.append(f"{header}\n{hit['text']}")
            context_blocks.append("\n\n".join(vector_formatted))
    except FileNotFoundError:
        pass

    context_blocks.extend(_format_entry(entry, keywords) for entry in top_entries)
    context = "\n\n".join(block for block in context_blocks if block.strip())
    if history := state.get("history_summary"):
        context = f"[이전 요약] {history}\n\n{context}"

    return {"context": context.strip()}
