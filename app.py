"""FastAPI 기반 간단 업로드+채팅 데모."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import numpy as np
from src.document_parser import extract_text_from_path as parser_extract
from src.text_chunker import split_into_chunks
from src.vector_store import _ensure_embeddings
from src.nodes.answer import answer_question

app = FastAPI(title="RFP RAG Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 세션 단순 저장 (메모리)
SESSIONS: Dict[str, Dict[str, List]] = {}


@app.get("/", response_class=HTMLResponse)
async def home():
    index = Path("index.html")
    if index.exists():
        return index.read_text(encoding="utf-8")
    return "<h1>RFP RAG Demo</h1>"


@app.post("/reset")
async def reset():
    SESSIONS.clear()
    return {"status": "cleared"}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """파일 업로드 → 청킹/임베딩 생성 후 세션 저장."""
    session_id = "default"
    session = SESSIONS.setdefault(session_id, {"chunks": [], "vectors": [], "docs": []})
    embeddings = _ensure_embeddings()

    total_chunks = 0
    for file in files:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / file.filename
            content = await file.read()
            tmp_path.write_bytes(content)

            text = read_text_any(tmp_path)
            if not text:
                continue
            chunks = split_into_chunks(text)
            vectors = embeddings.embed_documents(chunks)
            session["chunks"].extend(chunks)
            session["vectors"].extend(vectors)
            session["docs"].append(file.filename)
            total_chunks += len(chunks)

    return {"session_id": session_id, "chunks": len(session["chunks"]), "docs": session.get("docs", [])}


@app.post("/chat")
async def chat(
    question: str = Form(...),
    session_id: str = Form("default"),
    history: str = Form(""),
):
    """업로드된 세션을 사용해 간단 검색+답변."""
    session = SESSIONS.get(session_id)
    if not session:
        return {"error": "no session, upload first"}

    chunks = session["chunks"]
    vectors = session["vectors"]

    # 간단한 코사인 유사도 검색
    embeddings = _ensure_embeddings()
    q_vec = embeddings.embed_query(question)

    m = np.array(vectors, dtype=float) if vectors else np.zeros((0, 0))
    m_norm = np.linalg.norm(m, axis=1, keepdims=True)
    m_norm[m_norm == 0] = 1
    m = m / m_norm
    q = np.array(q_vec, dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        sims = np.zeros(len(chunks))
    else:
        sims = (m @ (q / q_norm))
    top_k = min(3, len(chunks))
    top_idx = np.argsort(sims)[::-1][:top_k] if len(chunks) else []
    context = "\n\n".join(chunks[i] for i in top_idx)

    state = {
        "current_question": question,
        "context": context,
        "history_summary": history,
    }
    ans = answer_question(state)
    return {"answer": ans.get("last_answer"), "context": context, "usage": ans.get("usage", {})}


def read_text_any(path: Path) -> str:
    """업로드된 임시 경로에서 텍스트 추출 (document_parser.extract_text_from_path 사용)."""
    try:
        return parser_extract(path)
    except Exception:
        return ""


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
