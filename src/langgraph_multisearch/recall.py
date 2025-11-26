import re
import unicodedata
from langchain_core.documents import Document

def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", str(s or "")).strip()

def compare_recall(source_index, key):
    key_norm = _norm(key)
    tokens = [t for t in re.split(r"[^\w가-힣]+", key_norm) if len(t) > 1]

    matched = []

    for src, docs in source_index.items():
        for d in docs:
            meta = d.metadata

            # 문서 식별 필드 3개 통합
            text = " ".join([
                _norm(str(meta.get("org", ""))),
                _norm(str(meta.get("category", ""))),
                _norm(str(meta.get("source", "")))
            ])

            if any(tok in text for tok in tokens):
                matched.append(d)

    return matched


def metadata_recall(full_vs, query):
    q = _norm(query)

    tokens = [tok for tok in re.split(r"[^\w가-힣]+", q) if len(tok) > 2]

    result = []
    for meta, content in zip(full_vs["metadatas"], full_vs["documents"]):

        meta_text = " ".join([
            _norm(str(meta.get("org", ""))),
            _norm(str(meta.get("category", ""))),
            _norm(str(meta.get("source", "")))
        ])

        if any(tok in meta_text for tok in tokens):
            result.append(
                Document(
                    page_content=_norm(content),
                    metadata=meta
                )
            )

    return result