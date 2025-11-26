def extract_metadata_compare(state, docs, compare_keys):
    items = []

    for key in compare_keys:
        key_norm = key.replace(" ", "")

        # key와 기관명이 일치하는 문서만 필터링
        matched_docs = []
        for d in docs:
            org = d.metadata.get("org", "").replace(" ", "")
            if key_norm in org or org in key_norm:
                matched_docs.append(d)

        # 혹시 필터링이 안 되면 안전장치로 모든 문서 중에서 기관명 가장 유사한 문서 선택
        if not matched_docs:
            # 문자열 유사도 기반 fallback
            best_doc = None
            best_score = -1
            for d in docs:
                org = d.metadata.get("org", "")
                sim = _string_similarity(key, org)  # 아래에서 정의
                if sim > best_score:
                    best_score = sim
                    best_doc = d
        else:
            # org 필터링된 문서 중 rerank 점수 높은 문서 선택
            best_doc = None
            best_score = -1
            for d, s in zip(docs, state["rerank_scores"]):
                if d in matched_docs and s > best_score:
                    best_score = s
                    best_doc = d

        meta = best_doc.metadata

        items.append({
            "key": key,
            "source": meta.get("source", "미기재"),
            "org": meta.get("org", "미상"),
            "category": meta.get("category", "미상"),
            "budget": meta.get("budget", "미기재"),
            "open_date": meta.get("open_date", "미기재"),
            "end_date": meta.get("end_date", "미기재"),
            "context": best_doc.page_content[:3000],
        })

    return {"mode": "compare", "items": items}


def _string_similarity(a, b):
    a = a.replace(" ", "")
    b = b.replace(" ", "")
    same = sum(x == y for x, y in zip(a, b))
    return same / max(len(a), 1)

def extract_metadata(docs):

    if not docs:
        return {
            "source": "미기재",
            "org": "미상",
            "category": "미상",
            "budget": "미기재",
            "open_date": "미기재",
            "end_date": "미기재",
            "context": "문서 없음"
        }

    doc = docs[0]
    meta = doc.metadata

    return {
        "source": meta.get("source", "미기재"),
        "org": meta.get("org", "미상"),
        "category": meta.get("category", "미상"),
        "budget": meta.get("budget", "미기재"),
        "open_date": meta.get("open_date", "미기재"),
        "end_date": meta.get("end_date", "미기재"),
        "context": doc.page_content
    }