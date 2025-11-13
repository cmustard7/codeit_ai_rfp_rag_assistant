def extract_metadata(docs):
    if not docs:
        return {
            "source": "미기재", "org": "미상", "category": "미상",
            "budget": "미기재", "open_date": "미기재", "end_date": "미기재",
            "context": "문서 없음"
        }

    doc = docs[0]
    meta = doc.get("metadata", {}) if isinstance(doc, dict) else doc.metadata
    content = doc.get("page_content", "") if isinstance(doc, dict) else doc.page_content

    # 여러 개 문서를 비교할 경우
    if "sources" in meta:
        srcs = [m.get("source", "미기재") for m in meta["sources"]]
        source_label = " / ".join(srcs)
    else:
        source_label = meta.get("source", "미기재")

    return {
        "source": source_label,
        "org": meta.get("org", "미상"),
        "category": meta.get("category", "미상"),
        "budget": meta.get("budget", "미기재"),
        "open_date": meta.get("open_date", "미기재"),
        "end_date": meta.get("end_date", "미기재"),
        "context": content
    }