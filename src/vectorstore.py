import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

emb = OpenAIEmbeddings(model="text-embedding-3-small")

def build_vector_store(chunks):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persist_dir = os.path.join(BASE_DIR, "chroma_db")

    vector_store = Chroma(
        collection_name="project_collection",
        embedding_function=emb,
        persist_directory=persist_dir,
    )

    # 기존 DB에 데이터가 있으면 로딩만 하고 넘어가기
    if vector_store._collection.count() > 0:
        print("📌 기존 벡터스토어 감지됨 — 새로 추가하지 않고 로드만 진행")
        return vector_store

    # 새로운 DB 생성
    print("🔨 벡터스토어 비어있음 → 문서 Embedding 시작")

    BATCH_SIZE = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        vector_store.add_documents(batch)
        print(f"{i + len(batch)}/{len(chunks)} 청크 Embedding 완료")

    print("💾 벡터스토어 저장 완료")
    return vector_store
