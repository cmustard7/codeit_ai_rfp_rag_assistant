import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

emb = OpenAIEmbeddings(model="text-embedding-3-small")

def build_vector_store(chunks):

    # vectorstore.py 기준으로 프로젝트 root 계산
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persist_dir = os.path.join(BASE_DIR, "chroma_db")

    vector_store = Chroma(
        collection_name="project_collection",
        embedding_function=emb,
        persist_directory=persist_dir,  # 절대경로지만 코드엔 노출 안됨
    )

    BATCH_SIZE = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        vector_store.add_documents(batch)
        print(f"{i + len(batch)}/{len(chunks)} 청크 추가 완료")

    return vector_store
