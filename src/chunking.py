from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
import os
import pickle

import document_loader as loader

data_path = './Data'
meta_df = loader.load_metadata(f"{data_path}/data_list.csv")

def save_chunks(chunks, path="./Data/chunks.pkl"):
    with open(path, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks(path="./Data/chunks.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def merge_documents(docs):
    texts = []
    base_meta = docs[0].metadata if docs else {}

    for d in docs:
        page = d.metadata.get("page", None)
        if page:
            texts.append(f"[PAGE:{page}]\n{d.page_content}\n")
        else:
            texts.append(d.page_content)

    return "\n".join(texts), base_meta

def load_and_chunk_all_docs(data_path):
    text_splitter = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    all_chunks = []

    meta_df = loader.load_metadata(f"{data_path}/data_list.csv")

    for root, _, files in os.walk(data_path):
        for file in files:
            if not file.lower().endswith((".pdf", ".hwp")):
                continue
            
            path = os.path.join(root, file)

            try:
                docs = loader.load_document(path, meta_df)

                # PDF: 여러 Document → 단일 텍스트
                # HWP: Document 하나 → 단일 텍스트
                full_text, base_meta = merge_documents(docs)
                full_text = loader.clean_text_for_rag(full_text)

                # 실제 chunking
                split_texts = text_splitter.split_text(full_text)

                for idx, chunk in enumerate(split_texts):
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            **base_meta,
                            "chunk_index": idx,
                            "chunk_id": f"{file}_{idx}"
                        }
                    )
                    all_chunks.append(chunk_doc)

                print(f"✅ {file} → 청킹 {len(split_texts)}개")

            except Exception as e:
                print(f"⚠️ {file} 로드 실패 → {e}")

    print(f"총 청크 개수: {len(all_chunks)}")
    return all_chunks