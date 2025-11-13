from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

import src.document_loader as loader

data_path = './Data'
meta_df = loader.load_metadata(f"{data_path}/data_list.csv")

def load_and_chunk_all_docs(data_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )

    all_splits = []

    for root, _, files in os.walk(data_path):
        for file in files:
            path = os.path.join(root, file)
            try:
                docs = loader.load_document(path, meta_df)  # ✅ 각 문서별로 list[Document] 반환
                splits = text_splitter.split_documents(docs)  # ✅ 문서별 청킹
                all_splits.extend(splits)

                print(f"✅ {file} 청킹 완료 ({len(splits)}개)")
            except Exception as e:
                print(f"⚠️ {file} 로드 실패 → {e}")

    print(f"총 청크 개수: {len(all_splits)}")
    return all_splits