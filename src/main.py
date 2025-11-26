import wandb
import os
import pathlib
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv

import rag_logger as rl
from rag_engines import ENGINES
from chunking import load_chunks, save_chunks, load_and_chunk_all_docs
from vectorstore import build_vector_store
from llm_config import config_llm

CHUNK_PATH = "./Data/chunks.pkl"
FILES_PATH = "./Data"

# ============================================
# ✅ 전역 변수 추가 (세션 상태 유지)
# ============================================
_retriever = None
_vector_store = None
_experiment_active = False

def load_env():    
    load_dotenv()

def config_root_path():
    p = pathlib.Path().resolve()

    while True:
        if (p / ".git").exists():
            os.chdir(p)
            print("프로젝트 루트:", p)
            break

        if p.parent == p:
            raise RuntimeError("프로젝트 루트를 찾지 못함.")

        p = p.parent

def check_exist_chunk_vs():
    if os.path.exists(CHUNK_PATH):
        print("기존 chunks.pkl 발견 — 로드합니다.")
        chunks = load_chunks(CHUNK_PATH)
    else:
        print("기존 chunks.pkl 없음 — 새로 생성합니다.")
        chunks = load_and_chunk_all_docs(FILES_PATH)
        save_chunks(chunks, CHUNK_PATH)
        print("새 chunks.pkl 저장 완료")

    vector_store = build_vector_store(chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    print("청크 & 벡터스토어 준비완료")

    return retriever, vector_store

MODEL_PRICING = {
    "gpt-5-mini": {
        "input": 0.00025,
        "output": 0.002
    },
    "gpt-5-nano": {
        "input": 0.00005,
        "output": 0.0004
    }
}

def wandb_log(engine: str):
    wandb.init(
        project="rfp-rag",
        name=f"{engine}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "engine": engine,
            "retriever_type": "Chroma + CrossEncoder",
            "llm_model": "gpt-5-mini",
            "judge_model": "gpt-5-mini",
            "classifier_model": "gpt-5-nano",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 5,
            "temperature": 0.0,
            "max_retries": 5,
            "pricing": {
                "gpt-5-mini": {
                    "input_per_1k": 0.00025,
                    "output_per_1k": 0.002,
                    "input_per_1m": 0.25,
                    "output_per_1m": 2.0
                },
                "gpt-5-nano": {
                    "input_per_1k": 0.00005,
                    "output_per_1k": 0.0004,
                    "input_per_1m": 0.05,
                    "output_per_1m": 0.4
                }
            },
            "usd_to_krw": 1460.86
        },
        tags=[engine, "production", "rfp-analysis"]
    )

# ============================================
# ✅ 새 함수 1: 실험 초기화 (한 번만 호출)
# ============================================
def initialize_system(engine: str):
    """시스템 초기화 + WandB run 시작"""
    global _retriever, _vector_store, _experiment_active
    
    load_env()
    config_llm()
    config_root_path()
    
    # Retriever 준비
    _retriever, _vector_store = check_exist_chunk_vs()
    
    # WandB 시작
    wandb_log(engine)
    
    # RAGLogger 시작
    rl.start_run(engine)
    
    _experiment_active = True
    print(f"[SYSTEM] 실험 시작: {engine}")


# ============================================
# ✅ 새 함수 2: 질문 하나 실행 (여러 번 호출 가능)
# ============================================
def run_rag_single_question(engine: str, question: str, question_id: int, use_distill: bool = False):
    """질문 하나만 실행 (run은 계속 유지)"""
    global _retriever, _vector_store, _experiment_active
    
    if not _experiment_active:
        raise RuntimeError("initialize_system()을 먼저 호출하세요")
    
    if engine not in ENGINES:
        raise ValueError(f"지원하지 않는 엔진: {engine}")
    
    # 질문 context 설정
    rl.set_question_context(question_id, question, engine)
    
    print(f"\n[질문 #{question_id+1}] {question[:50]}...")
    
    # 실행
    if engine == "langgraph_multisearch":
        result = ENGINES[engine](
            question=question,
            retriever=_retriever,
            vector_store=_vector_store,
            use_distill=use_distill
        )
    else:
        result = ENGINES[engine](
            question=question,
            retriever=_retriever,
            vector_store=_vector_store
        )
    
    return result


# ============================================
# ✅ 새 함수 3: 실험 종료 (한 번만 호출)
# ============================================
def finalize_system():
    """실험 종료 + Table 생성"""
    global _experiment_active
    
    if not _experiment_active:
        print("[SYSTEM] 이미 종료됨")
        return
    
    # RAGLogger 종료 (Table 생성)
    rl.finalize(create_tables=True)
    
    # WandB 종료
    wandb.finish()
    
    _experiment_active = False
    print("[SYSTEM] 실험 종료")


# ============================================
# ✅ 기존 run_rag 함수는 단일 질문용으로 유지
# ============================================
def run_rag(engine: str, question: str, use_distill: bool=False):
    """단일 질문 실행 (CLI용)"""
    initialize_system(engine)
    
    try:
        result = run_rag_single_question(engine, question, 0, use_distill)
        return result
    finally:
        finalize_system()


def main():
    is_on = False
    print('1. Langchain')
    print('2. LangGraph_Base')
    print('3. LangGraph_MultiSearch')
    select = input('실행할 모드를 선택하십시오: ')

    while is_on:
        if int(select) == 1:
            run_rag("langchain", "질문을 입력하세요")
            is_on = True
        elif int(select) == 2:
            run_rag("langgraph_base", "질문을 입력하세요")
            is_on = True
        elif int(select) == 3:
            run_rag("langgraph_multisearch", "질문을 입력하세요")
            is_on = True
        else:
            print('잘못된 입력입니다.')
            continue

def __main__():
    main()