import time
import rag_logger as rl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

from metadata import extract_metadata
from langchain.retrieval import find_docs_by_question
from prompt_template import prompt
from llm_config import get_llm, log_usage

def run_chain(question, vector_store, retriever):
    """포트폴리오급 wandb 로깅이 포함된 RAG chain"""
    
    # ============================================
    # 1. 실행 시작 시간 기록
    # ============================================
    start_time = time.time()
    llm = get_llm('llm')
    
    # ============================================
    # 2. 문서 검색 + 로깅
    # ============================================
    def retrieve_and_log(question_input):
        retrieve_start = time.time()
        
        docs = find_docs_by_question(
            question_input["question"], 
            vector_store, 
            retriever
        )
        
        retrieve_time = time.time() - retrieve_start
        
        # 검색 메트릭 로깅
        rl.log_retrieval({
            "retrieval/time_sec": retrieve_time,
            "retrieval/num_docs": len(docs),
            "retrieval/has_results": len(docs) > 0
        })
        
        # 문서별 상세 정보 (선택적)
        for i, doc in enumerate(docs):
            if hasattr(doc, 'metadata'):
                meta = doc.metadata
                rl.log({
                    f"retrieval/doc_{i}_source": meta.get('source', 'unknown'),
                    f"retrieval/doc_{i}_length": len(doc.page_content)
                })
        
        return {"docs": docs}
    
    # ============================================
    # 3. 메타데이터 추출 + 로깅
    # ============================================
    def extract_and_log_metadata(docs_dict):
        metadata = extract_metadata(docs_dict["docs"])
        
        # 메타데이터 로깅
        rl.log_metadata({
            "metadata/context_length": len(metadata.get('context', '')),
            "metadata/num_sources": len(metadata.get('source', '').split(' / ')),
            "metadata/has_budget": bool(metadata.get('budget', '미기재') != '미기재'),
            "metadata/has_dates": bool(metadata.get('open_date', '미기재') != '미기재')
        })
        
        return metadata
    
    # ============================================
    # 4. 프롬프트 구성 + 로깅
    # ============================================
    def build_and_log_prompt(inputs):
        metadata = inputs["metadata"]
        question_text = inputs["question"]
        
        formatted_prompt = prompt.format(
            context=metadata['context'],
            question=question_text,
            source=metadata['source'],
            org=metadata['org'],
            category=metadata['category'],
            budget=metadata['budget'],
            open_date=metadata['open_date'],
            end_date=metadata['end_date']
        )
        
        # 프롬프트 통계
        rl.log_prompt({
            "prompt/total_length": len(formatted_prompt),
            "prompt/question_length": len(question_text),
            "prompt/context_ratio": len(metadata['context']) / len(formatted_prompt) if len(formatted_prompt) > 0 else 0
        })
        
        return formatted_prompt
    
    # ============================================
    # 5. LLM 생성 + 로깅
    # ============================================
    def generate_and_log(prompt_text):
        generation_start = time.time()
        
        raw = llm.invoke(prompt_text)
        answer = raw.content if hasattr(raw, "content") else str(raw)
        
        generation_time = time.time() - generation_start
        
        # 생성 메트릭
        log_usage("main", raw)  # 토큰 사용량
        
        rl.log_generation({
            "generation/time_sec": generation_time,
            "generation/answer_length": len(answer),
            "generation/answer_word_count": len(answer.split())
        })
        
        return answer
    
    # ============================================
    # 6. Chain 구성
    # ============================================
    metadata_chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(retrieve_and_log)
        | RunnableLambda(extract_and_log_metadata)
    )

    rag_chain = (
        {
            "metadata": metadata_chain,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_and_log_prompt)
        | RunnableLambda(generate_and_log)
        | StrOutputParser()
    )
    
    # ============================================
    # 7. 실행 + 최종 로깅
    # ============================================
    result = rag_chain.invoke({"question": question})
    
    total_time = time.time() - start_time
    
    # 전체 파이프라인 메트릭
    rl.log_pipeline({
        "pipeline/total_time_sec": total_time,
        "pipeline/question": question,
        "pipeline/answer_preview": result[:100] + "..." if len(result) > 100 else result,
        "pipeline/success": True
    })
    
    return result