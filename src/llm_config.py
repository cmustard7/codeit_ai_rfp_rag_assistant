import rag_logger as rl
from langchain_openai import ChatOpenAI

llm = classifier_llm = judge_llm = distill_llm = None

llm_dict = {}

def config_llm():
    global llm, classifier_llm, judge_llm, distill_llm, llm_dict

    llm = ChatOpenAI(model='gpt-5-mini', temperature=0.0)
    classifier_llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0)
    judge_llm = ChatOpenAI(model='gpt-5-mini', temperature=0.0)
    distill_llm = ChatOpenAI(model='gpt-5-mini', temperature=0.0)

    llm_dict = {'llm': llm, 'classifier': classifier_llm, 'judge': judge_llm, 'distill': distill_llm}

def get_llm(name: str):
    return llm_dict[name]

def log_usage(name, result):
    """토큰 사용량 + 비용 로깅"""
    usage = result.response_metadata["token_usage"]
    
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    total_tokens = usage["total_tokens"]
    
    # 🔥 모델별 정확한 가격
    model_pricing = {
        'llm': {"input": 0.00025, "output": 0.002},          # gpt-5-mini
        'judge': {"input": 0.00025, "output": 0.002},        # gpt-5-mini
        'distill': {"input": 0.00025, "output": 0.002},      # gpt-5-mini
        'classifier': {"input": 0.00005, "output": 0.0004},  # gpt-5-nano
    }
    
    pricing = model_pricing.get(name, {"input": 0.00025, "output": 0.002})
    
    # 비용 계산
    input_cost = (prompt_tokens / 1000) * pricing["input"]
    output_cost = (completion_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    rl.log({
        # 토큰 수
        f"{name}_prompt_t": prompt_tokens,
        f"{name}_completion_t": completion_tokens,
        f"{name}_total_t": total_tokens,
        
        # 비용 (USD) - 소수점 8자리까지
        f"{name}_input_cost_usd": round(input_cost, 8),
        f"{name}_output_cost_usd": round(output_cost, 8),
        f"{name}_total_cost_usd": round(total_cost, 8),
        
        # 비용 (KRW) - 소수점 4자리까지
        f"{name}_total_cost_krw": round(total_cost * 1300, 4)
    })