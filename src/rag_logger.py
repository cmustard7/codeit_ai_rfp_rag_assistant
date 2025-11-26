# RAGLogger.py
import wandb
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

class RAGLogger:
    """
    RAG 시스템 통합 로거
    - wandb.log()로 실시간 그래프 로깅
    - 동시에 데이터를 수집해서 나중에 Table로 변환
    """
    
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.current_context = {
            "question_id": None,
            "model": None,
            "engine": None,
            "timestamp": None
        }
        self._is_active = False
    
    def start_run(self, engine: str):
        """새로운 실험 시작"""
        self._is_active = True
        self.current_context["engine"] = engine
        print(f"[RAGLogger] 로깅 시작: {engine}")
    
    def set_question_context(self, question_id: int, question: str, model: str = None):
        """현재 처리 중인 질문 컨텍스트 설정"""
        self.current_context.update({
            "question_id": question_id,
            "question": question[:100],
            "model": model or self.current_context.get("engine", "unknown"),
            "timestamp": datetime.now().isoformat()
        })
        print(f"[RAGLogger] 질문 #{question_id} 설정: {model}")
    
    def log(self, metrics: Dict[str, Any], phase: str = None):
        """메트릭 로깅 (그래프 + Table 동시)"""
        if not self._is_active:
            wandb.log(metrics)
            return
        
        # 1. 실시간 그래프용 로그
        wandb.log(metrics)
        
        # 2. Table용 데이터 수집
        log_entry = {
            **self.current_context,
            "phase": phase,
            **metrics
        }
        self.logs.append(log_entry)
    
    def log_retrieval(self, metrics: Dict[str, Any]):
        """검색 단계 로깅"""
        self.log(metrics, phase="retrieval")
    
    def log_generation(self, metrics: Dict[str, Any]):
        """생성 단계 로깅"""
        self.log(metrics, phase="generation")
    
    def log_evaluation(self, metrics: Dict[str, Any]):
        """평가 단계 로깅"""
        self.log(metrics, phase="evaluation")
    
    def log_metadata(self, metrics: Dict[str, Any]):
        """메타데이터 추출 로깅"""
        self.log(metrics, phase="metadata")
    
    def log_prompt(self, metrics: Dict[str, Any]):
        """프롬프트 구성 로깅"""
        self.log(metrics, phase="prompt")
    
    def log_pipeline(self, metrics: Dict[str, Any]):
        """전체 파이프라인 로깅"""
        self.log(metrics, phase="pipeline")
    
    def log_routing(self, metrics: Dict[str, Any]):
        """라우팅 결정 로깅"""
        self.log(metrics, phase="routing")
    
    def create_summary_table(self) -> Optional[wandb.Table]:
        """질문별 최종 결과만 모은 요약 테이블"""
        if not self.logs:
            return None
        
        summary_logs = [
            log for log in self.logs 
            if log.get("phase") == "pipeline"
        ]
        
        if not summary_logs:
            return None
        
        df = pd.DataFrame(summary_logs)
        
        columns_to_keep = [
            "question_id", "model", "question",
            "pipeline/total_time_sec", "pipeline/final_score", 
            "pipeline/final_retry_count", "pipeline/success"
        ]
        
        existing_cols = [col for col in columns_to_keep if col in df.columns]
        df_summary = df[existing_cols]
        
        return wandb.Table(dataframe=df_summary)
    
    def create_detailed_table(self) -> Optional[wandb.Table]:
        """모든 로그를 담은 상세 테이블"""
        if not self.logs:
            return None
        
        df = pd.DataFrame(self.logs)
        return wandb.Table(dataframe=df)
    
    def create_phase_tables(self) -> Dict[str, wandb.Table]:
        """단계별로 분리된 테이블들"""
        if not self.logs:
            return {}
        
        df = pd.DataFrame(self.logs)
        tables = {}
        
        for phase in df["phase"].unique():
            if pd.isna(phase):
                continue
            
            phase_df = df[df["phase"] == phase]
            tables[phase] = wandb.Table(dataframe=phase_df)
        
        return tables
    
    def finalize(self, create_tables: bool = True):
        """로깅 종료 및 테이블 생성"""
        if not self._is_active:
            return
        
        print(f"\n[RAGLogger] 로깅 종료: 총 {len(self.logs)}개 로그 수집됨")
        
        if create_tables and self.logs:
            summary_table = self.create_summary_table()
            if summary_table:
                wandb.log({"results_summary": summary_table})
                print("  ✓ 요약 테이블 생성 완료")
            
            detailed_table = self.create_detailed_table()
            if detailed_table:
                wandb.log({"results_detailed": detailed_table})
                print("  ✓ 상세 테이블 생성 완료")
        
        self._is_active = False
    
    def reset(self):
        """로거 초기화"""
        self.logs = []
        self.current_context = {
            "question_id": None,
            "model": None,
            "engine": None,
            "timestamp": None
        }
        self._is_active = False
    
    def get_stats(self) -> Dict[str, Any]:
        """현재까지 수집된 로그 통계"""
        if not self.logs:
            return {"total_logs": 0}
        
        df = pd.DataFrame(self.logs)
        
        return {
            "total_logs": len(self.logs),
            "unique_questions": df["question_id"].nunique() if "question_id" in df else 0,
            "phases": df["phase"].value_counts().to_dict() if "phase" in df else {},
            "models": df["model"].unique().tolist() if "model" in df else []
        }


# ============================================
# 전역 싱글톤 인스턴스 & 헬퍼 함수들
# ============================================

_global_logger = RAGLogger()

# ✅ 이제 이렇게 바로 쓸 수 있음!
def log(metrics: Dict[str, Any], phase: str = None):
    """전역 로거로 바로 로깅"""
    _global_logger.log(metrics, phase)

def log_retrieval(metrics: Dict[str, Any]):
    """검색 로깅"""
    _global_logger.log_retrieval(metrics)

def log_generation(metrics: Dict[str, Any]):
    """생성 로깅"""
    _global_logger.log_generation(metrics)

def log_evaluation(metrics: Dict[str, Any]):
    """평가 로깅"""
    _global_logger.log_evaluation(metrics)

def log_metadata(metrics: Dict[str, Any]):
    """메타데이터 로깅"""
    _global_logger.log_metadata(metrics)

def log_prompt(metrics: Dict[str, Any]):
    """프롬프트 로깅"""
    _global_logger.log_prompt(metrics)

def log_pipeline(metrics: Dict[str, Any]):
    """파이프라인 로깅"""
    _global_logger.log_pipeline(metrics)

def log_routing(metrics: Dict[str, Any]):
    """라우팅 로깅"""
    _global_logger.log_routing(metrics)

def start_run(engine: str):
    """실험 시작"""
    _global_logger.start_run(engine)

def set_question_context(question_id: int, question: str, model: str = None):
    """질문 컨텍스트 설정"""
    _global_logger.set_question_context(question_id, question, model)

def finalize(create_tables: bool = True):
    """로깅 종료"""
    _global_logger.finalize(create_tables)

def reset():
    """로거 초기화"""
    _global_logger.reset()

def get_stats():
    """통계 조회"""
    return _global_logger.get_stats()

# ✅ 하위 호환성을 위해
def get_logger() -> RAGLogger:
    """전역 로거 인스턴스 반환 (하위 호환용)"""
    return _global_logger