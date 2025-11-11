"""
RAG 시스템 설정 파일
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 청킹 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 임베딩 설정
EMBEDDING_MODEL = "text-embedding-3-small"

# Vector DB 설정
COLLECTION_NAME = "rfp_documents"
TOP_K = 5

# LLM 설정
LLM_MODEL = "gpt-4o-mini"  # GPT-4o-mini (권장) 또는 "gpt-4o", "gpt-3.5-turbo"
# 참고: GPT-5 모델은 아직 공개되지 않음 (2025.11 기준)
# GPT-4o 계열 사용 권장
TEMPERATURE = 0.3  # GPT-4o는 temperature 지원
MAX_TOKENS = 500

# Google Drive 설정 (선택)
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
