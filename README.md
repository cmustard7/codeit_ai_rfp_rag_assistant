# LangGraph RAG Starter

이 프로젝트는 LangGraph 기반으로 RAG 파이프라인을 빠르게 구성하기 위한 최소 템플릿입니다. 모든 스크립트는 `src/` 디렉터리에 정리되어 있으며, 문서 파싱(HWP 포함)부터 벡터스토어 구축, LangGraph 워크플로 실행, GPT-5 기반 평가까지 한 번에 수행할 수 있습니다.

## 구성
```
LangGraph_rag/
├── README.md
├── requirements.txt
├── data/
│   ├── data_list.csv|xlsx   # 기관/사업 메타데이터
│   ├── files/               # 원본 문서(HWP/PDF/DOCX 등)
│   ├── questions.json       # 평가용 질문(자동 생성)
│   ├── results.json         # run_eval 결과
│   └── eval/                # judge 평가 결과 등
└── src/
    ├── build_vectorstore.py # data/files → vectorstore.json 생성
    ├── data_loader.py       # data_list.* + 문서 본문 로딩
    ├── document_parser.py   # TXT/PDF/DOCX/HWP 파서
    ├── evaluate_llm_judge.py # GPT-5 기반 자동 평가 스크립트
    ├── generate_questions.py # 메타데이터 기반 질문 생성
    ├── graph_state.py       # LangGraph 상태 정의
    ├── nodes/               # LangGraph 노드(question/retrieve/answer/update)
    ├── run_chat.py          # 대화형 테스트 CLI
    ├── run_eval.py          # 질문 세트 일괄 평가
    ├── text_chunker.py      # 청킹 유틸
    ├── vector_store.py      # 임베딩 생성/검색 헬퍼
    └── workflow.py          # LangGraph DAG 정의
```

## 데이터 준비
- `data/data_list.csv` 또는 `data/data_list.xlsx` : 기관/사업 메타데이터. 열 이름은 자유롭게 사용하되, 아래 키 중 하나와 매칭되면 자동으로 인식됩니다.
  - 기관 : `발주 기관`, `발주기관`, `agency`
  - 사업명 : `사업명`, `project_name`, `title`
  - 요약 : `사업 요약`, `요약`, `summary`
- `data/files/` : 실제 문서 디렉터리. HWP/PDF/DOCX/TXT 등을 지원하며, 없으면 `data_list`의 `텍스트` 열을 사용합니다.

## 빠른 시작
1. 가상환경 생성 및 패키지 설치
   ```bash
   python -m venv .venv
   .venv/Scripts/activate   # Windows
   pip install -r requirements.txt
   ```
2. `.env` 에 OpenAI API Key 설정
   ```bash
   echo OPENAI_API_KEY=sk-... > .env
   ```
3. 질문 생성
   ```bash
   python -m src.generate_questions
   # (--limit, --follow-up 옵션으로 샘플 수/후속 질문 여부 조정 가능)
   ```
4. 벡터스토어 구축 (최초 1회 혹은 데이터 변경 시)
   ```bash
   python -m src.build_vectorstore
   ```
5. LangGraph 워크플로 실행
   ```bash
   python -m src.run_eval
   # (--questions, --output 옵션으로 경로 변경 가능)
   ```
6. 대화형 테스트 (선택)
   ```bash
   python -m src.run_chat
   # 'exit' 입력 시 종료, --reset 옵션으로 매 질문마다 상태 초기화 가능
   ```
7. GPT-5 Judge 평가 (선택)
   ```bash
   python -m src.evaluate_llm_judge --results data/results.json --output data/eval/judge_scores.json
   # --model gpt-5-nano 처럼 Judge 모델을 지정할 수 있습니다.
   ```
8. Retrieval 정량 평가 (선택)
   ```bash
   python -m src.evaluate_retrieval --results data/results.json --gold data/gold_targets.json --output data/eval/retrieval_scores.json
   ```

### 주요 CLI 옵션
- `generate_questions`
  - `--limit <int>`: data_list 상단에서 몇 개의 행을 사용할지 지정 (기본 100).
  - `--follow-up`: 각 사업에 대해 후속 질문 추가 생성 여부(플래그).
  - `--csv/--xlsx/--output <path>`: 입력/출력 경로 지정.
  - `--shuffle`: 데이터를 무작위로 섞은 뒤 limit 만큼 추출.
  - `--fake-rate <float>`: 0~1 값, 문서에 존재하지 않을 가능성이 높은 질문을 섞어 생성.
  - `--compare-rate <float>`: 0~1 값, 서로 다른 두 문서를 비교하는 질문을 추가 생성.
- `run_eval`
  - `--questions <path>`: 기본 `data/questions.json` 대신 다른 질문 파일 사용.
  - `--output <path>`: 결과 JSON 저장 위치 지정.
- `run_chat`
  - `--reset`: 매 질문마다 상태 초기화(기본은 세션 상태 유지).
- `build_vectorstore`
  - `--output <path>`: 생성된 벡터스토어 JSON 위치 지정.
- `evaluate_llm_judge`
  - `--results <path>`: 평가할 run_eval 결과 경로 지정.
  - `--output <path>`: Judge 결과 저장 경로 지정.
  - `--limit <int>`: 평가 샘플 수 제한.
  - `--model <name>`: 사용할 GPT-5 계열 Judge 모델 명 (기본 `gpt-5-mini`).
- `evaluate_retrieval`
  - `--results <path>`: run_eval 결과 JSON (context/retrieved_docs 포함).
  - `--gold <path>`: 정답 매핑 JSON.
  - `--output <path>`: 평가 지표 저장 경로 지정.

#### 정답 매핑 JSON 예시 (`data/gold_targets.json`)
```json
{
  "gold": [
    {
      "id": "Q01",
      "expected_files": [
        "국민연금공단_이러닝_사업.hwp",
        "국민연금공단_이러닝_사업.pdf"
      ],
      "keywords": [
        "콘텐츠 개발",
        "학습관리"
      ]
    }
  ]
}
```

## TODO
- Retrieval 단계 정량 평가(Recall@K 등) 도입
- Judge 프롬프트 고도화 및 다중 기준 점수화
- CI 파이프라인에 자동 평가 통합
