import streamlit as st
from main import initialize_system, run_rag_single_question, finalize_system

st.title("RAG Engine Test")

# 세션 state
if "active" not in st.session_state:
    st.session_state.active = False
    st.session_state.count = 0
    st.session_state.results = []

engine = st.selectbox("Engine", ["langchain", "langgraph_base", "langgraph_multisearch"])
use_distill = st.checkbox("Distillation", value=False)

# 시작/종료 버튼
col1, col2 = st.columns(2)

with col1:
    if not st.session_state.active:
        if st.button("🚀 실험 시작"):
            initialize_system(engine)  # ✅ 한 번만
            st.session_state.active = True
            st.session_state.count = 0
            st.session_state.results = []
            st.success(f"{engine} 실험 시작!")

with col2:
    if st.session_state.active:
        if st.button("🏁 실험 종료"):
            finalize_system()  # ✅ 한 번만
            st.session_state.active = False
            st.success(f"종료! 총 {st.session_state.count}개 질문")

# 질문 입력 (활성화 상태에서만)
if st.session_state.active:
    st.write(f"### 질문 {st.session_state.count + 1}")
    question = st.text_input("질문:", key=f"q_{st.session_state.count}")
    
    if st.button("실행"):
        if question:
            with st.spinner("처리 중..."):
                result = run_rag_single_question(
                    engine=engine,
                    question=question,
                    question_id=st.session_state.count,  # ✅ 0, 1, 2, 3, 4...
                    use_distill=use_distill
                )
                
                st.success("완료!")
                st.write(result)
                
                st.session_state.results.append((question, result))
                st.session_state.count += 1
        else:
            st.warning("질문 입력하세요")
else:
    st.info("먼저 '실험 시작' 버튼을 누르세요")