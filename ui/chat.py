"""간단한 채팅 UI. 실행: streamlit run ui/chat.py  (API: uvicorn sop_rag.api:app --port 8100)"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("SOP_API_URL", "http://127.0.0.1:8100")

st.set_page_config(page_title="SOP RAG 에이전트", layout="wide")
st.title("제조 SOP·정비 매뉴얼 도우미")
st.caption("문서는 모두 합성(가상) 문서입니다 · 답변의 [문서ID §섹션]은 근거 출처")

mode = st.sidebar.radio("모드", ["에이전트 (설비 상태 + SOP)", "문서 질의응답 (RAG)"])
st.sidebar.markdown("예시 질문\n- 3번 설비 이상 점수가 높은데 어떻게 조치해?\n- 에러코드 E302는 뭐야?\n- 냉각수 농도 기준은?")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("details"):
            with st.expander("근거·도구 호출 보기"):
                st.json(turn["details"])

question = st.chat_input("질문을 입력하세요")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    endpoint = "/agent" if mode.startswith("에이전트") else "/ask"
    with st.chat_message("assistant"):
        with st.spinner("검색·생성 중..."):
            try:
                response = httpx.post(f"{API_URL}{endpoint}", json={"question": question}, timeout=180)
                response.raise_for_status()
                result = response.json()
                answer = result["answer"]
                details = {key: result.get(key) for key in
                           ("provider", "latency_ms", "citations", "tool_trace", "contexts",
                            "used_mock_equipment_data", "citation_check") if key in result}
            except httpx.HTTPError as error:
                answer, details = f"API 호출 실패: {error}", None
        if (details or {}).get("used_mock_equipment_data"):
            st.warning("설비 상태는 목업 데이터입니다 (factory-pdm 미실행).")
        st.markdown(answer)
        if details:
            with st.expander("근거·도구 호출 보기"):
                st.json(details)
    st.session_state.history.append({"role": "assistant", "content": answer, "details": details})
