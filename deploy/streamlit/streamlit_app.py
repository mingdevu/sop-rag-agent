"""sop-rag-agent 단독 실행 데모 (Streamlit Community Cloud용).

무료 호스팅은 프로세스 하나만 띄울 수 있어서, FastAPI를 거치지 않고 RAG·에이전트 함수를 직접 부른다.
비밀값은 Streamlit secrets에서 읽는다 (코드·레포에는 키가 없다):
    ANTHROPIC_API_KEY = "..."
    LLM_MODEL = "claude-haiku-5-5"      # 선택
    DAILY_REQUEST_LIMIT = 100           # 선택, 서비스 전체 하루 한도 (기본 100)

실행: streamlit run deploy/streamlit/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DAILY_LIMIT = 100


def load_secrets_into_env() -> None:
    """secrets 값을 환경변수로 옮긴다. sop_rag가 import될 때 설정을 읽으므로 import보다 먼저 해야 한다."""
    try:
        secrets = dict(st.secrets)
    except Exception:  # secrets 파일이 없으면(로컬 실행) .env와 환경변수를 그대로 쓴다
        secrets = {}
    for key in ("ANTHROPIC_API_KEY", "LLM_MODEL", "DAILY_REQUEST_LIMIT"):
        if key in secrets and not os.getenv(key):
            os.environ[key] = str(secrets[key])


load_secrets_into_env()

from sop_rag.agent import run_agent  # noqa: E402
from sop_rag.limits import DailyLimiter  # noqa: E402
from sop_rag.llm import get_llm  # noqa: E402
from sop_rag.rag import answer_question  # noqa: E402
from sop_rag.retriever import get_retriever  # noqa: E402

st.set_page_config(page_title="SOP RAG 에이전트 데모", layout="wide")


@st.cache_resource
def get_limiter() -> DailyLimiter:
    # cache_resource라서 모든 접속자가 같은 카운터를 공유한다 (= 서비스 전체 한도)
    limit = int(os.getenv("DAILY_REQUEST_LIMIT", DEFAULT_DAILY_LIMIT))
    return DailyLimiter(limit=limit, state_path=Path("/tmp") / "sop_rag_usage.json")


@st.cache_resource(show_spinner="문서 색인과 임베딩 모델 준비 중 (첫 실행만, 1~2분)...")
def warm_up() -> str:
    get_retriever()
    return get_llm().provider


st.title("제조 SOP·정비 매뉴얼 도우미 (데모)")
st.caption("문서는 모두 **합성(가상) 문서**입니다 · 답변의 [문서ID §섹션]은 근거 출처 · "
           "소스: [github.com/mingdevu/sop-rag-agent](https://github.com/mingdevu/sop-rag-agent)")

provider = warm_up()
limiter = get_limiter()

mode = st.sidebar.radio("모드", ["에이전트 (설비 상태 + SOP)", "문서 질의응답 (RAG)"])
st.sidebar.markdown("**예시 질문**\n- 3번 설비 이상 점수가 높은데 어떻게 조치해?\n"
                    "- 설비에서 E302 에러가 떴어. 뭘 해야 해?\n- 냉각수 농도 기준은?\n"
                    "- MX-500 구매 가격은? (문서에 없음 → '모름')")
usage = limiter.usage()
if usage["limit"]:
    st.sidebar.caption(f"오늘 데모 사용량: {usage['used']} / {usage['limit']}회 (한국 시간 자정에 초기화, 모든 방문자 합산)")
st.sidebar.caption(f"LLM: `{provider}`")
st.sidebar.info("이 데모에서는 예지보전 서비스(factory-pdm)와 연결되지 않아, 에이전트의 설비 상태 조회는 "
                "**목업 데이터**로 응답합니다. 실시간 연동은 로컬 Docker 실행에서 확인할 수 있습니다.")
if provider == "mock":
    st.warning("LLM API에 연결하지 못해 목업 모드(검색 원문만 표시)로 동작 중입니다.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("details"):
            with st.expander("근거·도구 호출 보기"):
                st.json(turn["details"])

question = st.chat_input("질문을 입력하세요 (최대 500자)", max_chars=500)
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        details = None
        if not limiter.try_acquire():
            answer = (f"⚠️ 오늘 데모 요청 한도({limiter.limit}회)를 모두 썼습니다. "
                      "한국 시간 자정 이후 다시 시도해 주세요.")
        else:
            with st.spinner("검색·생성 중..."):
                try:
                    result = run_agent(question) if mode.startswith("에이전트") else answer_question(question)
                    answer = result["answer"]
                    details = {key: result.get(key) for key in
                               ("provider", "latency_ms", "citations", "tool_trace", "contexts",
                                "used_mock_equipment_data", "citation_check") if key in result}
                except Exception as error:  # 화면이 깨지지 않게 하되, 오류는 숨기지 않고 보여 준다
                    answer = f"처리 중 오류가 발생했습니다: {type(error).__name__}: {error}"
        if (details or {}).get("used_mock_equipment_data"):
            st.warning("설비 상태는 목업 데이터입니다 (실제 설비 값 아님).")
        st.markdown(answer)
        if details:
            with st.expander("근거·도구 호출 보기"):
                st.json(details)
    st.session_state.history.append({"role": "assistant", "content": answer, "details": details})
