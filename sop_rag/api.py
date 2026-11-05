"""RAG·에이전트 API.

실행: uvicorn sop_rag.api:app --host 127.0.0.1 --port 8100
  POST /ask    {"question": "...", "k": 5}   문서 질의응답 (출처 인용)
  POST /agent  {"question": "..."}           설비 상태 조회 + SOP 검색 에이전트
  GET  /usage                               오늘 사용량 / 하루 한도
  GET  /health

환경변수 DAILY_REQUEST_LIMIT(기본 0 = 제한 없음)를 주면 /ask·/agent 호출 수를
서비스 전체 기준 하루(KST) 한도로 막는다. 한도를 넘으면 429를 돌려준다.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sop_rag.agent import run_agent
from sop_rag.config import PROJECT_ROOT
from sop_rag.limits import DailyLimiter
from sop_rag.llm import get_llm
from sop_rag.rag import answer_question
from sop_rag.retriever import get_retriever


class UTF8JSONResponse(JSONResponse):
    """Content-Type에 charset을 명시한다. 없으면 PowerShell 5.1 등 일부 클라이언트가 한글을 Latin-1로 읽어 깨진다."""
    media_type = "application/json; charset=utf-8"


app = FastAPI(title="sop-rag-agent", version="0.1.0", default_response_class=UTF8JSONResponse)
limiter = DailyLimiter(
    limit=int(os.getenv("DAILY_REQUEST_LIMIT", "0")),
    state_path=os.getenv("USAGE_STATE_PATH", PROJECT_ROOT / "index_store" / "usage.json"),
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=10)


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


def check_daily_limit() -> None:
    if not limiter.try_acquire():
        raise HTTPException(
            status_code=429,
            detail=f"오늘 데모 요청 한도({limiter.limit}회)를 모두 썼습니다. 한국 시간 자정 이후 다시 시도해 주세요.",
        )


@app.get("/health")
def health() -> dict:
    retriever = get_retriever()
    return {"status": "ok", "chunks": len(retriever.chunks), "llm_provider": get_llm().provider}


@app.get("/usage")
def usage() -> dict:
    return limiter.usage()


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    check_daily_limit()
    return answer_question(request.question, k=request.k)


@app.post("/agent")
def agent(request: AgentRequest) -> dict:
    check_daily_limit()
    return run_agent(request.question)
