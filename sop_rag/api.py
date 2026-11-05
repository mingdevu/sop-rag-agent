"""RAG·에이전트 API.

실행: uvicorn sop_rag.api:app --host 127.0.0.1 --port 8100
  POST /ask    {"question": "...", "k": 5}   문서 질의응답 (출처 인용)
  POST /agent  {"question": "..."}           설비 상태 조회 + SOP 검색 에이전트
  GET  /health
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from sop_rag.agent import run_agent
from sop_rag.llm import get_llm
from sop_rag.rag import answer_question
from sop_rag.retriever import get_retriever

app = FastAPI(title="sop-rag-agent", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=10)


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


@app.get("/health")
def health() -> dict:
    retriever = get_retriever()
    return {"status": "ok", "chunks": len(retriever.chunks), "llm_provider": get_llm().provider}


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return answer_question(request.question, k=request.k)


@app.post("/agent")
def agent(request: AgentRequest) -> dict:
    return run_agent(request.question)
