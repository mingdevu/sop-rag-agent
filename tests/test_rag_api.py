from fastapi.testclient import TestClient

from sop_rag.rag import answer_question, extract_citations, validate_citations
from sop_rag.retriever import get_retriever


def test_extract_citations_dedup_and_parse():
    answer = "70%로 낮춘다 [SOP-003 §2. 즉시 조치]. 다시 [SOP-003 §2. 즉시 조치] 그리고 [ERR-001 §1. 온도 계열 (E1xx)]"
    assert extract_citations(answer) == [("SOP-003", "2. 즉시 조치"), ("ERR-001", "1. 온도 계열 (E1xx)")]


def test_validate_citations_flags_made_up_source():
    results = get_retriever().search("HDF 즉시 조치", k=3)
    shown = (results[0]["chunk"].doc_id, results[0]["chunk"].section)
    check = validate_citations([shown, ("SOP-999", "없는 섹션")], results)
    assert check["valid"] == [shown] and check["invalid"] == [("SOP-999", "없는 섹션")]


def test_answer_question_mock_mode_cites_top_chunk(mock_llm):
    result = answer_question("냉각수 농도 기준은?")
    assert result["provider"] == "mock"
    assert result["citations"] and not result["citation_check"]["invalid"]


def test_api_endpoints(mock_llm):
    from sop_rag.api import app
    client = TestClient(app)
    assert client.get("/health").json()["chunks"] > 50
    assert client.post("/ask", json={"question": "LOTO 해제 순서"}).status_code == 200
    assert client.post("/ask", json={"question": ""}).status_code == 422
    agent_result = client.post("/agent", json={"question": "3번 설비 상태 알려줘"}).json()
    assert agent_result["tool_trace"][0]["tool"] == "get_equipment_status"


def test_daily_limiter_blocks_after_limit(tmp_path):
    from sop_rag.limits import DailyLimiter
    limiter = DailyLimiter(limit=2, state_path=tmp_path / "usage.json")
    assert limiter.try_acquire() and limiter.try_acquire()
    assert limiter.try_acquire() is False
    assert limiter.usage()["remaining"] == 0
    # 새 인스턴스(재시작)도 같은 날이면 이어서 센다
    assert DailyLimiter(limit=2, state_path=tmp_path / "usage.json").try_acquire() is False
    assert DailyLimiter(limit=0, state_path=tmp_path / "x.json").try_acquire() is True


def test_api_returns_429_when_limit_exceeded(mock_llm, tmp_path, monkeypatch):
    from sop_rag import api
    from sop_rag.limits import DailyLimiter
    monkeypatch.setattr(api, "limiter", DailyLimiter(limit=1, state_path=tmp_path / "u.json"))
    client = TestClient(api.app)
    assert client.post("/ask", json={"question": "냉각수 농도"}).status_code == 200
    blocked = client.post("/agent", json={"question": "3번 설비"})
    assert blocked.status_code == 429 and "한도" in blocked.json()["detail"]
    assert client.get("/usage").json()["used"] == 1
