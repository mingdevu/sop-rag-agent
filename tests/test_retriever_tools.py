from sop_rag import tools
from sop_rag.retriever import get_retriever, tokenize


def test_tokenize_adds_korean_bigrams():
    assert tokenize("방열판을") == ["방열판을", "방열", "열판", "판을"]


def test_bm25_finds_error_code_section():
    results = get_retriever().search("E302 과부하 누적", k=3, mode="bm25")
    assert ("ERR-001", "3. 기계·부하 계열 (E3xx)") in [(r["chunk"].doc_id, r["chunk"].section) for r in results]


def test_hybrid_semantic_query():
    results = get_retriever().search("열이 잘 안 빠질 때 판단 기준", k=5, mode="hybrid")
    assert "SOP-003" in [r["chunk"].doc_id for r in results]


def test_indexed_notice_is_sanitized():
    notice = [c for c in get_retriever().chunks if c.doc_id == "NOTICE-001" and c.flags]
    assert notice and "무시하라" not in notice[0].text


def test_error_code_lookup():
    row = tools.error_code_impl("e302")
    assert row["found"] and row["related_doc"] == "SOP-005" and row["cite_as"].startswith("[ERR-001")
    assert tools.error_code_impl("E999")["found"] is False


def test_equipment_status_falls_back_to_mock(monkeypatch):
    monkeypatch.setattr(tools, "PDM_INGEST_URL", "http://127.0.0.1:9")  # 닫힌 포트
    status = tools.equipment_status_impl(3)
    assert status["mock"] is True and status["anomaly"]["anomaly_score"] > 0.9
    assert "목업" in status["source"]
