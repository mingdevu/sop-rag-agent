"""테스트는 LLM API를 호출하지 않는다: get_llm을 목업으로 바꿔 끼운다 (비용·네트워크 의존 제거)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sop_rag import agent, api, llm, rag  # noqa: E402


@pytest.fixture()
def mock_llm(monkeypatch):
    handle = llm.LLMHandle(model=None, provider="mock")
    monkeypatch.setattr(rag, "get_llm", lambda: handle)
    monkeypatch.setattr(agent, "get_llm", lambda: handle)
    monkeypatch.setattr(api, "get_llm", lambda: handle)
    monkeypatch.setattr(agent, "get_graph", lambda: None)
    return handle
