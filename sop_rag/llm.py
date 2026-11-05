"""LLM 선택: Anthropic(Claude) → 로컬 Ollama → 목업 순으로 대체한다.

- 1순위: langchain-anthropic의 ChatAnthropic (.env의 LLM_MODEL, 기본 claude-haiku-5-5)
- 2순위: 키가 없거나 호출이 실패하면 로컬 Ollama
- 3순위: 둘 다 안 되면 model=None(목업). rag/agent가 LLM 없이 규칙 기반 응답을 만든다 (품질 평가에 쓰지 않는다)
선택 결과(provider)는 응답에 함께 실어 보내서 어떤 모델이 답했는지 숨기지 않는다.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from sop_rag.config import LLM_MODEL, OLLAMA_MODEL

logger = logging.getLogger("llm")


@dataclass
class LLMHandle:
    model: BaseChatModel | None
    provider: str  # "anthropic:<model>" | "ollama:<model>" | "mock"


def _anthropic() -> BaseChatModel | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    from langchain_anthropic import ChatAnthropic
    # Haiku 5.5는 temperature 등 샘플링 파라미터에 기본값이 아닌 값을 보내면 400을 반환하므로 지정하지 않는다.
    model = ChatAnthropic(model=LLM_MODEL, max_tokens=4096, timeout=60, max_retries=2)
    model.invoke("ping")  # 키·모델명이 유효한지 시작 시 1회 확인
    return model


def _ollama() -> BaseChatModel | None:
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        return None
    model = ChatOllama(model=OLLAMA_MODEL, temperature=0)
    model.invoke("ping")
    return model


@lru_cache(maxsize=1)
def get_llm() -> LLMHandle:
    for name, factory in (("anthropic", _anthropic), ("ollama", _ollama)):
        try:
            model = factory()
        except Exception as error:  # 대체 경로로 넘어가되 이유는 로그에 남긴다
            logger.warning("%s 사용 불가: %s", name, error)
            continue
        if model is not None:
            label = LLM_MODEL if name == "anthropic" else OLLAMA_MODEL
            return LLMHandle(model=model, provider=f"{name}:{label}")
    logger.warning("Anthropic·Ollama 모두 사용 불가 → 목업(LLM 없음) 모드")
    return LLMHandle(model=None, provider="mock")


def message_text(message) -> str:
    """AIMessage의 content가 문자열이든 블록 리스트(thinking/text 등)든 텍스트만 꺼낸다."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content
                   if isinstance(block, dict) and block.get("type") == "text")
