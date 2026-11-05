"""설정값을 한 곳에서 읽는다. (.env → 환경변수 → 기본값)"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DOCS_DIR = PROJECT_ROOT / "docs_corpus"
INDEX_DIR = PROJECT_ROOT / "index_store"

# 다국어(한국어 포함) 지원 경량 임베딩 모델: 약 118M 파라미터, 384차원
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

LLM_MODEL = os.getenv("LLM_MODEL") or "claude-haiku-5-5"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:latest")

PDM_INGEST_URL = os.getenv("PDM_INGEST_URL", "http://127.0.0.1:8001")
PDM_SERVE_URL = os.getenv("PDM_SERVE_URL", "http://127.0.0.1:8000")
