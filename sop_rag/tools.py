"""에이전트 도구 3종: search_sop · get_equipment_status · get_error_code."""
from __future__ import annotations

import re
from functools import lru_cache

import httpx
from langchain_core.tools import tool

from sop_rag.config import DOCS_DIR, PDM_INGEST_URL, PDM_SERVE_URL
from sop_rag.retriever import get_retriever

SENSOR_FIELDS = ["product_type", "air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min"]

# factory-pdm이 꺼져 있을 때 쓰는 목업 상태. 응답에 mock=True를 넣어 실제 값과 구분한다.
MOCK_READINGS = {
    3: {"product_type": "L", "air_temp_k": 301.2, "process_temp_k": 310.4, "rpm": 1180.0,
        "torque_nm": 68.5, "tool_wear_min": 215.0},
}
MOCK_DEFAULT = {"product_type": "M", "air_temp_k": 300.1, "process_temp_k": 310.2, "rpm": 1520.0,
                "torque_nm": 39.8, "tool_wear_min": 60.0}


def search_sop_impl(query: str, k: int = 4) -> list[dict]:
    return [{"doc_id": r["chunk"].doc_id, "title": r["chunk"].title,
             "section": r["chunk"].section, "text": r["chunk"].text,
             "cite_as": r["chunk"].citation()}
            for r in get_retriever().search(query, k=k)]


def equipment_status_impl(machine_id: int) -> dict:
    """factory-pdm의 수집 API에서 최신 센서값을 받아 /predict·/anomaly로 진단한다."""
    try:
        with httpx.Client(timeout=3.0) as client:
            latest = client.get(f"{PDM_INGEST_URL}/machines/{machine_id}/latest")
            latest.raise_for_status()
            reading = {key: latest.json()[key] for key in SENSOR_FIELDS}
            payload = {**reading, "machine_id": machine_id}
            prediction = client.post(f"{PDM_SERVE_URL}/predict", json=payload).raise_for_status().json()
            anomaly = client.post(f"{PDM_SERVE_URL}/anomaly", json=payload).raise_for_status().json()
        return {"machine_id": machine_id, "mock": False, "source": "factory-pdm (실시간)",
                "received_at": latest.json().get("received_at"), "reading": reading,
                "prediction": prediction, "anomaly": anomaly}
    except (httpx.HTTPError, KeyError) as error:
        return mock_equipment_status(machine_id, reason=str(error))


def mock_equipment_status(machine_id: int, reason: str) -> dict:
    reading = MOCK_READINGS.get(machine_id, MOCK_DEFAULT)
    abnormal = machine_id in MOCK_READINGS
    return {
        "machine_id": machine_id, "mock": True,
        "source": f"목업 응답 (factory-pdm 연결 실패: {reason[:120]})",
        "reading": reading,
        "prediction": {"failure_probability": 0.81 if abnormal else 0.02,
                       "predicted_failure": abnormal,
                       "failure_type": "OSF" if abnormal else "NONE"},
        "anomaly": {"anomaly_score": 0.93 if abnormal else 0.12, "is_anomaly": abnormal},
    }


@lru_cache(maxsize=1)
def error_code_table() -> dict[str, dict]:
    """ERR-001 마크다운 표를 파싱해 {코드: 행} 사전으로 만든다."""
    text = (DOCS_DIR / "ERR-001_설비_에러코드표.md").read_text(encoding="utf-8")
    table, section = {}, ""
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        match = re.match(r"^\|\s*(E\d{3})\s*\|(.+)\|\s*$", line)
        if match:
            name, meaning, action, related = [cell.strip() for cell in match.group(2).split("|")]
            table[match.group(1)] = {"code": match.group(1), "name": name, "meaning": meaning,
                                     "first_action": action, "related_doc": related,
                                     "cite_as": f"[ERR-001 §{section}]"}
    return table


def error_code_impl(code: str) -> dict:
    normalized = code.strip().upper()
    row = error_code_table().get(normalized)
    if row is None:
        return {"code": normalized, "found": False, "message": "ERR-001 에러코드표에 없는 코드입니다."}
    return {**row, "found": True}


@tool
def search_sop(query: str) -> list[dict]:
    """설비 SOP·정비 매뉴얼·정책 문서를 하이브리드 검색한다. 조치 절차나 기준을 찾을 때 쓴다.
    결과의 cite_as 값을 답변의 출처 표기에 그대로 사용한다."""
    return search_sop_impl(query)


@tool
def get_equipment_status(machine_id: int) -> dict:
    """설비 번호(1~5)의 최신 센서값, 고장 확률·예측 고장 유형, 이상 점수(0~1)를 조회한다.
    결과의 mock 값이 true이면 실제 데이터가 아닌 목업 응답이다."""
    return equipment_status_impl(machine_id)


@tool
def get_error_code(code: str) -> dict:
    """설비 에러코드(예: E302)의 의미, 1차 조치, 관련 문서를 ERR-001 표에서 찾는다."""
    return error_code_impl(code)


ALL_TOOLS = [search_sop, get_equipment_status, get_error_code]
