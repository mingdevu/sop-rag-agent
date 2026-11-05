"""하루 요청 수 제한 (공개 데모에서 API 키 비용이 무한정 나가지 않게 막는다).

- 사용자별이 아니라 '서비스 전체' 기준으로 센다. 목적이 비용 상한이기 때문이다.
- 날짜는 한국 시간(KST) 기준으로 바뀐다.
- 카운트는 JSON 파일에 저장해서 프로세스가 재시작돼도 같은 날이면 이어서 센다.
  (단, 무료 호스팅의 디스크가 초기화되면 0부터 다시 센다 — README에 명시)
- limit이 0이면 제한하지 않는다 (로컬 개발 기본값).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))


class DailyLimiter:
    def __init__(self, limit: int, state_path: Path):
        self.limit = limit
        self.state_path = Path(state_path)
        self._lock = threading.Lock()  # 동시에 들어온 요청이 같은 숫자를 읽지 않도록

    @staticmethod
    def today() -> str:
        return datetime.now(KST).strftime("%Y-%m-%d")

    def _load(self) -> dict:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            state = {}
        if state.get("date") != self.today():
            state = {"date": self.today(), "count": 0}  # 날짜가 바뀌면 초기화
        return state

    def _save(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state), encoding="utf-8")

    def try_acquire(self) -> bool:
        """한도 안이면 1회 사용으로 기록하고 True, 한도를 넘으면 기록 없이 False."""
        if self.limit <= 0:
            return True
        with self._lock:
            state = self._load()
            if state["count"] >= self.limit:
                return False
            state["count"] += 1
            self._save(state)
            return True

    def usage(self) -> dict:
        with self._lock:
            state = self._load()
        return {"date_kst": state["date"], "used": state["count"],
                "limit": self.limit if self.limit > 0 else None,
                "remaining": max(self.limit - state["count"], 0) if self.limit > 0 else None}
