"""마크다운 문서 로딩 → 섹션 단위 청킹.

SOP·매뉴얼은 '## 섹션' 하나가 하나의 절차/기준을 담는 구조라서, 글자 수로 자르기보다
섹션 경계로 자르는 편이 의미가 덜 끊기고 인용(문서명 §섹션)도 정확해진다.
섹션이 너무 길면 줄 단위로 다시 나눈다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHUNK_CHARS = 900


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    source_file: str
    flags: list[str] = field(default_factory=list)

    def citation(self) -> str:
        return f"[{self.doc_id} §{self.section}]"

    def embedding_text(self) -> str:
        # 문서 제목·섹션명을 본문 앞에 붙여 검색 단서를 늘린다
        return f"{self.doc_id} {self.title} > {self.section}\n{self.text}"


def parse_front_matter(raw_text: str) -> tuple[dict, str]:
    """'---'로 감싼 머리말(key: value)을 dict로 분리한다."""
    match = re.match(r"^---\n(.*?)\n---\n", raw_text, flags=re.S)
    if not match:
        return {}, raw_text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, raw_text[match.end():]


def split_long_text(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > limit and current:
            parts.append(current)
            current = ""
        current += line
    if current:
        parts.append(current)
    return parts


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id = meta.get("doc_id", path.stem)
        title = meta.get("title", path.stem)
        # '## ' 헤더 기준으로 섹션 분리 (첫 '# ' 제목 줄은 버린다)
        sections = re.split(r"^## +(.+)$", body, flags=re.M)
        for position in range(1, len(sections), 2):
            section_name = sections[position].strip()
            section_text = sections[position + 1].strip()
            for part_index, part in enumerate(split_long_text(section_text)):
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}#{(position + 1) // 2}-{part_index}",
                    doc_id=doc_id, title=title, section=section_name,
                    text=part.strip(), source_file=path.name,
                ))
    return chunks
