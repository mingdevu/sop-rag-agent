"""하이브리드 검색: BM25(키워드) + 벡터(의미) → RRF로 순위 결합.

- BM25는 '에러코드 E302', '8.6 K'처럼 정확한 토큰 일치에 강하고,
- 벡터 검색은 '열이 안 빠져'처럼 표현이 달라도 의미가 같은 질의에 강하다.
- RRF(Reciprocal Rank Fusion)는 점수 스케일이 다른 두 검색기를 '순위'로만 합쳐서 정규화 고민이 없다.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from sop_rag.config import DOCS_DIR, EMBEDDING_MODEL, INDEX_DIR
from sop_rag.guard import sanitize
from sop_rag.loader import Chunk, load_chunks

RRF_K = 60


def tokenize(text: str) -> list[str]:
    """한국어는 조사가 붙어 공백 토큰이 잘 안 맞으므로, 단어 + 글자 2-gram을 함께 쓴다.

    예) '방열판을' → ['방열판을', '방열', '열판', '판을']
    형태소 분석기(설치 부담) 없이도 '방열판'과 '방열판을'이 겹치게 된다.
    """
    tokens = []
    for word in re.findall(r"[0-9A-Za-z.\-]+|[가-힣]+", text.lower()):
        tokens.append(word)
        if re.match(r"[가-힣]", word) and len(word) > 2:
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
    return tokens


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def embed(texts: list[str], kind: str) -> np.ndarray:
    # e5 계열은 학습 때 쓴 접두어(query:/passage:)를 붙여야 성능이 나온다
    prefixed = [f"{kind}: {text}" for text in texts]
    vectors = get_embedder().encode(prefixed, normalize_embeddings=True, batch_size=32)
    return np.asarray(vectors, dtype="float32")


def build_index(docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR) -> dict:
    chunks = load_chunks(docs_dir)
    for chunk in chunks:
        chunk.text, redacted = sanitize(chunk.text)
        if redacted:
            chunk.flags.append("injection_redacted")
    vectors = embed([chunk.embedding_text() for chunk in chunks], kind="passage")
    # 정규화된 벡터의 내적 = 코사인 유사도
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "vectors.faiss"))
    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=1), encoding="utf-8")
    return {"chunks": len(chunks), "docs": len({chunk.doc_id for chunk in chunks}),
            "redacted_chunks": sum(1 for chunk in chunks if chunk.flags)}


class HybridRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        if not (index_dir / "vectors.faiss").exists():
            build_index(index_dir=index_dir)
        self.chunks = [Chunk(**raw) for raw in
                       json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
        self.vector_index = faiss.read_index(str(index_dir / "vectors.faiss"))
        self.bm25 = BM25Okapi([tokenize(chunk.embedding_text()) for chunk in self.chunks])
        self.by_doc = {}
        for chunk in self.chunks:
            self.by_doc.setdefault(chunk.doc_id, []).append(chunk)

    def bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores) if scores[i] > 0]

    def vector_ranking(self, query: str, depth: int = 30) -> list[int]:
        _, indices = self.vector_index.search(embed([query], kind="query"), depth)
        return [int(i) for i in indices[0] if i >= 0]

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
        """mode: 'bm25' | 'vector' | 'hybrid'. 반환: [{chunk, score}] (점수 높은 순)"""
        rankings = []
        if mode in ("bm25", "hybrid"):
            rankings.append(self.bm25_ranking(query))
        if mode in ("vector", "hybrid"):
            rankings.append(self.vector_ranking(query))
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, chunk_index in enumerate(ranking[:30]):
                fused[chunk_index] = fused.get(chunk_index, 0.0) + 1.0 / (RRF_K + rank + 1)
        ordered = sorted(fused.items(), key=lambda item: -item[1])[:k]
        return [{"chunk": self.chunks[i], "score": round(score, 5)} for i, score in ordered]

    def has_section(self, doc_id: str, section: str) -> bool:
        return any(chunk.section == section for chunk in self.by_doc.get(doc_id, []))


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


if __name__ == "__main__":
    print(build_index())
