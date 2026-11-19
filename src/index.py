"""Hybrid (keyword + embedding) search with permission filtering.

The store is deliberately tiny and in-memory so it runs anywhere, with no
database and no Docker. The public surface (build / search / save / load) is
small on purpose: swapping in Postgres + pgvector later means reimplementing
this one class, not touching the rest of the pipeline.
"""
from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_NUM_COMMA = re.compile(r"(?<=\d),(?=\d)")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(_NUM_COMMA.sub("", text.lower()))


# ---------------------------------------------------------------- embedders
class SentenceTransformerEmbedder:
    """Local embedding model (runs on CPU or GPU). Downloads weights on first use."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True,
                                            batch_size=32, show_progress_bar=False))

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(self.model.encode(self.QUERY_PREFIX + query, normalize_embeddings=True))


class HashEmbedder:
    """Dependency-free bag-of-words hashing embedder. For tests and offline smoke runs only."""

    def __init__(self, dim: int = 512):
        self.name = f"hash-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, query: str) -> np.ndarray:
        return self._vec(query)


# -------------------------------------------------------------------- index
@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float
    dense: float | None


class HybridIndex:
    def __init__(self, embedder=None, rrf_k: int = 60):
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self.emb: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None

    # -- building -------------------------------------------------------
    def build(self, chunks: list[Chunk]) -> "HybridIndex":
        self.chunks = list(chunks)
        texts = [c.index_text for c in self.chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.emb = self.embedder.encode_docs(texts) if self.embedder else None
        return self

    # -- permissions ----------------------------------------------------
    def _visible(self, user_groups: list[str] | None) -> list[int]:
        allowed = set(user_groups or []) | {"all"}
        return [i for i, c in enumerate(self.chunks) if allowed & set(c.acl)]

    # -- searching ------------------------------------------------------
    def search(self, query: str, k: int = 4, user_groups: list[str] | None = None,
               pool: int = 30) -> list[Hit]:
        """Return the top-k chunks the given user is allowed to see.

        Permission filtering happens BEFORE ranking, so restricted text can
        never reach the model or leak through scores.
        """
        visible = self._visible(user_groups)
        if not visible:
            return []
        vis = np.asarray(visible)

        bm25_all = np.asarray(self.bm25.get_scores(tokenize(query)))
        bm25_vis = bm25_all[vis]
        dense_vis = None
        if self.embedder is not None and self.emb is not None:
            dense_vis = self.emb[vis] @ self.embedder.encode_query(query)

        fused: dict[int, float] = {}
        order_bm25 = np.argsort(-bm25_vis)[:pool]
        for rank, j in enumerate(order_bm25):
            if bm25_vis[j] > 0:
                fused[j] = fused.get(j, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        if dense_vis is not None:
            for rank, j in enumerate(np.argsort(-dense_vis)[:pool]):
                fused[j] = fused.get(j, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            Hit(chunk=self.chunks[visible[j]], score=float(s), bm25=float(bm25_vis[j]),
                dense=float(dense_vis[j]) if dense_vis is not None else None)
            for j, s in top
        ]

    # -- persistence ----------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks]))
        if self.emb is not None:
            np.save(d / "embeddings.npy", self.emb)
        (d / "meta.json").write_text(json.dumps(
            {"embedder": getattr(self.embedder, "name", None), "rrf_k": self.rrf_k}))

    @classmethod
    def load(cls, directory: str | Path, embedder=None) -> "HybridIndex":
        d = Path(directory)
        idx = cls(embedder=embedder)
        idx.chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text())]
        idx.bm25 = BM25Okapi([tokenize(c.index_text) for c in idx.chunks])
        emb_path = d / "embeddings.npy"
        idx.emb = np.load(emb_path) if emb_path.exists() else None
        return idx
