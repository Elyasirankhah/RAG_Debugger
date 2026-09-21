"""Stable RAG trace. Every adapter, API, and CLI command consumes this."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {"id": self.id, "text": self.text}
        if self.doc_id:
            data["doc_id"] = self.doc_id
        return data


@dataclass
class TraceMetadata:
    retriever: Optional[str] = None
    top_k: Optional[int] = None
    model: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.extra)
        if self.retriever:
            data["retriever"] = self.retriever
        if self.top_k is not None:
            data["top_k"] = self.top_k
        if self.model:
            data["model"] = self.model
        return data


@dataclass
class Trace:
    question: str
    answer: str
    retrieved_chunks: List[Chunk] = field(default_factory=list)
    corpus_chunks: List[Chunk] = field(default_factory=list)
    metadata: TraceMetadata = field(default_factory=TraceMetadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieved_chunks": [chunk.to_dict() for chunk in self.retrieved_chunks],
            "corpus_chunks": [chunk.to_dict() for chunk in self.corpus_chunks],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Trace":
        meta_raw = raw.get("metadata") or {}
        known = {"retriever", "top_k", "model"}
        metadata = TraceMetadata(
            retriever=meta_raw.get("retriever"),
            top_k=meta_raw.get("top_k"),
            model=meta_raw.get("model"),
            extra={key: value for key, value in meta_raw.items() if key not in known},
        )
        corpus = raw.get("corpus_chunks") or []
        return cls(
            question=str(raw.get("question") or ""),
            answer=str(raw.get("answer") or ""),
            retrieved_chunks=[_chunk(item, i) for i, item in enumerate(raw.get("retrieved_chunks") or [])],
            corpus_chunks=[_chunk(item, i) for i, item in enumerate(corpus)],
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Trace":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Trace file must be a JSON object")
        return cls.from_dict(data)


def _chunk(raw: Any, index: int) -> Chunk:
    if not isinstance(raw, dict):
        return Chunk(id=f"chunk_{index}", text=str(raw))
    return Chunk(
        id=str(raw.get("id") or raw.get("chunk_id") or raw.get("doc_id") or f"chunk_{index}"),
        text=str(raw.get("text") or raw.get("page_content") or raw.get("content") or ""),
        doc_id=raw.get("doc_id"),
    )
