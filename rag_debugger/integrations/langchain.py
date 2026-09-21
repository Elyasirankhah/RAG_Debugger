"""LangChain adapter. Produces the stable Trace schema."""
from typing import Any, Optional

from rag.trace import Trace
from rag_debugger.adapters import from_langchain


def trace(
    query: str,
    result: Any,
    corpus: Optional[list] = None,
    retriever: Optional[str] = None,
    top_k: Optional[int] = None,
    model: Optional[str] = None,
) -> Trace:
    data = from_langchain(query, result, corpus=corpus)
    data["metadata"] = {
        "retriever": retriever,
        "top_k": top_k,
        "model": model,
    }
    return Trace.from_dict(data)
