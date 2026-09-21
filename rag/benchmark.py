"""Controlled RAG fault injection. The planted cause is known."""
from __future__ import annotations

from typing import Any, Dict, List

from rag.analyzer import (
    CHUNKING_MISS,
    EMBEDDING_RETRIEVAL_FAILURE,
    HALLUCINATION,
    K_TOO_SMALL,
    RETRIEVAL_MISS,
    TraceAnalyzer,
)

QUESTION = "What is the capital of France?"
GOLD = "Paris is the capital of France."
ANSWER = "Paris is the capital of France."
HALLUCINATED = "Bananas grow on Mars."


def _fillers(n: int) -> List[Dict[str, str]]:
    return [{"id": f"d{i}", "text": f"Unrelated weather note number {i}."} for i in range(n)]


def planted_cases() -> List[Dict[str, Any]]:
    gold = {"id": "gold", "text": GOLD}
    return [
        {
            "name": "k_too_small",
            "true_cause": K_TOO_SMALL,
            "question": QUESTION,
            "answer": ANSWER,
            "retrieved_chunks": _fillers(5),
            "corpus_chunks": _fillers(5) + [gold] + _fillers(3),
        },
        {
            "name": "chunking_miss",
            "true_cause": CHUNKING_MISS,
            "question": "When was CERN founded?",
            "answer": "CERN was founded in 1961.",
            "retrieved_chunks": [
                {"id": "r1", "text": "CERN was founded"},
                {"id": "r2", "text": "in 1961 near Geneva."},
            ],
            "corpus_chunks": [
                {"id": "r1", "text": "CERN was founded"},
                {"id": "r2", "text": "in 1961 near Geneva."},
            ],
        },
        {
            "name": "hallucination",
            "true_cause": HALLUCINATION,
            "question": QUESTION,
            "answer": HALLUCINATED,
            "retrieved_chunks": [gold],
            "corpus_chunks": [gold],
        },
        {
            "name": "embedding_retrieval_failure",
            "true_cause": EMBEDDING_RETRIEVAL_FAILURE,
            "question": QUESTION,
            "answer": ANSWER,
            "retrieved_chunks": _fillers(5),
            "corpus_chunks": _fillers(40) + [gold],
        },
    ]


def _matches(true_cause: str, predicted_fault: str, predicted_component: str) -> bool:
    predicted = predicted_component or predicted_fault
    if true_cause == predicted:
        return True
    if true_cause in {K_TOO_SMALL, EMBEDDING_RETRIEVAL_FAILURE} and predicted_fault == RETRIEVAL_MISS:
        return predicted == true_cause
    return False


def evaluate_analyzer(analyzer: TraceAnalyzer, cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    cases = cases or planted_cases()
    rows = []
    correct = 0
    for case in cases:
        report = analyzer.analyze(
            case["question"],
            case["answer"],
            case["retrieved_chunks"],
            case.get("corpus_chunks"),
        )
        predicted_component = report.root_cause or ""
        predicted_fault = report.sentences[0].fault if report.sentences else ""
        ok = _matches(case["true_cause"], predicted_fault, predicted_component)
        correct += int(ok)
        rows.append(
            {
                "name": case["name"],
                "true_cause": case["true_cause"],
                "predicted": predicted_component,
                "correct": ok,
            }
        )
    return {
        "total": len(cases),
        "correct": correct,
        "accuracy": round(correct / len(cases), 3) if cases else 0.0,
        "cases": rows,
    }
