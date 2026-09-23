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


FACTS = [
    ("france", "What is the capital of France?", "Paris is the capital of France.", "Paris is the capital", "of France.", "Bananas grow on Mars."),
    ("cern", "When was CERN founded?", "CERN was founded in 1961.", "CERN was founded", "in 1961.", "CERN was founded in 1492."),
    ("museum", "When did the museum on Broad Street open?", "The museum opened in 1884 on Broad Street.", "The museum opened in", "1884 on Broad Street.", "The museum opened in 2020 on King Street."),
    ("library", "When did the Oak Street library open?", "The Oak Street library opened in 1998.", "The Oak Street library opened", "in 1998.", "The Oak Street library opened in 1801."),
    ("bridge", "How long is the Harbor Bridge?", "The Harbor Bridge is 900 meters long.", "The Harbor Bridge is", "900 meters long.", "The Harbor Bridge is 12 meters long."),
    ("treaty", "Where was the treaty signed?", "The treaty was signed in Geneva.", "The treaty was signed", "in Geneva.", "The treaty was signed in Cairo."),
    ("prize", "Who won the 2014 city prize?", "Mina Cole won the 2014 city prize.", "Mina Cole won", "the 2014 city prize.", "Jonah Hale won the 2014 city prize."),
    ("dose", "What dose does the label recommend?", "The label recommends a dose of 50 mg.", "The label recommends", "a dose of 50 mg.", "The label recommends a dose of 500 mg."),
    ("flight", "What time does flight 214 depart?", "Flight 214 departs at 18:40.", "Flight 214 departs", "at 18:40.", "Flight 214 departs at 06:05."),
    ("population", "What was the town population in 2020?", "The town population in 2020 was 12000.", "The town population in 2020", "was 12000.", "The town population in 2020 was 900000."),
    ("alloy", "What is the melting point of the alloy?", "The alloy melts at 640 degrees.", "The alloy melts", "at 640 degrees.", "The alloy melts at 20 degrees."),
    ("park", "How many acres is Cedar Park?", "Cedar Park covers 80 acres.", "Cedar Park covers", "80 acres.", "Cedar Park covers 8000 acres."),
]


def _fillers_named(prefix: str, n: int) -> List[Dict[str, str]]:
    return [{"id": f"{prefix}-w{i}", "text": f"Unrelated weather note number {i} about rainfall."} for i in range(n)]


def build_expanded_cases() -> List[Dict[str, Any]]:
    """One known cause per construction. Retrieval subtypes are scored together."""
    cases = []
    for name, question, answer, left, right, false_answer in FACTS:
        full = {"id": f"{name}-full", "text": answer}
        fillers = _fillers_named(name, 8)
        cases.append(
            {
                "name": f"{name}-supported",
                "true_cause": "supported",
                "question": question,
                "answer": answer,
                "retrieved_chunks": [full] + fillers[:2],
                "corpus_chunks": [full] + fillers[:2],
            }
        )
        cases.append(
            {
                "name": f"{name}-hallucination",
                "true_cause": HALLUCINATION,
                "question": question,
                "answer": false_answer,
                "retrieved_chunks": [full],
                "corpus_chunks": [full],
            }
        )
        cases.append(
            {
                "name": f"{name}-chunking_miss",
                "true_cause": CHUNKING_MISS,
                "question": question,
                "answer": answer,
                "retrieved_chunks": [
                    {"id": f"{name}-left", "text": left},
                    {"id": f"{name}-right", "text": right},
                ],
                "corpus_chunks": [
                    {"id": f"{name}-left", "text": left},
                    {"id": f"{name}-right", "text": right},
                ],
            }
        )
        cases.append(
            {
                "name": f"{name}-retrieval_miss",
                "true_cause": RETRIEVAL_MISS,
                "question": question,
                "answer": answer,
                "retrieved_chunks": fillers[:5],
                "corpus_chunks": fillers + [full],
            }
        )
    return cases


def _family(label: str) -> str:
    if label in {K_TOO_SMALL, EMBEDDING_RETRIEVAL_FAILURE, RETRIEVAL_MISS}:
        return RETRIEVAL_MISS
    return label


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
    family_correct = 0
    for case in cases:
        report = analyzer.analyze(
            case["question"],
            case["answer"],
            case["retrieved_chunks"],
            case.get("corpus_chunks"),
        )
        predicted_component = report.root_cause or ""
        predicted_fault = report.sentences[0].fault if report.sentences else ""
        predicted = predicted_component or predicted_fault
        ok = _matches(case["true_cause"], predicted_fault, predicted_component)
        family_ok = _family(case["true_cause"]) == _family(predicted)
        correct += int(ok)
        family_correct += int(family_ok)
        rows.append(
            {
                "name": case["name"],
                "true_cause": case["true_cause"],
                "predicted": predicted,
                "correct": ok,
                "family_correct": family_ok,
            }
        )
    total = len(cases)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "family_correct": family_correct,
        "family_accuracy": round(family_correct / total, 3) if total else 0.0,
        "cases": rows,
    }


def main() -> None:
    import argparse
    import json
    import os

    from rag.datasets.ragtruth import make_analyzer

    parser = argparse.ArgumentParser(description="Score planted RAG faults with a known cause")
    parser.add_argument("--set", choices=("small", "expanded"), default="expanded")
    args = parser.parse_args()
    cases = planted_cases() if args.set == "small" else build_expanded_cases()
    judge = os.getenv("RAG_DEBUGGER_BENCH_JUDGE", "local")
    report = evaluate_analyzer(make_analyzer(judge), cases)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
