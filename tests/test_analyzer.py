from rag.analyzer import (
    CHUNKING_MISS,
    HALLUCINATION,
    RETRIEVAL_MISS,
    SUPPORTED,
    UNSUPPORTED,
    TraceAnalyzer,
)
from rag.judge import JudgeResult, _parse_judge_output


class FakeEmbedder:
    keys = ["paris", "france", "capital", "banana", "founded", "1961", "mars"]

    def embed_text(self, text: str):
        lowered = text.lower()
        return [float(lowered.count(key)) for key in self.keys]

    def embed_batch(self, texts):
        return [self.embed_text(text) for text in texts]


class FakeJudge:
    def judge(self, claim: str, evidence: str, question: str = "") -> JudgeResult:
        claim_l = claim.lower()
        evidence_l = evidence.lower()

        if "mars" in claim_l or "banana" in claim_l:
            return JudgeResult("unsupported", "not in evidence")

        if "paris" in claim_l or "capital of france" in claim_l:
            if "paris" in evidence_l and "capital" in evidence_l and "france" in evidence_l:
                return JudgeResult("supported", "entailed by evidence")
            return JudgeResult("unsupported", "paris fact missing")

        if "founded in 1961" in claim_l:
            if "founded" in evidence_l and "1961" in evidence_l:
                return JudgeResult("supported", "both halves present")
            if "founded" in evidence_l or "1961" in evidence_l:
                return JudgeResult("partial", "only one half")
            return JudgeResult("unsupported", "cern fact missing")

        return JudgeResult("unsupported", "default")


def _analyzer():
    return TraceAnalyzer(FakeEmbedder(), FakeJudge())


def test_supported_when_retrieved_entails_claim():
    report = _analyzer().analyze(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=[{"id": "r1", "text": "Paris is the capital of France."}],
        corpus_chunks=[{"id": "r1", "text": "Paris is the capital of France."}],
    )
    assert report.sentences[0].fault == SUPPORTED
    assert report.summary[SUPPORTED] == 1


def test_retrieval_miss_when_corpus_has_evidence_but_retriever_did_not():
    report = _analyzer().analyze(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=[{"id": "r1", "text": "Bananas are a yellow fruit."}],
        corpus_chunks=[
            {"id": "r1", "text": "Bananas are a yellow fruit."},
            {"id": "c9", "text": "Paris is the capital of France."},
        ],
    )
    assert report.sentences[0].fault == RETRIEVAL_MISS
    assert "c9" in report.sentences[0].supporting_chunk_ids


def test_hallucination_when_neither_retrieved_nor_corpus_support_claim():
    report = _analyzer().analyze(
        question="What is the capital of France?",
        answer="Bananas grow on Mars.",
        retrieved_chunks=[{"id": "r1", "text": "Paris is the capital of France."}],
        corpus_chunks=[{"id": "r1", "text": "Paris is the capital of France."}],
    )
    assert report.sentences[0].fault == HALLUCINATION


def test_chunking_miss_when_only_concatenated_chunks_entail_claim():
    report = _analyzer().analyze(
        question="When was CERN founded?",
        answer="CERN was founded in 1961.",
        retrieved_chunks=[
            {"id": "r1", "text": "CERN was founded"},
            {"id": "r2", "text": "in 1961 near Geneva."},
        ],
        corpus_chunks=[
            {"id": "r1", "text": "CERN was founded"},
            {"id": "r2", "text": "in 1961 near Geneva."},
        ],
    )
    assert report.sentences[0].fault == CHUNKING_MISS


def test_unsupported_without_corpus_cannot_split_miss_and_hallucination():
    report = _analyzer().analyze(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_chunks=[{"id": "r1", "text": "Bananas are a yellow fruit."}],
    )
    assert report.sentences[0].fault == UNSUPPORTED
    assert any("corpus_chunks" in note for note in report.notes)


def test_parse_judge_json_and_fenced_output():
    parsed = _parse_judge_output('{"label":"supported","reason":"yes"}')
    assert parsed.label == "supported"
    fenced = _parse_judge_output('```json\n{"label": "partial", "reason": "half"}\n```')
    assert fenced.label == "partial"
    bad = _parse_judge_output("not json")
    assert bad.label == "unsupported"
