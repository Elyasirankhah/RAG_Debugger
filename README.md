# RAG Debugger

**v0.2 — root-cause debugging.** Give it one failed RAG trace. It tells you which claims failed, where the correct evidence was, which stage likely failed, and one thing to test next.

```bash
rag-debugger analyze trace.json
rag-debugger compare before.json after.json
```

No web page required.

## Trace

Every adapter and the API consume the same schema:

```json
{
  "question": "...",
  "answer": "...",
  "retrieved_chunks": [{"id": "r1", "text": "..."}],
  "corpus_chunks": [{"id": "c47", "text": "..."}],
  "metadata": {"retriever": "faiss", "top_k": 5, "model": "gpt-4o-mini"}
}
```

```python
from rag_debugger.integrations.langchain import trace
from rag.diagnose import diagnose

report = diagnose(trace(query, langchain_result, corpus=nodes, retriever="faiss", top_k=5))
print(report.format())
```

Each claim:

```json
{
  "claim": "...",
  "label": "retrieval_miss",
  "confidence": 0.8,
  "best_retrieved_chunk": "r3",
  "best_corpus_chunk": "c47",
  "evidence": "...",
  "reason": "Relevant evidence exists in corpus but was not retrieved."
}
```

Then a pipeline summary: how many claims are supported, how many misses, the primary suspected stage, and one next test. `confidence` is a heuristic, not a calibrated probability.

## Install

```bash
pip install git+https://github.com/Elyasirankhah/RAG_Debugger.git
$env:OPENAI_API_KEY="your-key"   # Windows
rag-debugger analyze trace.json
```

`rag-debugger` with no subcommand still starts the HTTP API (`POST /diagnose`). `rag-debugger serve` does the same.

## Benchmarks

We do **not** yet have comparison numbers against public leaderboards. Those sets exist and are the right external baselines, but they are not downloaded or scored in this repo:

| Set | Use it for |
|---|---|
| RAGTruth | supported vs hallucination |
| RAGChecker | diagnostic quality baseline |
| RAGBench | broader domain validation |
| RGB / CRAG | hard retrieval and conflicts |

None of them label the full root-cause taxonomy (`retrieval_miss` vs `chunking_miss` vs planted rank). That is the paper contribution, and it is **not done**. In-repo today there is only a small planted-fault unit set, not a 500–2,000 example benchmark and not a results table.

## What comes after v0.2

Only after `analyze` is reliable on real traces:

1. More adapters (LlamaIndex, OpenAI, OpenInference) on this same `Trace`
2. Repair experiments (`compare` on traces the tool re-runs)
3. The editor view (sentence colors, click a claim, see missed evidence)
4. The fault-injection benchmark large enough to publish

The `/upload` and `/ask` demo is frozen. It is not the product.

## License

MIT.
