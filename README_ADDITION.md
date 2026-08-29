
---

## 📊 Retrieval Latency Benchmark

To validate the retrieval-latency claims for this project, `benchmark.py` (in this repo)
compares two retrieval strategies over transcript chunks generated with this project's
own chunking config (`chunk_size=500`, `chunk_overlap=50`):

- **Naive keyword search** — a linear scan that term-frequency-scores every chunk against
  the query and returns the top-k matches (a stand-in for "no vector index at all").
- **FAISS vector retrieval** — the same chunks indexed in FAISS and queried via vector
  similarity search.

### Results (30 trials per corpus size, averaged)

| Chunks | Keyword search (mean ms) | FAISS search (mean ms) | Latency reduction | One-time index build (ms) |
|---|---|---|---|---|
| 500 | 1.43 | 0.39 | 72.8% | 18.7 |
| 2,000 | 6.11 | 0.51 | 91.7% | 65.1 |
| 5,000 | 14.86 | 0.61 | 95.9% | 152.4 |
| 10,000 | 27.91 | 0.94 | 96.6% | 365.0 |
| 20,000 | 61.11 | 1.57 | 97.4% | 669.7 |

**At the ~10,000-chunk scale this project targets, FAISS retrieval reduced mean query
latency by ~96.6% versus a naive keyword scan.**

### Methodology notes (important for interpreting these numbers)

- Both methods are timed for **query-time cost only** (the cost paid per question asked).
  Index build time is a one-time cost per video and is reported separately.
- The "keyword search" baseline is a naive linear Python scan, not a tuned system like
  BM25/Elasticsearch — a fair comparison against "no vector index," but a tuned keyword
  system would likely close part of this gap.
- Vectors are TF-IDF (384-dim, matching this project's embedding size) rather than a
  hosted transformer embedding, since the benchmark environment couldn't download model
  weights. This affects retrieval *quality*, not the *latency characteristics* being
  measured here — FAISS query latency scales with vector dimensionality, index type, and
  corpus size, not with what produced the vectors.
- Reproduce with: `python benchmark.py`

