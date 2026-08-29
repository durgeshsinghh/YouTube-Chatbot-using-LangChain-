#!/usr/bin/env python3
"""
Benchmark: naive keyword search vs. FAISS vector retrieval latency.

Goal: produce a REAL, reproducible latency number to back up a resume claim,
using the same chunking config as the actual YouTube-Chatbot-using-LangChain repo
(chunk_size=500, chunk_overlap=50).

Honesty notes (read before quoting a number from this):
1. We can't download a hosted embedding model (e.g. sentence-transformers) in this
   sandboxed environment (no huggingface.co egress), so we use TF-IDF vectors as the
   FAISS input instead of a real transformer embedding. This is a fair proxy for
   RETRIEVAL SPEED (FAISS query latency depends on vector dimensionality, index type,
   and corpus size -- not on what produced the vectors), but NOT a proxy for retrieval
   QUALITY. Don't claim this benchmarks answer accuracy -- only claim it benchmarks
   query latency.
2. "Baseline keyword search" here = a naive linear scan that term-frequency-scores
   every chunk against the query tokens, sorts, and returns top-k. That's a realistic
   stand-in for "search without a vector index."
3. Only QUERY-TIME cost is measured for both methods (i.e. the cost per question
   asked). One-time index-build cost is measured and reported separately, since a
   real user pays that once per video, not once per question.
"""

import time
import random
import statistics as stats
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Build a synthetic transcript long enough to produce 10,000+ chunks at the
#    repo's chunk_size=500 / chunk_overlap=50 setting (~450 new chars/chunk).
# ---------------------------------------------------------------------------
TOPICS = ["machine learning", "neural networks", "gradient descent", "transformers",
          "attention mechanism", "vector databases", "retrieval augmented generation",
          "prompt engineering", "large language models", "embeddings", "tokenization",
          "fine tuning", "reinforcement learning", "computer vision", "natural language",
          "data pipelines", "model evaluation", "hyperparameter tuning", "overfitting",
          "regularization", "backpropagation", "convolutional layers", "recurrent networks",
          "supervised learning", "unsupervised learning", "clustering", "classification",
          "regression", "feature engineering", "cross validation"]

FILLER = ["so basically what happens here is", "and then the next thing to understand is",
          "let's talk about", "one important detail about", "moving on to",
          "a common misconception about", "in practice", "if you think about it",
          "the key insight is that", "to summarize this part"]

def make_transcript(target_chars):
    words = []
    length = 0
    while length < target_chars:
        sentence = f"{random.choice(FILLER)} {random.choice(TOPICS)} works well when you " \
                   f"combine it with {random.choice(TOPICS)} and {random.choice(TOPICS)}. "
        words.append(sentence)
        length += len(sentence)
    return "".join(words)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def chunk_transcript(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

# ---------------------------------------------------------------------------
# 2. Baseline: naive keyword search (linear scan, term-frequency score)
# ---------------------------------------------------------------------------
def keyword_search(chunks, query, k=3):
    q_terms = query.lower().split()
    scores = []
    for c in chunks:
        c_lower = c.lower()
        score = sum(c_lower.count(t) for t in q_terms)
        scores.append(score)
    top_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_idx]

# ---------------------------------------------------------------------------
# 3. FAISS vector retrieval (TF-IDF vectors as a download-free embedding proxy)
# ---------------------------------------------------------------------------
def build_faiss_index(chunks):
    vectorizer = TfidfVectorizer(max_features=384)  # 384 dims, matching repo's FakeEmbeddings size
    matrix = vectorizer.fit_transform(chunks).toarray().astype("float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return vectorizer, index

def faiss_search(vectorizer, index, chunks, query, k=3):
    qvec = vectorizer.transform([query]).toarray().astype("float32")
    faiss.normalize_L2(qvec)
    _, idx = index.search(qvec, k)
    return [chunks[i] for i in idx[0]]

# ---------------------------------------------------------------------------
# 4. Benchmark harness
# ---------------------------------------------------------------------------
QUERIES = [
    "how does attention mechanism work",
    "explain gradient descent",
    "what is retrieval augmented generation",
    "tell me about overfitting",
    "how do vector databases store data",
    "what is fine tuning",
    "explain tokenization",
    "how does backpropagation work",
    "what is prompt engineering",
    "explain feature engineering",
]

def benchmark_corpus_size(n_target_chunks, n_trials=30):
    # Build a transcript that yields roughly n_target_chunks chunks.
    approx_chars = n_target_chunks * (CHUNK_SIZE - CHUNK_OVERLAP)
    transcript = make_transcript(approx_chars)
    chunks = chunk_transcript(transcript)

    # One-time index build cost (paid once per video, not per question)
    t0 = time.perf_counter()
    vectorizer, index = build_faiss_index(chunks)
    build_time_ms = (time.perf_counter() - t0) * 1000

    keyword_times, faiss_times = [], []
    for _ in range(n_trials):
        query = random.choice(QUERIES)

        t0 = time.perf_counter()
        keyword_search(chunks, query, k=3)
        keyword_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        faiss_search(vectorizer, index, chunks, query, k=3)
        faiss_times.append((time.perf_counter() - t0) * 1000)

    return {
        "n_chunks": len(chunks),
        "index_build_ms": build_time_ms,
        "keyword_mean_ms": stats.mean(keyword_times),
        "keyword_median_ms": stats.median(keyword_times),
        "faiss_mean_ms": stats.mean(faiss_times),
        "faiss_median_ms": stats.median(faiss_times),
    }

if __name__ == "__main__":
    print(f"{'Chunks':>8} | {'Keyword mean (ms)':>18} | {'FAISS mean (ms)':>16} | {'Reduction':>10} | {'Index build (ms)':>17}")
    print("-" * 90)
    results = []
    for target in [500, 2000, 5000, 10000, 20000]:
        r = benchmark_corpus_size(target, n_trials=30)
        reduction = (1 - r["faiss_mean_ms"] / r["keyword_mean_ms"]) * 100
        r["reduction_pct"] = reduction
        results.append(r)
        print(f"{r['n_chunks']:>8} | {r['keyword_mean_ms']:>18.3f} | {r['faiss_mean_ms']:>16.3f} | "
              f"{reduction:>9.1f}% | {r['index_build_ms']:>17.1f}")

    import json
    with open("/home/claude/build/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
