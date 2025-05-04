# Contexto-Crusher 🚀

An autonomous semantic sleuth that cracks [Contexto.me](https://contexto.me/) in single‑digit guesses, powered by Cognitive Mirrors recursive reasoning and local embeddings.

## ⚡️ Why This Exists

Qdrant's DNA is vector search. Contexto is a public playground that measures semantic proximity solely by ranking—perfect ground to demonstrate how an introspection‑driven loop can navigate high‑dimensional space with ruthless efficiency.

Our goal: average ≤ 7 guesses across 100 consecutive daily puzzles, all offline on a laptop (no OpenAI calls).

## 🧩 High‑level Architecture

- **Core Engine** – Orchestrates the guessing loop
- **Local Embedding Index** – HNSW index over ~200k common English words using sentence-transformers/all-mpnet-base-v2
- **Cognitive Mirrors Loop** – Recursive, self‑critic module that refines candidate distribution after each feedback rank
- **Contexto API Shim** – Headless browser (Playwright) hitting the daily puzzle

## 🔧 Installation

```bash
# 1. Clone
$ git clone https://github.com/<you>/contexto-crusher && cd contexto-crusher

# 2. Install dependencies
$ pip install -r requirements.txt

# 3. Download word list & build index
$ python scripts/build_index.py --download
```

## 🚀 Usage

### Solving the Daily Puzzle

```bash
$ python crush.py
Contexto-Crusher 🚀
-------------------
Initializing vector database...
Initializing cognitive mirrors...
Initializing browser...
Navigating to Contexto.me...

Solving the puzzle...

Results:
Time taken: 12.34 seconds
Attempts: 5
Solution: papyrus 🎉

Guess history:
1. Guessed: "paper" → rank 823
2. Guessed: "document" → rank 172
3. Guessed: "manuscript" → rank 23
4. Guessed: "scroll" → rank 5
5. Guessed: "papyrus" → rank 1
```

### Evaluation on Historical Puzzles

```bash
$ python eval.py --days 10 --verbose
Contexto-Crusher Evaluation 🚀
-----------------------------
Initializing vector database...
Initializing cognitive mirrors...
Initializing browser (mock)...

Evaluating solver on 10 historical puzzles...

Evaluation Results:
Mean attempts: 6.82
Median attempts: 6.00
95th percentile attempts: 9.00
Min attempts: 4
Max attempts: 10
Success rate: 100.00%
Mean time per puzzle: 5.67 seconds
Total time: 56.70 seconds

Results saved to results.json
```

### Running Ablation Studies

```bash
$ python eval.py --days 10 --ablate
# This will run ablation studies and generate a plot
```

## 📚 Documentation

- [Architecture](./ARCHITECTURE.md) - Detailed system design and component interactions
- [Development](./DEVELOPMENT.md) - Setup, workflow, and contribution guidelines
- [API](./API.md) - API documentation for core components
- [Testing](./TESTING.md) - Testing strategy and procedures

## 🧪 Benchmark Protocol

- **Historical dataset** – We replay the last 100 daily puzzles (scraped w/ checksum)
- **Metrics** – Mean, median, p95 attempts; total runtime
- **Ablations** – Disable Cognitive Mirrors loop, vary introspection depth, swap embedding models
- **Reproduction** – `eval.py --ablate` auto‑generates a results table + plot

## 🛠️ Tech Stack

| Category | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.11 | Fast prototyping + rich ML ecosystem |
| Vector DB | Qdrant (embedded) | Efficient vector search with HNSW index |
| Embeddings | sentence-transformers/all-mpnet-base-v2 | Strong semantic signal, 768‑d, lightweight |
| Headless client | Playwright | Reliable, handles JS, rate‑limit friendly |
| Introspection depth | 2 iterations | Empirically best trade‑off vs. latency |
| Compute | CPU‑only, <500 MB RAM | Must run on vanilla laptop |

## 📄 License

[MIT](LICENSE)
