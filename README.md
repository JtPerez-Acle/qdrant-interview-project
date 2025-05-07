# Contexto-Crusher 🚀

An autonomous semantic sleuth that cracks [Contexto.me](https://contexto.me/) in single‑digit guesses, powered by a curated word list, Cognitive Mirrors recursive reasoning, and local embeddings.

## ⚡️ Why This Exists

Qdrant's DNA is vector search. Contexto is a public playground that measures semantic proximity solely by ranking—perfect ground to demonstrate how an introspection‑driven loop can navigate high‑dimensional space with ruthless efficiency.

Our goal: average ≤ 7 guesses across 100 consecutive daily puzzles, all offline on a laptop (no OpenAI calls).

## 🧩 High‑level Architecture

- **Core Engine** – Orchestrates the guessing loop
- **Curated Word List** – Focused vocabulary of 5,000-10,000 common words
- **Local Embedding Index** – HNSW index over the curated word list using sentence-transformers/all-mpnet-base-v2
- **Cognitive Mirrors Loop** – Recursive, self‑critic module that refines candidate distribution after each feedback rank
- **Contexto API Shim** – Headless browser (Playwright) hitting the daily puzzle

## 🔧 Installation

```bash
# 1. Clone
$ git clone https://github.com/<you>/contexto-crusher && cd contexto-crusher

# 2. Install dependencies
$ pip install -r requirements.txt

# 3. Install Playwright browsers
$ playwright install

# 4. Start Qdrant in Docker (recommended)
$ python scripts/start_qdrant_docker.py

# If you encounter Docker permission issues:
$ sudo $(which python) scripts/start_qdrant_docker.py

# 5. Download word list & build index
$ python scripts/build_index.py --download --use-docker
```

## 🚀 Usage

### Building the Vector Index

```bash
# Download the word frequency list and build the vector index
$ python scripts/build_index.py --download --use-docker

# Use a custom word list
$ python scripts/build_index.py --word-list path/to/wordlist.txt --use-docker

# Specify maximum number of words to include
$ python scripts/build_index.py --download --max-words 5000 --use-docker
```

### Downloading Historical Solutions (Optional)

```bash
# Download solutions from the past 30 days
$ python scripts/download_historical_solutions.py --days 30

# Download solutions for a specific date range
$ python scripts/download_historical_solutions.py --start-date 2023-01-01 --end-date 2023-12-31
```

### Solving the Daily Puzzle

```bash
# Solve the daily puzzle
$ python crush.py

# Example output:
2023-12-15 10:30:45,123 - __main__ - INFO - Contexto-Crusher 🚀
2023-12-15 10:30:45,123 - __main__ - INFO - -------------------
2023-12-15 10:30:45,124 - __main__ - INFO - Initializing vector database...
2023-12-15 10:30:45,125 - contexto.vector_db - INFO - Initializing VectorDB with collection: words_curated
2023-12-15 10:30:45,125 - contexto.vector_db - INFO - Using batch size: 64
2023-12-15 10:30:45,126 - contexto.vector_db - INFO - Using local Qdrant instance at ./data/vector_index
2023-12-15 10:30:45,127 - __main__ - INFO - Initializing cognitive mirrors...
2023-12-15 10:30:45,128 - __main__ - INFO - Initializing browser...
2023-12-15 10:30:45,129 - contexto.contexto_api - INFO - Launching browser...
2023-12-15 10:30:46,234 - contexto.contexto_api - INFO - Browser launched successfully
2023-12-15 10:30:46,235 - __main__ - INFO - Navigating to Contexto.me...
2023-12-15 10:30:46,236 - contexto.contexto_api - INFO - Navigating to Contexto.me...
2023-12-15 10:30:47,345 - contexto.contexto_api - INFO - Waiting for page to load...
2023-12-15 10:30:47,678 - contexto.contexto_api - INFO - Page loaded successfully

2023-12-15 10:30:47,679 - __main__ - INFO - Solving the puzzle...
2023-12-15 10:30:47,680 - contexto.contexto_api - INFO - Submitting guess: 'apple'
2023-12-15 10:30:48,123 - contexto.contexto_api - INFO - Guess submitted, waiting for result...
2023-12-15 10:30:48,456 - contexto.contexto_api - INFO - Received rank: 589
2023-12-15 10:30:49,567 - contexto.contexto_api - INFO - Submitting guess: 'document'
2023-12-15 10:30:50,123 - contexto.contexto_api - INFO - Guess submitted, waiting for result...
2023-12-15 10:30:50,456 - contexto.contexto_api - INFO - Received rank: 172
2023-12-15 10:30:51,567 - contexto.contexto_api - INFO - Submitting guess: 'manuscript'
2023-12-15 10:30:52,123 - contexto.contexto_api - INFO - Guess submitted, waiting for result...
2023-12-15 10:30:52,456 - contexto.contexto_api - INFO - Received rank: 23
2023-12-15 10:30:53,567 - contexto.contexto_api - INFO - Submitting guess: 'scroll'
2023-12-15 10:30:54,123 - contexto.contexto_api - INFO - Guess submitted, waiting for result...
2023-12-15 10:30:54,456 - contexto.contexto_api - INFO - Received rank: 5
2023-12-15 10:30:55,567 - contexto.contexto_api - INFO - Submitting guess: 'papyrus'
2023-12-15 10:30:56,123 - contexto.contexto_api - INFO - Guess submitted, waiting for result...
2023-12-15 10:30:56,456 - contexto.contexto_api - INFO - Received rank: 1

2023-12-15 10:30:56,457 - __main__ - INFO - Results:
2023-12-15 10:30:56,458 - __main__ - INFO - Time taken: 10.33 seconds
2023-12-15 10:30:56,459 - __main__ - INFO - Attempts: 5
2023-12-15 10:30:56,460 - __main__ - INFO - Solution: papyrus 🎉

2023-12-15 10:30:56,461 - __main__ - INFO - Guess history:
2023-12-15 10:30:56,462 - __main__ - INFO - 1. Guessed: "apple" → rank 589
2023-12-15 10:30:56,463 - __main__ - INFO - 2. Guessed: "document" → rank 172
2023-12-15 10:30:56,464 - __main__ - INFO - 3. Guessed: "manuscript" → rank 23
2023-12-15 10:30:56,465 - __main__ - INFO - 4. Guessed: "scroll" → rank 5
2023-12-15 10:30:56,466 - __main__ - INFO - 5. Guessed: "papyrus" → rank 1
2023-12-15 10:30:56,467 - contexto.contexto_api - INFO - Closing browser...
2023-12-15 10:30:56,789 - contexto.contexto_api - INFO - Browser closed successfully
```

### Command-Line Options

```bash
# Start with a specific word
$ python crush.py --initial-word "apple"

# Run with visible browser (not headless)
$ python crush.py --no-headless

# Set maximum number of turns
$ python crush.py --max-turns 30

# Get help and see all available options
$ python crush.py --help
```

## 📚 Documentation

- [Architecture](./ARCHITECTURE.md) - Detailed system design and component interactions
- [Development](./DEVELOPMENT.md) - Setup, workflow, and contribution guidelines
- [API](./API.md) - API documentation for core components
- [Testing](./TESTING.md) - Testing strategy and procedures

## 🧪 Testing & Benchmarks

### Quick End-to-End Test

To verify that the entire pipeline is working correctly:

```bash
# Run the end-to-end test (starts Docker, builds index, tests solver, cleans up)
$ python scripts/test_pipeline.py

# If you encounter Docker permission issues, use the helper script:
$ ./scripts/run_test_with_docker.sh

# If Docker doesn't work at all, use the simple test without Docker:
$ python scripts/test_without_docker.py
```

This test script:
1. Starts Qdrant in Docker
2. Creates a small test dataset
3. Builds a vector index
4. Tests the solver with a known target word
5. Cleans up everything when done

### Benchmark Protocol

- **Historical dataset** – We replay the last 100 daily puzzles (scraped w/ checksum)
- **Metrics** – Mean, median, p95 attempts; total runtime
- **Ablations** – Disable Cognitive Mirrors loop, vary introspection depth, swap embedding models
- **Reproduction** – `eval.py --ablate` auto‑generates a results table + plot

## 🛠️ Tech Stack

| Category | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.10+ | Fast prototyping + rich ML ecosystem |
| Vector DB | Qdrant (embedded) | Efficient vector search with HNSW index |
| Word List | Curated 5-10k words | Focused vocabulary for better performance |
| Embeddings | sentence-transformers/all-mpnet-base-v2 | Strong semantic signal, 768‑d, lightweight |
| Headless client | Playwright | Reliable, handles JS, rate‑limit friendly |
| Introspection depth | 2 iterations | Empirically best trade‑off vs. latency |
| Compute | CPU‑only, <300 MB RAM | Reduced memory footprint with curated list |

## 📄 License

[MIT](LICENSE)
