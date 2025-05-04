# Contexto-Crusher 🚀 (Basic architecture complete, performance evaluation and real tests pending)

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

# For large datasets (recommended):
# First start Qdrant in Docker
$ python scripts/start_qdrant_docker.py

# If you encounter Docker permission issues:
$ sudo $(which python) scripts/start_qdrant_docker.py

# Then build the index using Docker mode
$ python scripts/build_index.py --download --use-docker
```

## 🚀 Usage

### Building the Vector Index

```bash
# Download the word list and build the vector index
$ python scripts/build_index.py --download

# Use a custom word list
$ python scripts/build_index.py --word-list path/to/wordlist.txt

# Use Qdrant in Docker mode
$ python scripts/build_index.py --use-docker --qdrant-url http://localhost:6333
```

### Solving the Daily Puzzle

```bash
# Solve the daily puzzle
$ python scripts/solve_contexto.py daily

# Example output:
Contexto-Crusher - Daily Puzzle Solver
--------------------------------------------------------------------------------
Initializing vector database from data/vector_index...
Using Qdrant in local mode
Initializing cognitive mirrors...
Initializing Contexto API...
Starting Contexto API...
Navigating to the daily puzzle...
Initializing solver...

Solving the puzzle...
Initial guess: 'strawberry' → rank 823
Turn 2: Guessed 'apple' → rank 589
Turn 3: Guessed 'banana' → rank 612
Turn 4: Guessed 'document' → rank 172
Turn 5: Guessed 'manuscript' → rank 23
Turn 6: Guessed 'scroll' → rank 5
Turn 7: Guessed 'papyrus' → rank 1

✅ Success! Solved in 7 attempts.
Solution: papyrus

Guess history:
1. Guessed: "strawberry" → rank 823
2. Guessed: "apple" → rank 589
3. Guessed: "banana" → rank 612
4. Guessed: "document" → rank 172
5. Guessed: "manuscript" → rank 23
6. Guessed: "scroll" → rank 5
7. Guessed: "papyrus" → rank 1

Stopping Contexto API...
```

### Solving Historical Puzzles

```bash
# Solve a historical puzzle (specify the date in YYYY-MM-DD format)
$ python scripts/solve_contexto.py historical 2023-05-01

# Use Qdrant in Docker mode
$ python scripts/solve_contexto.py daily --use-docker --qdrant-url http://localhost:6333

# Specify a custom vector database path
$ python scripts/solve_contexto.py daily --vector-db path/to/vector_index

# Start with a specific word
$ python scripts/solve_contexto.py daily --initial-word "apple"

# Run with visible browser (not headless)
$ python scripts/solve_contexto.py daily --headless=false

# Set maximum number of turns
$ python scripts/solve_contexto.py daily --max-turns 100
```

### Advanced Usage

```bash
# Get help and see all available options
$ python scripts/solve_contexto.py --help
$ python scripts/solve_contexto.py daily --help
$ python scripts/solve_contexto.py historical --help
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
| Language | Python 3.11 | Fast prototyping + rich ML ecosystem |
| Vector DB | Qdrant (embedded) | Efficient vector search with HNSW index |
| Embeddings | sentence-transformers/all-mpnet-base-v2 | Strong semantic signal, 768‑d, lightweight |
| Headless client | Playwright | Reliable, handles JS, rate‑limit friendly |
| Introspection depth | 2 iterations | Empirically best trade‑off vs. latency |
| Compute | CPU‑only, <500 MB RAM | Must run on vanilla laptop |

## 📄 License

[MIT](LICENSE)
