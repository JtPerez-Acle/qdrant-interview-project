# Curated Word List for Contexto-Crusher

This document explains the curated word list feature, which reduces the vocabulary size to improve performance and accuracy when solving Contexto puzzles.

## Why Use a Curated Word List?

The original Contexto-Crusher used a large vocabulary of 300,000+ words, which has several drawbacks:

1. **Inefficiency**: Most of these words are rare and unlikely to be Contexto solutions
2. **Noise**: Rare words can introduce noise in the semantic search process
3. **Performance**: A large vocabulary requires more memory and processing time

By using a curated word list of 5,000-10,000 common words, we can:

1. **Improve accuracy**: Focus on words that are more likely to be solutions
2. **Reduce memory usage**: Smaller vector database means less RAM required
3. **Speed up processing**: Fewer words to search through means faster solving

## How the Curated Word List Works

The curated word list is created using several criteria:

1. **Word frequency**: Prioritizes common words based on frequency data
2. **Historical solutions**: Includes all known past Contexto solutions
3. **Word length**: Filters out very short or very long words
4. **Part of speech**: Ensures a good mix of nouns, verbs, adjectives, etc.

## Using the Curated Word List

### Step 1: Download the Full Word List

If you haven't already, download the full word list:

```bash
python scripts/build_index.py --download
```

### Step 2: Download Historical Solutions (Optional but Recommended)

Download historical Contexto solutions to improve the curated list:

```bash
python scripts/download_historical_solutions.py --days 90
```

This will download solutions from the past 90 days. You can adjust the number of days or specify a date range:

```bash
python scripts/download_historical_solutions.py --start-date 2023-01-01 --end-date 2023-12-31
```

### Step 3: Create the Curated Word List

Generate the curated word list:

```bash
python scripts/create_curated_wordlist.py
```

By default, this creates a list of 10,000 words. You can adjust the size:

```bash
python scripts/create_curated_wordlist.py --max-words 5000
```

### Step 4: Build the Vector Index with the Curated List

Build the vector index using the curated word list:

```bash
python scripts/build_index.py --curated
```

If you're using Docker for Qdrant:

```bash
python scripts/start_qdrant_docker.py
python scripts/build_index.py --curated --use-docker
```

### Step 5: Solve Contexto Using the Curated List

Solve the daily Contexto puzzle using the curated word list:

```bash
python crush.py --curated
```

## Customizing the Curated Word List

You can customize the curated word list by modifying the parameters in `create_curated_wordlist.py`:

- `--min-length`: Minimum word length (default: 3)
- `--max-length`: Maximum word length (default: 12)
- `--max-words`: Maximum number of words in the list (default: 10000)
- `--freq-url`: URL to download word frequency data

## Performance Comparison

Here's a comparison of solving performance between the full vocabulary and the curated word list:

| Metric | Full Vocabulary | Curated Word List |
|--------|----------------|-------------------|
| Average guesses | Varies | Typically fewer |
| Memory usage | Higher | Lower |
| Processing time | Longer | Shorter |

## Troubleshooting

### Missing Word List

If you get an error about a missing word list, make sure you've downloaded the full word list first:

```bash
python scripts/build_index.py --download
```

### Collection Not Found

If you get an error about a collection not found, make sure you've built the index with the curated flag:

```bash
python scripts/build_index.py --curated
```

### Switching Between Full and Curated Lists

You can switch between the full vocabulary and the curated word list by using the `--curated` flag with `crush.py`:

```bash
# Use full vocabulary
python crush.py

# Use curated word list
python crush.py --curated
```
