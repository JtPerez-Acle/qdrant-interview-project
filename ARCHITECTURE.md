# Contexto-Crusher Architecture

This document outlines the architecture of Contexto-Crusher, explaining how the components interact to solve Contexto puzzles efficiently.

## System Overview

Contexto-Crusher is designed as a modular system with four main components:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   Core Engine   │◄────┤  Local Vector   │     │  Contexto.me    │
│   (Solver)      │────►│  Database       │     │  Website        │
│                 │     │  (Qdrant)       │     │                 │
└────────┬────────┘     └─────────────────┘     └────────▲────────┘
         │                                               │
         │                                               │
         │                                               │
         │                                               │
         │                                               │
┌────────▼────────┐                            ┌─────────┴────────┐
│                 │                            │                  │
│   Cognitive     │                            │   Headless       │
│   Mirrors Loop  │                            │   Browser        │
│                 │                            │   (Playwright)   │
└─────────────────┘                            └──────────────────┘
```

## Component Details

### 1. Core Engine (Solver)

The Core Engine is the central orchestrator that:
- Manages the guessing loop
- Coordinates between the vector database and Cognitive Mirrors
- Selects the best candidate words to guess
- Tracks history and state of the current puzzle

**Key Functions:**
- `propose()`: Generate initial candidate words
- `select()`: Choose the best word to guess
- `update_history()`: Record guess results and update state

### 2. Local Embedding Index (Qdrant)

A vector database that:
- Stores embeddings for ~200,000 common English words
- Provides efficient nearest-neighbor search capabilities
- Enables semantic similarity queries

**Implementation Details:**
- Uses Qdrant in embedded mode (no separate server)
- HNSW index for fast approximate nearest neighbor search
- Stores 768-dimensional vectors from sentence-transformers/all-mpnet-base-v2

### 3. Cognitive Mirrors Loop

The recursive reasoning module that:
- Analyzes patterns in previous guesses and their rankings
- Questions assumptions and refines search strategy
- Improves candidate selection through introspection

**Key Functions:**
- `critic()`: Analyze current candidates and history
- `refine()`: Generate improved candidates based on reflection
- `introspect()`: Run internal dialogue to question assumptions

**Pseudocode:**
```python
for step in range(max_turns):
    candidates = propose()
    reflection = critic(candidates, history)
    candidates = refine(candidates, reflection)
    guess = select(candidates)
    rank = contexto.submit(guess)
    history.append((guess, rank))
    if rank == 1:
        break
```

### 4. Contexto API Shim (Playwright)

A headless browser interface that:
- Interacts with the Contexto.me website
- Submits guesses and retrieves rankings
- Handles rate limiting and website navigation

**Key Functions:**
- `submit(word)`: Submit a guess and return its rank
- `get_daily_puzzle()`: Navigate to the current day's puzzle
- `handle_rate_limits()`: Implement backoff strategies if needed

## Data Flow

1. The Core Engine proposes initial candidate words based on vector similarity
2. Candidates are passed to the Cognitive Mirrors Loop for reflection
3. Refined candidates are selected and submitted to Contexto via Playwright
4. The resulting rank is recorded in history
5. This process repeats, with each iteration becoming more focused, until rank 1 is achieved

## Key Design Decisions

1. **Embedded Qdrant**: Using embedded mode eliminates network overhead and simplifies deployment
2. **Introspection Depth**: Two iterations of reflection provides the optimal balance between accuracy and speed
3. **CPU-Only**: All components are designed to run efficiently without GPU acceleration
4. **Stateful History**: The system maintains a complete history of guesses and ranks to inform future guesses

## Performance Considerations

- The system is optimized to run on a standard laptop with <500MB RAM
- Vector operations are the most compute-intensive part of the system
- Caching is used to avoid redundant embedding calculations
- Rate limiting is implemented to avoid overloading the Contexto website
