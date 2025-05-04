# API Documentation

This document provides detailed API documentation for the core components of Contexto-Crusher.

## Core Engine (Solver)

The Core Engine is responsible for orchestrating the guessing process and integrating all components.

### `Solver` Class

```python
class Solver:
    def __init__(self, vector_db, cognitive_mirrors, contexto_api, max_turns=20):
        """
        Initialize the solver with required components.
        
        Args:
            vector_db: VectorDB instance for word embeddings
            cognitive_mirrors: CognitiveMirrors instance for reflection
            contexto_api: ContextoAPI instance for submitting guesses
            max_turns: Maximum number of turns to attempt (default: 20)
        """
        
    def solve(self, initial_word=None):
        """
        Solve the current Contexto puzzle.
        
        Args:
            initial_word: Optional starting word (default: None)
            
        Returns:
            dict: Results including solution word, number of attempts, and history
        """
        
    def propose_candidates(self, k=10):
        """
        Propose k candidate words based on current state.
        
        Args:
            k: Number of candidates to propose (default: 10)
            
        Returns:
            list: List of candidate words
        """
        
    def select_best_candidate(self, candidates):
        """
        Select the best word from candidates to guess next.
        
        Args:
            candidates: List of candidate words
            
        Returns:
            str: Selected word to guess
        """
```

## Cognitive Mirrors Loop

The Cognitive Mirrors Loop implements the recursive reasoning and reflection capabilities.

### `CognitiveMirrors` Class

```python
class CognitiveMirrors:
    def __init__(self, vector_db, introspection_depth=2):
        """
        Initialize the cognitive mirrors module.
        
        Args:
            vector_db: VectorDB instance for word embeddings
            introspection_depth: Number of reflection iterations (default: 2)
        """
        
    def critic(self, candidates, history):
        """
        Analyze candidates and history to generate reflection.
        
        Args:
            candidates: List of candidate words
            history: List of (word, rank) tuples from previous guesses
            
        Returns:
            str: Reflection text analyzing patterns and suggesting improvements
        """
        
    def refine(self, candidates, reflection, history):
        """
        Refine candidates based on reflection.
        
        Args:
            candidates: List of candidate words
            reflection: Reflection text from critic
            history: List of (word, rank) tuples from previous guesses
            
        Returns:
            list: Refined list of candidate words
        """
        
    def introspect(self, history):
        """
        Generate introspective questions based on guess history.
        
        Args:
            history: List of (word, rank) tuples from previous guesses
            
        Returns:
            list: List of introspective questions
        """
```

## Vector Database (Qdrant)

The Vector Database component manages word embeddings and semantic similarity searches.

### `VectorDB` Class

```python
class VectorDB:
    def __init__(self, collection_name="words", path="./data/vector_index"):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the Qdrant collection (default: "words")
            path: Path to store the Qdrant database (default: "./data/vector_index")
        """
        
    def build_index(self, word_list_path, model_name="sentence-transformers/all-mpnet-base-v2"):
        """
        Build the vector index from a word list.
        
        Args:
            word_list_path: Path to text file with words (one per line)
            model_name: Name of the sentence transformer model (default: "sentence-transformers/all-mpnet-base-v2")
            
        Returns:
            int: Number of words indexed
        """
        
    def search(self, query, limit=10):
        """
        Search for semantically similar words.
        
        Args:
            query: Query word or phrase
            limit: Maximum number of results (default: 10)
            
        Returns:
            list: List of (word, score) tuples
        """
        
    def get_embedding(self, word):
        """
        Get the embedding vector for a word.
        
        Args:
            word: Word to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        
    def get_distance(self, word1, word2):
        """
        Calculate the distance between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            float: Distance between the words
        """
```

## Contexto API (Playwright)

The Contexto API component handles interactions with the Contexto.me website.

### `ContextoAPI` Class

```python
class ContextoAPI:
    def __init__(self, headless=True):
        """
        Initialize the Contexto API client.
        
        Args:
            headless: Whether to run the browser in headless mode (default: True)
        """
        
    async def start(self):
        """
        Start the browser session.
        
        Returns:
            bool: True if successful
        """
        
    async def stop(self):
        """
        Stop the browser session.
        
        Returns:
            bool: True if successful
        """
        
    async def navigate_to_daily(self):
        """
        Navigate to the daily puzzle.
        
        Returns:
            bool: True if successful
        """
        
    async def submit_guess(self, word):
        """
        Submit a guess and get the rank.
        
        Args:
            word: Word to guess
            
        Returns:
            int: Rank of the guessed word (1 is the target word)
        """
        
    async def get_history(self):
        """
        Get the history of guesses from the current session.
        
        Returns:
            list: List of (word, rank) tuples
        """
```

## Utility Functions

### Word Processing

```python
def normalize_word(word):
    """
    Normalize a word (lowercase, remove special characters).
    
    Args:
        word: Word to normalize
        
    Returns:
        str: Normalized word
    """
    
def is_valid_word(word, word_list):
    """
    Check if a word is valid (in the word list).
    
    Args:
        word: Word to check
        word_list: List of valid words
        
    Returns:
        bool: True if the word is valid
    """
```

### Evaluation

```python
def evaluate_solver(solver, puzzles, verbose=False):
    """
    Evaluate solver performance on multiple puzzles.
    
    Args:
        solver: Solver instance
        puzzles: List of puzzle target words
        verbose: Whether to print detailed results (default: False)
        
    Returns:
        dict: Evaluation metrics (mean, median, p95 attempts)
    """
```

## Command Line Interfaces

### crush.py

```
usage: crush.py [-h] [--initial-word INITIAL_WORD] [--max-turns MAX_TURNS]

Solve the daily Contexto puzzle.

optional arguments:
  -h, --help            show this help message and exit
  --initial-word INITIAL_WORD
                        Initial word to start with
  --max-turns MAX_TURNS
                        Maximum number of turns (default: 20)
```

### eval.py

```
usage: eval.py [-h] [--days DAYS] [--ablate] [--output OUTPUT]

Evaluate solver performance on historical puzzles.

optional arguments:
  -h, --help       show this help message and exit
  --days DAYS      Number of historical days to evaluate (default: 100)
  --ablate         Run ablation studies
  --output OUTPUT  Output file for results (default: results.json)
```
