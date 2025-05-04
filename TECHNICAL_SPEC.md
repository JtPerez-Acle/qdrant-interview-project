# Technical Specification

This document provides detailed technical specifications for implementing the Contexto-Crusher project.

## 1. Vector Database (Qdrant)

### 1.1 Implementation Details

#### Embedding Model
- **Model**: sentence-transformers/all-mpnet-base-v2
- **Vector Dimensions**: 768
- **Normalization**: L2-normalized vectors

#### Qdrant Configuration
- **Mode**: Embedded (file-based, no server)
- **Collection Name**: "words"
- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Distance Function**: Cosine
- **HNSW Parameters**:
  - `ef_construct`: 128
  - `m`: 16
  - `ef_search`: 100

#### Word List
- **Source**: Common English words (~200,000)
- **Format**: Plain text file, one word per line
- **Preprocessing**: Lowercase, remove special characters

### 1.2 API Design

```python
class VectorDB:
    def __init__(self, collection_name="words", path="./data/vector_index"):
        """Initialize the vector database."""
        self.collection_name = collection_name
        self.path = path
        self.client = QdrantClient(path=path)
        self.model = None
        
    def load_model(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        """Load the embedding model."""
        self.model = SentenceTransformer(model_name)
        
    def build_index(self, word_list_path):
        """Build the vector index from a word list."""
        # Implementation details
        
    def search(self, query, limit=10):
        """Search for semantically similar words."""
        # Implementation details
        
    def get_embedding(self, word):
        """Get the embedding vector for a word."""
        # Implementation details
```

## 2. Contexto API (Playwright)

### 2.1 Implementation Details

#### Browser Configuration
- **Browser**: Chromium
- **Mode**: Headless (default)
- **Viewport**: 1280x720
- **User Agent**: Standard browser user agent

#### Page Interaction
- **URL**: https://contexto.me/
- **Input Selector**: `input[type="text"]`
- **Submit Selector**: `button[type="submit"]`
- **Result Selector**: `.result-item`
- **Rank Extraction**: Regular expression from result text

#### Rate Limiting
- **Delay Between Requests**: 1-2 seconds (randomized)
- **Backoff Strategy**: Exponential backoff on 429 responses
- **Max Retries**: 3

### 2.2 API Design

```python
class ContextoAPI:
    def __init__(self, headless=True):
        """Initialize the Contexto API client."""
        self.headless = headless
        self.browser = None
        self.page = None
        
    async def start(self):
        """Start the browser session."""
        # Implementation details
        
    async def stop(self):
        """Stop the browser session."""
        # Implementation details
        
    async def navigate_to_daily(self):
        """Navigate to the daily puzzle."""
        # Implementation details
        
    async def submit_guess(self, word):
        """Submit a guess and get the rank."""
        # Implementation details
        
    async def get_history(self):
        """Get the history of guesses from the current session."""
        # Implementation details
```

## 3. Cognitive Mirrors Loop

### 3.1 Implementation Details

#### Reflection Process
- **Introspection Depth**: 2 iterations
- **Reflection Types**:
  - Semantic basin analysis
  - Polysemy detection
  - Word morphology shifts
  - Contextual domain analysis

#### Candidate Refinement
- **Initial Candidates**: Top-k from vector search
- **Refinement Strategy**: Re-ranking based on reflection
- **Selection Criteria**: Weighted combination of:
  - Vector similarity
  - Historical rank patterns
  - Reflection insights

#### Internal Dialogue
- **Question Types**:
  - "Are we stuck in a local semantic basin?"
  - "Do ranks suggest a polysemous cluster we ignored?"
  - "Should we pivot word morphology (noun → verb)?"
  - "Is there a domain shift we're missing?"

### 3.2 API Design

```python
class CognitiveMirrors:
    def __init__(self, vector_db, introspection_depth=2):
        """Initialize the cognitive mirrors module."""
        self.vector_db = vector_db
        self.introspection_depth = introspection_depth
        
    def critic(self, candidates, history):
        """Analyze candidates and history to generate reflection."""
        # Implementation details
        
    def refine(self, candidates, reflection, history):
        """Refine candidates based on reflection."""
        # Implementation details
        
    def introspect(self, history):
        """Generate introspective questions based on guess history."""
        # Implementation details
        
    def _analyze_semantic_basin(self, history):
        """Analyze if we're stuck in a local semantic basin."""
        # Implementation details
        
    def _detect_polysemy(self, history):
        """Detect if we're missing polysemous meanings."""
        # Implementation details
        
    def _analyze_morphology(self, history):
        """Analyze if we should shift word morphology."""
        # Implementation details
```

## 4. Core Engine (Solver)

### 4.1 Implementation Details

#### Solving Strategy
- **Initial Guess**: Either user-provided or selected from common words
- **Candidate Generation**: Combination of:
  - Vector similarity search
  - Historical pattern matching
  - Cognitive reflection

#### Selection Algorithm
- **Scoring Function**: Weighted combination of:
  - Vector similarity to estimated target
  - Exploration value (semantic distance from previous guesses)
  - Reflection-based adjustments

#### Termination Conditions
- **Success**: Rank 1 achieved
- **Failure**: Max turns reached (default: 20)
- **Early Stopping**: If estimated rank improvement is below threshold

### 4.2 API Design

```python
class Solver:
    def __init__(self, vector_db, cognitive_mirrors, contexto_api, max_turns=20):
        """Initialize the solver with required components."""
        self.vector_db = vector_db
        self.cognitive_mirrors = cognitive_mirrors
        self.contexto_api = contexto_api
        self.max_turns = max_turns
        self.history = []
        
    def solve(self, initial_word=None):
        """Solve the current Contexto puzzle."""
        # Implementation details
        
    def propose_candidates(self, k=10):
        """Propose k candidate words based on current state."""
        # Implementation details
        
    def select_best_candidate(self, candidates):
        """Select the best word from candidates to guess next."""
        # Implementation details
        
    def _estimate_target_vector(self):
        """Estimate the target word's vector based on history."""
        # Implementation details
        
    def _calculate_exploration_value(self, word):
        """Calculate exploration value for a candidate word."""
        # Implementation details
```

## 5. Data Structures

### 5.1 History Entry
```python
HistoryEntry = namedtuple("HistoryEntry", ["word", "rank", "vector", "timestamp"])
```

### 5.2 Candidate
```python
Candidate = namedtuple("Candidate", ["word", "score", "vector", "exploration_value", "reflection_score"])
```

### 5.3 Reflection
```python
Reflection = namedtuple("Reflection", ["text", "insights", "suggested_pivots"])
```

### 5.4 Insight
```python
Insight = namedtuple("Insight", ["type", "description", "confidence", "suggested_action"])
```

## 6. Configuration Management

### 6.1 Configuration File Structure
```yaml
# config.yaml
vector_db:
  collection_name: "words"
  path: "./data/vector_index"
  model_name: "sentence-transformers/all-mpnet-base-v2"
  
contexto_api:
  headless: true
  delay_min: 1.0
  delay_max: 2.0
  max_retries: 3
  
cognitive_mirrors:
  introspection_depth: 2
  reflection_types:
    - "semantic_basin"
    - "polysemy"
    - "morphology"
    - "domain"
    
solver:
  max_turns: 20
  candidate_count: 10
  weights:
    similarity: 0.6
    exploration: 0.3
    reflection: 0.1
```

### 6.2 Configuration Loading
```python
def load_config(config_path="./config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
```

## 7. Performance Considerations

### 7.1 Caching
- **Embedding Cache**: Cache word embeddings to avoid recomputation
- **Search Cache**: Cache search results for common queries
- **Reflection Cache**: Cache reflection insights for similar histories

### 7.2 Parallelization
- **Async API Calls**: Use async/await for Playwright interactions
- **Batch Processing**: Process candidate embeddings in batches
- **Concurrent Reflection**: Run reflection analyses concurrently

### 7.3 Memory Management
- **Lazy Loading**: Load embedding model only when needed
- **Incremental Index**: Build index incrementally if needed
- **History Pruning**: Keep only relevant history entries

## 8. Error Handling

### 8.1 Error Types
- **VectorDBError**: Errors related to vector database operations
- **ContextoAPIError**: Errors related to website interactions
- **CognitiveMirrorsError**: Errors in reflection process
- **SolverError**: Errors in the solving process

### 8.2 Recovery Strategies
- **API Failures**: Retry with exponential backoff
- **Model Loading Failures**: Fall back to simpler model
- **Index Corruption**: Rebuild index from scratch
- **Solver Stagnation**: Reset and try alternative strategy

## 9. Logging and Monitoring

### 9.1 Logging Structure
```python
import logging

logger = logging.getLogger("contexto-crusher")
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("contexto-crusher.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(console_handler)
```

### 9.2 Metrics Collection
- **Attempts**: Number of guesses per puzzle
- **Success Rate**: Percentage of puzzles solved
- **Time**: Time taken to solve each puzzle
- **Memory Usage**: Peak memory usage
- **Reflection Impact**: Improvement due to reflection

## 10. Testing Strategy

### 10.1 Unit Tests
- **VectorDB Tests**: Test embedding and search functionality
- **ContextoAPI Tests**: Test website interaction with mocks
- **CognitiveMirrors Tests**: Test reflection and refinement
- **Solver Tests**: Test solving strategy with mocked components

### 10.2 Integration Tests
- **Component Integration**: Test interactions between components
- **End-to-End Flow**: Test complete solving process with mocks

### 10.3 Performance Tests
- **Benchmark Tests**: Measure solving time and attempts
- **Memory Tests**: Monitor memory usage during solving
- **Scalability Tests**: Test with varying word list sizes
