# Testing Strategy

This document outlines the testing strategy for Contexto-Crusher, including test types, coverage goals, and procedures.

## Testing Philosophy

Contexto-Crusher follows a test-driven development (TDD) approach with a focus on:

1. **Unit tests** for individual components
2. **Integration tests** for component interactions
3. **End-to-end tests** for complete system behavior
4. **Performance tests** to ensure efficiency

## Test Structure

Tests are organized in the `tests/` directory, mirroring the structure of the main package:

```
tests/
├── __init__.py
├── test_solver.py              # Core Engine tests
├── test_cognitive_mirrors.py   # Cognitive Mirrors tests
├── test_vector_db.py           # Vector Database tests
├── test_contexto_api.py        # Contexto API tests
├── test_integration.py         # Component integration tests
└── test_end_to_end.py          # Complete system tests
```

## Test Types

### Unit Tests

Unit tests verify the behavior of individual functions and classes in isolation, using mocks for dependencies.

**Key Areas:**
- Word proposal and selection algorithms
- Reflection and introspection logic
- Vector similarity calculations
- API interaction patterns

**Example:**
```python
def test_propose_candidates():
    # Arrange
    mock_vector_db = Mock()
    mock_vector_db.search.return_value = [("word1", 0.9), ("word2", 0.8)]
    solver = Solver(vector_db=mock_vector_db, cognitive_mirrors=Mock(), contexto_api=Mock())
    
    # Act
    candidates = solver.propose_candidates(k=2)
    
    # Assert
    assert len(candidates) == 2
    assert "word1" in candidates
    assert "word2" in candidates
    mock_vector_db.search.assert_called_once()
```

### Integration Tests

Integration tests verify that components work together correctly.

**Key Interactions:**
- Solver + Cognitive Mirrors
- Solver + Vector Database
- Cognitive Mirrors + Vector Database

**Example:**
```python
def test_solver_with_cognitive_mirrors():
    # Arrange
    vector_db = VectorDB(collection_name="test_collection", path="./test_data")
    vector_db.build_index("./test_data/small_word_list.txt")
    cognitive_mirrors = CognitiveMirrors(vector_db)
    mock_api = Mock()
    mock_api.submit_guess.side_effect = lambda word: 10 if word != "target" else 1
    solver = Solver(vector_db, cognitive_mirrors, mock_api)
    
    # Act
    result = solver.solve(initial_word="start")
    
    # Assert
    assert result["solution"] == "target"
    assert result["attempts"] <= 10
```

### End-to-End Tests

End-to-end tests verify the complete system behavior using either:
- Mocked Contexto API with predefined responses
- Historical puzzles with known solutions

**Example:**
```python
def test_end_to_end_with_mock():
    # Arrange
    mock_api = MockContextoAPI()
    mock_api.set_target_word("democracy")
    solver = create_solver(contexto_api=mock_api)
    
    # Act
    result = solver.solve()
    
    # Assert
    assert result["solution"] == "democracy"
    assert result["attempts"] <= 10
```

### Performance Tests

Performance tests ensure the system meets efficiency requirements.

**Key Metrics:**
- Time to solve a puzzle
- Memory usage
- Number of API calls

**Example:**
```python
def test_solver_performance():
    # Arrange
    solver = create_solver()
    puzzles = load_test_puzzles(10)
    
    # Act
    start_time = time.time()
    results = [solver.solve(puzzle) for puzzle in puzzles]
    end_time = time.time()
    
    # Assert
    avg_attempts = sum(r["attempts"] for r in results) / len(results)
    assert avg_attempts <= 10
    assert end_time - start_time <= 60  # Should solve 10 puzzles in under 60 seconds
```

## Test Fixtures

Common test fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def vector_db():
    """Create a test vector database with a small word list."""
    db = VectorDB(collection_name="test_collection", path="./test_data")
    db.build_index("./test_data/small_word_list.txt")
    yield db
    # Cleanup
    shutil.rmtree("./test_data", ignore_errors=True)

@pytest.fixture
def mock_contexto_api():
    """Create a mock Contexto API that returns predefined ranks."""
    api = Mock()
    api.submit_guess.side_effect = lambda word: MOCK_RANKS.get(word, 1000)
    return api
```

## Mocking

The project uses the following mocking strategies:

1. **Vector Database**: Small test collections with predefined embeddings
2. **Contexto API**: Mock implementation that returns predefined ranks
3. **Cognitive Mirrors**: Simplified implementation for testing

## Test Data

Test data includes:

1. **Small word list**: ~1000 common words for quick testing
2. **Test embeddings**: Pre-computed embeddings for test words
3. **Historical puzzles**: A subset of past Contexto puzzles with known solutions

## Continuous Integration

Tests are run automatically on:
- Pull requests
- Commits to main branch
- Daily scheduled runs

## Test Coverage

The project aims for:
- 90%+ line coverage for core components
- 80%+ line coverage overall

Coverage reports are generated during CI runs.

## Running Tests

### Running All Tests

```bash
poetry run pytest
```

### Running Specific Test Types

```bash
# Unit tests only
poetry run pytest tests/test_*.py -k "not integration and not end_to_end"

# Integration tests
poetry run pytest tests/test_integration.py

# End-to-end tests
poetry run pytest tests/test_end_to_end.py
```

### Running with Coverage

```bash
poetry run pytest --cov=contexto
```

## Debugging Tests

For detailed output:

```bash
poetry run pytest -v
```

For even more detail:

```bash
poetry run pytest -vv --log-cli-level=DEBUG
```

## Test-Driven Development Workflow

1. Write a failing test for the feature/fix
2. Implement the minimum code to make the test pass
3. Refactor while keeping tests passing
4. Repeat
