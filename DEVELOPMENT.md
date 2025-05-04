# Development Guide

This document provides guidelines and instructions for developing and contributing to the Contexto-Crusher project.

## Development Environment Setup

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- Git

### Setting Up Your Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/<you>/contexto-crusher
   cd contexto-crusher
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

3. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

4. Build the word embedding index:
   ```bash
   poetry run python build_index.py
   ```

## Project Structure

```
contexto-crusher/
├── contexto/                  # Main package
│   ├── __init__.py
│   ├── solver.py              # Core Engine implementation
│   ├── cognitive_mirrors.py   # Cognitive Mirrors Loop
│   ├── vector_db.py           # Qdrant interface
│   ├── contexto_api.py        # Playwright interface
│   └── utils.py               # Utility functions
├── data/                      # Data files
│   ├── word_list.txt          # Common English words
│   ├── historical_puzzles.json # Past Contexto puzzles
│   └── vector_index/          # Qdrant index files
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_solver.py
│   ├── test_cognitive_mirrors.py
│   ├── test_vector_db.py
│   └── test_contexto_api.py
├── scripts/                   # Utility scripts
│   ├── build_index.py         # Build vector index
│   └── scrape_historical.py   # Scrape historical puzzles
├── crush.py                   # CLI entry point
├── eval.py                    # Evaluation script
├── pyproject.toml             # Poetry configuration
└── README.md                  # Project documentation
```

## Development Workflow

### Feature Development

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement your changes, following the coding standards below.

3. Write tests for your changes:
   ```bash
   poetry run pytest tests/
   ```

4. Submit a pull request with a clear description of your changes.

### Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use type hints for all function parameters and return values
- Document all public functions and classes with docstrings
- Keep functions small and focused on a single responsibility
- Use meaningful variable and function names

### Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Include integration tests for component interactions
- Use pytest fixtures for common test setup

## Component Development Guidelines

### Core Engine (Solver)

- Keep the main loop simple and delegate complex logic to helper functions
- Maintain clear separation between word proposal and selection logic
- Use dependency injection to allow for component mocking in tests

### Cognitive Mirrors Loop

- Implement the reflection logic as a pipeline of discrete steps
- Keep the internal dialogue focused on specific questions
- Document the reasoning patterns used for introspection

### Vector Database (Qdrant)

- Abstract Qdrant-specific implementation details behind a clean interface
- Include error handling for index creation and query failures
- Implement caching to avoid redundant vector calculations

### Contexto API (Playwright)

- Handle website changes gracefully with robust selectors
- Implement rate limiting and backoff strategies
- Mock the API for testing to avoid hitting the actual website

## Performance Optimization

- Profile the application to identify bottlenecks
- Optimize vector operations for CPU efficiency
- Implement caching for frequently accessed data
- Consider batch operations where appropriate

## Release Process

1. Update version in pyproject.toml
2. Update CHANGELOG.md with notable changes
3. Create a new release tag
4. Build and publish the package:
   ```bash
   poetry build
   poetry publish
   ```

## Troubleshooting

### Common Issues

- **Qdrant Index Errors**: Delete the index directory and rebuild
- **Playwright Connection Issues**: Check network connectivity and website changes
- **Memory Usage**: Monitor RAM usage during large batch evaluations
