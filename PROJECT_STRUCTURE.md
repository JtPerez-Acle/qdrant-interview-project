# Project Structure

This document outlines the directory and file organization for the Contexto-Crusher project.

```
qdrant-project/                # Project root
├── contexto/                  # Main package
│   ├── __init__.py            # Package initialization
│   ├── solver.py              # Core Engine implementation (✅ Implemented)
│   ├── cognitive_mirrors.py   # Cognitive Mirrors Loop (✅ Implemented)
│   ├── vector_db.py           # Qdrant interface (✅ Implemented)
│   └── contexto_api.py        # Playwright interface (✅ Implemented)
│
├── data/                      # Data files
│   ├── word_list.txt          # Common English words
│   └── vector_index/          # Qdrant index files (gitignored)
│
├── tests/                     # Test suite
│   ├── __init__.py            # Test package initialization
│   ├── conftest.py            # Test fixtures and configuration
│   ├── test_solver.py         # Tests for Core Engine (✅ Implemented)
│   ├── test_cognitive_mirrors.py # Tests for Cognitive Mirrors (✅ Implemented)
│   ├── test_vector_db.py      # Tests for Vector Database (✅ Implemented)
│   └── test_contexto_api.py   # Tests for Contexto API (✅ Implemented)
│
├── scripts/                   # Utility scripts
│   └── build_index.py         # Build vector index (✅ Implemented)
│
├── crush.py                   # CLI entry point (✅ Implemented)
├── eval.py                    # Evaluation script (✅ Implemented)
├── requirements.txt           # Project dependencies
├── .gitignore                 # Git ignore file
├── .pre-commit-config.yaml    # Pre-commit hooks
├── LICENSE                    # License file
├── README.md                  # Project documentation
├── ARCHITECTURE.md            # System architecture
├── API.md                     # API documentation
├── DEVELOPMENT.md             # Development guide
├── TESTING.md                 # Testing strategy
├── DEVELOPMENT_PLAN.md        # Development plan
├── PROJECT_STRUCTURE.md       # Project structure
├── MVP_REQUIREMENTS.md        # MVP requirements
├── TECHNICAL_SPEC.md          # Technical specifications
└── COGNITIVE_MIRRORS.md       # Cognitive Mirrors approach
```

## Key Files and Their Purposes

### Core Implementation (All Implemented ✅)

- **solver.py**: Implements the Core Engine that orchestrates the guessing process
- **cognitive_mirrors.py**: Implements the recursive reasoning and reflection capabilities
- **vector_db.py**: Manages the Qdrant vector database and embedding operations
- **contexto_api.py**: Handles interactions with the Contexto.me website using Playwright

### Entry Points (All Implemented ✅)

- **crush.py**: Command-line interface for solving the daily Contexto puzzle
- **eval.py**: Script for evaluating solver performance on historical puzzles

### Data Files

- **word_list.txt**: List of common English words to be embedded (downloaded by build_index.py)
- **vector_index/**: Directory containing the Qdrant vector index (not committed to git)

### Utility Scripts (Implemented ✅)

- **build_index.py**: Script to build the vector index from the word list

### Configuration Files

- **requirements.txt**: Lists project dependencies
- **.gitignore**: Specifies files to be ignored by Git
- **.pre-commit-config.yaml**: Configuration for pre-commit hooks

### Documentation Files

- **README.md**: Project overview, installation, and usage instructions
- **ARCHITECTURE.md**: Detailed system architecture and component interactions
- **API.md**: API documentation for core components
- **DEVELOPMENT.md**: Development setup, workflow, and contribution guidelines
- **TESTING.md**: Testing strategy and procedures
- **DEVELOPMENT_PLAN.md**: Step-by-step development plan with progress tracking
- **PROJECT_STRUCTURE.md**: Directory and file organization (this file)
- **MVP_REQUIREMENTS.md**: Detailed requirements and acceptance criteria
- **TECHNICAL_SPEC.md**: Technical specifications for implementation
- **COGNITIVE_MIRRORS.md**: Detailed explanation of the Cognitive Mirrors approach

## Module Dependencies

```
solver.py
  ├── vector_db.py           # For word embeddings and semantic search
  ├── cognitive_mirrors.py   # For recursive reasoning and candidate refinement
  └── contexto_api.py        # For submitting guesses to Contexto.me

cognitive_mirrors.py
  └── vector_db.py           # For word embeddings and semantic analysis

vector_db.py
  └── (external: sentence-transformers, qdrant-client, numpy)

contexto_api.py
  └── (external: playwright, asyncio)

crush.py
  ├── solver.py              # Core solving logic
  ├── vector_db.py           # For word embeddings
  ├── cognitive_mirrors.py   # For recursive reasoning
  └── contexto_api.py        # For website interaction

eval.py
  ├── solver.py              # Core solving logic
  ├── vector_db.py           # For word embeddings
  ├── cognitive_mirrors.py   # For recursive reasoning
  └── contexto_api.py        # For mock website interaction
```

## Development Workflow

1. Clone the repository
2. Activate the virtual environment: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Download word list and build the vector index: `python scripts/build_index.py --download`
5. Run tests to ensure everything is working: `python -m pytest`
6. Solve the daily Contexto puzzle: `python crush.py`
7. Evaluate performance on historical puzzles: `python eval.py --days 10 --verbose`

## Current Status

We have successfully implemented all core components of the Contexto-Crusher project:

1. ✅ **VectorDB**: Vector database for storing and retrieving word embeddings using Qdrant
2. ✅ **ContextoAPI**: Headless browser interface for interacting with Contexto.me using Playwright
3. ✅ **CognitiveMirrors**: Recursive reasoning module for refining word guesses
4. ✅ **Solver**: Core engine that orchestrates the guessing process

We have also implemented the command-line interface and evaluation framework. All components have comprehensive test coverage and are working correctly.

## Next Steps

1. Performance optimization
2. Algorithm refinement
3. Comprehensive error handling
4. Benchmark evaluation on historical puzzles
5. Ablation studies to measure the impact of the Cognitive Mirrors approach
