"""Test fixtures for Contexto-Crusher."""

import os
import shutil
from unittest.mock import Mock

import pytest


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def vector_index_dir(test_data_dir):
    """Return the path to the vector index directory."""
    index_dir = os.path.join(test_data_dir, "vector_index")
    os.makedirs(index_dir, exist_ok=True)
    yield index_dir
    # Clean up after tests
    shutil.rmtree(index_dir, ignore_errors=True)


@pytest.fixture
def mock_contexto_api():
    """Create a mock Contexto API that returns predefined ranks."""
    api = Mock()
    
    # Define a dictionary of words and their ranks
    mock_ranks = {
        "paper": 823,
        "document": 172,
        "book": 45,
        "manuscript": 23,
        "scroll": 5,
        "papyrus": 1,
        # Add more words as needed
    }
    
    # Set up the mock to return ranks from the dictionary
    api.submit_guess = Mock(side_effect=lambda word: mock_ranks.get(word, 1000))
    
    return api


@pytest.fixture
def small_word_list(test_data_dir):
    """Create a small word list for testing."""
    word_list_path = os.path.join(test_data_dir, "small_word_list.txt")
    
    # Create a small word list if it doesn't exist
    if not os.path.exists(word_list_path):
        words = [
            "paper", "document", "book", "manuscript", "scroll", "papyrus",
            "pen", "pencil", "ink", "write", "author", "reader", "library",
            "page", "chapter", "paragraph", "sentence", "word", "letter",
            "novel", "story", "poem", "essay", "article", "journal"
        ]
        
        os.makedirs(os.path.dirname(word_list_path), exist_ok=True)
        with open(word_list_path, "w") as f:
            f.write("\n".join(words))
    
    return word_list_path
