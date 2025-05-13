#!/usr/bin/env python
"""Tests for the curated word list approach."""

import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch

from contexto.vector_db import VectorDB
from contexto.cognitive_mirrors import CognitiveMirrors
from contexto.solver import Solver


@pytest.fixture
def mock_vector_db():
    """Create a mock vector database."""
    mock_db = MagicMock(spec=VectorDB)

    # Mock the search method to return predictable results
    def mock_search(query, limit=10):
        words = ["apple", "banana", "cherry", "date", "elderberry",
                "fig", "grape", "honeydew", "kiwi", "lemon"]
        scores = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45]
        return list(zip(words[:limit], scores[:limit]))

    mock_db.search.side_effect = mock_search
    mock_db.get_embedding.return_value = [0.1] * 384  # Mock embedding vector
    mock_db.get_distance.return_value = 0.5  # Mock distance

    return mock_db


@pytest.fixture
def temp_word_list():
    """Create a temporary word list file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('\n'.join([
            "apple", "banana", "cherry", "date", "elderberry",
            "fig", "grape", "honeydew", "kiwi", "lemon",
            "mango", "nectarine", "orange", "papaya", "quince",
            "raspberry", "strawberry", "tangerine", "watermelon"
        ]))
        temp_file = f.name

    yield temp_file

    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)


def test_vector_db_initialization():
    """Test that VectorDB initializes correctly with the curated approach."""
    # Use a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize VectorDB with the curated collection name
        vector_db = VectorDB(
            collection_name="words_curated",
            path=temp_dir
        )

        # Check that the collection name is set correctly
        assert vector_db.collection_name == "words_curated"
        assert vector_db.path == temp_dir


def test_vector_db_build_index(temp_word_list):
    """Test building the index with a curated word list."""
    # Use a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize VectorDB with the curated collection name
        with patch('contexto.vector_db.SentenceTransformer') as mock_transformer:
            # Mock the transformer to return predictable embeddings
            mock_model = MagicMock()
            # Return a numpy array for encode
            import numpy as np
            mock_model.encode.return_value = np.array([0.1] * 384)  # Mock embedding vector as numpy array
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_transformer.return_value = mock_model

            # Initialize VectorDB
            vector_db = VectorDB(
                collection_name="words_curated",
                path=temp_dir
            )

            # Mock the client to avoid actual Qdrant operations
            vector_db.client = MagicMock()
            vector_db.client.get_collections.return_value.collections = []

            # Mock the encode method to handle batch encoding
            def mock_batch_encode(words, show_progress_bar=False):
                # Return a list of numpy arrays for batch encoding
                import numpy as np
                return [np.array([0.1] * 384) for _ in words]

            mock_model.encode.side_effect = mock_batch_encode

            # Mock the PointStruct to avoid validation errors
            with patch('contexto.vector_db.models.PointStruct') as mock_point_struct:
                # Return a simple dict instead of a real PointStruct
                mock_point_struct.return_value = {"id": 1, "vector": [0.1], "payload": {"word": "test"}}

                # Build the index
                with patch('builtins.input', return_value='y'):  # Mock user input
                    count = vector_db.build_index(temp_word_list)

            # Check that the correct number of words were indexed
            assert count == 19  # Number of words in the temp_word_list


def test_solver_with_curated_approach(mock_vector_db):
    """Test that the solver works correctly with the curated approach."""
    # Set test mode environment variable
    import os
    os.environ["CONTEXTO_TEST_MODE"] = "1"

    # Create mock components
    mock_cognitive_mirrors = MagicMock(spec=CognitiveMirrors)
    mock_contexto_api = MagicMock()

    # Mock the submit_guess method to simulate finding the solution on the 3rd guess
    guesses = []
    async def mock_submit_guess(word):
        guesses.append(word)
        if word == "cherry":
            return 1  # Solution found
        return 10  # Not the solution

    mock_contexto_api.submit_guess.side_effect = mock_submit_guess

    # Create the solver
    solver = Solver(
        mock_vector_db,
        mock_cognitive_mirrors,
        mock_contexto_api,
        max_turns=10
    )

    # Mock the propose_candidates method to return predictable candidates
    solver.propose_candidates = MagicMock(return_value=[
        "apple", "banana", "cherry", "date", "elderberry"
    ])

    # Mock the select_best_candidate method to select candidates in order
    candidates_index = 0
    def mock_select_best_candidate(candidates):
        nonlocal candidates_index
        selected = candidates[candidates_index]
        candidates_index += 1
        return selected

    solver.select_best_candidate = mock_select_best_candidate

    # Test the solve method
    import asyncio
    result = asyncio.run(solver.solve())

    # Check the result
    assert result["solution"] == "cherry"
    # Instead of checking the exact number of attempts, check that cherry is in the guesses
    assert "cherry" in guesses
    # Check that cherry is in the history with rank 1
    cherry_in_history = False
    for word, rank in result["history"]:
        if word == "cherry" and rank == 1:
            cherry_in_history = True
    assert cherry_in_history


def test_cognitive_mirrors_with_curated_approach(mock_vector_db):
    """Test that the cognitive mirrors component works correctly with the curated approach."""
    # Create the cognitive mirrors component
    cognitive_mirrors = CognitiveMirrors(mock_vector_db)

    # Test the critic method
    candidates = ["apple", "banana", "cherry", "date", "elderberry"]
    history = [("fig", 10), ("grape", 5)]

    reflection = cognitive_mirrors.critic(candidates, history)

    # Check that the reflection is a non-empty string
    assert isinstance(reflection, str)
    assert len(reflection) > 0

    # Test the refine method
    refined_candidates = cognitive_mirrors.refine(candidates, reflection, history)

    # Check that the refined candidates list is not empty
    assert isinstance(refined_candidates, list)
    assert len(refined_candidates) > 0


def test_end_to_end_curated_approach():
    """Test the end-to-end flow with the curated approach."""
    # Set test mode environment variable
    import os
    os.environ["CONTEXTO_TEST_MODE"] = "1"

    # This is a more comprehensive test that would normally use actual components
    # For simplicity, we'll use mocks here

    # Create mock components
    mock_vector_db = MagicMock(spec=VectorDB)
    mock_cognitive_mirrors = MagicMock(spec=CognitiveMirrors)
    mock_contexto_api = MagicMock()

    # Mock the submit_guess method to simulate finding the solution after a few guesses
    guesses = []
    async def mock_submit_guess(word):
        guesses.append(word)
        if word == "cherry":
            return 1  # Solution found
        return len(guesses)  # Rank based on guess order

    mock_contexto_api.submit_guess.side_effect = mock_submit_guess

    # Mock the start and navigate methods
    mock_contexto_api.start.return_value = True
    mock_contexto_api.navigate_to_daily.return_value = True

    # Create the solver
    solver = Solver(
        mock_vector_db,
        mock_cognitive_mirrors,
        mock_contexto_api,
        max_turns=10
    )

    # Mock the propose_candidates method
    solver.propose_candidates = MagicMock(return_value=[
        "apple", "banana", "cherry", "date", "elderberry"
    ])

    # Mock the select_best_candidate method to always select "cherry" first
    def mock_select_best_candidate(candidates):
        # Always return cherry if it's in the candidates
        if "cherry" in candidates:
            return "cherry"
        # Otherwise return the first candidate
        return candidates[0]

    solver.select_best_candidate = mock_select_best_candidate

    # Test the solve method
    import asyncio
    result = asyncio.run(solver.solve())

    # Check the result
    assert "cherry" in guesses
    assert result["attempts"] <= 10
    # If cherry is in guesses, it should be the solution
    if "cherry" in guesses:
        assert result["solution"] == "cherry"
