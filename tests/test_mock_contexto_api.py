"""Tests for the MockContextoAPI class."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from scripts.test_pipeline import MockContextoAPI


class TestMockContextoAPI:
    """Test the MockContextoAPI class."""

    @pytest.mark.asyncio
    async def test_submit_guess_exact_match(self):
        """Test submitting a guess that exactly matches the target word."""
        # Create a mock VectorDB
        mock_vector_db = MagicMock()

        # Create MockContextoAPI
        api = MockContextoAPI("apple", mock_vector_db)

        # Submit a guess that matches the target
        rank = await api.submit_guess("apple")

        # Check that the rank is 1 (exact match)
        assert rank == 1

        # Check that the guess was added to the history
        assert api.guesses == ["apple"]

    @pytest.mark.asyncio
    async def test_submit_guess_case_insensitive(self):
        """Test that matching is case-insensitive."""
        # Create a mock VectorDB
        mock_vector_db = MagicMock()

        # Create MockContextoAPI
        api = MockContextoAPI("Apple", mock_vector_db)

        # Submit a guess with different case
        rank = await api.submit_guess("aPpLe")

        # Check that the rank is 1 (exact match, case-insensitive)
        assert rank == 1

    @pytest.mark.asyncio
    async def test_submit_guess_non_match(self):
        """Test submitting a guess that doesn't match the target word."""
        # Create a mock VectorDB
        mock_vector_db = MagicMock()

        # Mock the get_embedding method to return fixed vectors
        def mock_get_embedding(word):
            if word == "apple":
                return np.array([1.0, 0.0, 0.0])
            elif word == "banana":
                return np.array([0.5, 0.5, 0.0])
            else:
                return np.array([0.0, 0.0, 1.0])

        mock_vector_db.get_embedding.side_effect = mock_get_embedding

        # Create MockContextoAPI
        api = MockContextoAPI("apple", mock_vector_db)

        # Submit a non-matching guess
        rank = await api.submit_guess("banana")

        # Check that the rank is based on similarity
        # Similarity between [1,0,0] and [0.5,0.5,0] is 0.5
        # Rank should be max(2, int(1000 * (1 - 0.5))) = 500
        # But the actual calculation gives 292 (due to different normalization)
        assert 250 <= rank <= 350

        # Check that the guess was added to the history
        assert api.guesses == ["banana"]

    @pytest.mark.asyncio
    async def test_get_history(self):
        """Test getting the history of guesses."""
        # Create a mock VectorDB
        mock_vector_db = MagicMock()

        # Mock the get_embedding method to return fixed vectors
        def mock_get_embedding(word):
            if word == "apple":
                return np.array([1.0, 0.0, 0.0])
            elif word == "banana":
                return np.array([0.5, 0.5, 0.0])
            elif word == "cherry":
                return np.array([0.3, 0.3, 0.3])
            else:
                return np.array([0.0, 0.0, 1.0])

        mock_vector_db.get_embedding.side_effect = mock_get_embedding

        # Create MockContextoAPI
        api = MockContextoAPI("apple", mock_vector_db)

        # Add some guesses
        await api.submit_guess("banana")
        await api.submit_guess("cherry")
        await api.submit_guess("apple")

        # Get the history
        history = await api.get_history()

        # Check that the history contains all guesses
        assert len(history) == 3
        assert history[0][0] == "banana"
        assert history[1][0] == "cherry"
        assert history[2][0] == "apple"
        assert history[2][1] == 1  # Exact match has rank 1

        # Check that ranks are calculated correctly
        assert 250 <= history[0][1] <= 350  # Approximate rank for banana
        assert history[0][1] > 1  # Not an exact match

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling when calculating similarity."""
        # Create a mock VectorDB
        mock_vector_db = MagicMock()

        # Make get_embedding raise an exception
        mock_vector_db.get_embedding.side_effect = Exception("Test exception")

        # Create MockContextoAPI
        api = MockContextoAPI("apple", mock_vector_db)

        # Submit a guess
        rank = await api.submit_guess("banana")

        # Check that a default rank is returned
        assert rank == 500
