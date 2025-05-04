"""Tests for the Solver class."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np

# We'll implement this class later
from contexto.solver import Solver


class TestSolver:
    """Test suite for the Solver class."""

    def test_init(self):
        """Test that the Solver class initializes correctly."""
        vector_db = MagicMock()
        cognitive_mirrors = MagicMock()
        contexto_api = MagicMock()

        solver = Solver(vector_db, cognitive_mirrors, contexto_api, max_turns=20)

        assert solver.vector_db == vector_db
        assert solver.cognitive_mirrors == cognitive_mirrors
        assert solver.contexto_api == contexto_api
        assert solver.max_turns == 20
        assert solver.history == []

    @pytest.mark.asyncio
    async def test_solve_success(self):
        """Test solving a puzzle successfully."""
        # Mock dependencies
        vector_db = MagicMock()
        cognitive_mirrors = MagicMock()
        contexto_api = AsyncMock()

        # Set up the mocks
        vector_db.search.return_value = [("close", 0.9), ("far", 0.8), ("target", 0.7)]
        cognitive_mirrors.critic.return_value = "Test reflection"
        cognitive_mirrors.refine.return_value = ["target", "refined1", "refined2"]

        # Mock the submit_guess method to return rank 100 for first guess, then 1 for "target"
        call_count = 0
        async def mock_submit_guess(word):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 100  # First guess is not correct
            else:
                return 1  # Second guess is correct
        contexto_api.submit_guess.side_effect = mock_submit_guess

        # Create the solver
        solver = Solver(vector_db, cognitive_mirrors, contexto_api)

        # Solve the puzzle
        result = await solver.solve()

        # Check the result
        assert result["solution"] == "target"
        assert result["attempts"] == 2
        assert len(result["history"]) == 2
        assert result["history"][1] == ("target", 1)

        # Verify method calls
        vector_db.search.assert_called()
        cognitive_mirrors.critic.assert_called()
        cognitive_mirrors.refine.assert_called()
        contexto_api.submit_guess.assert_called_with("target")

    @pytest.mark.asyncio
    async def test_solve_multiple_attempts(self):
        """Test solving a puzzle with multiple attempts."""
        # Mock dependencies
        vector_db = MagicMock()
        cognitive_mirrors = MagicMock()
        contexto_api = AsyncMock()

        # Set up the mocks
        vector_db.search.return_value = [("first", 0.9), ("second", 0.8), ("target", 0.7)]

        # Mock the refine method to return different candidates in each call
        refine_calls = 0
        def mock_refine(candidates, reflection, history):
            nonlocal refine_calls
            refine_calls += 1
            if refine_calls == 1:
                return ["second", "refined1", "refined2"]
            else:
                return ["target", "refined3", "refined4"]
        cognitive_mirrors.refine.side_effect = mock_refine

        # Mock the submit_guess method to return different ranks
        async def mock_submit_guess(word):
            if word == "first":
                return 100
            elif word == "second":
                return 50
            elif word == "target":
                return 1
            else:
                return 200
        contexto_api.submit_guess.side_effect = mock_submit_guess

        # Create the solver
        solver = Solver(vector_db, cognitive_mirrors, contexto_api)

        # Solve the puzzle
        result = await solver.solve()

        # Check the result
        assert result["solution"] == "target"
        assert result["attempts"] == 3
        assert len(result["history"]) == 3
        assert result["history"][0] == ("first", 100)
        assert result["history"][1] == ("second", 50)
        assert result["history"][2] == ("target", 1)

    @pytest.mark.asyncio
    async def test_solve_max_turns(self):
        """Test that solving stops after max_turns."""
        # Mock dependencies
        vector_db = MagicMock()
        cognitive_mirrors = MagicMock()
        contexto_api = AsyncMock()

        # Set up the mocks
        vector_db.search.return_value = [("guess1", 0.9), ("guess2", 0.8), ("guess3", 0.7)]
        cognitive_mirrors.refine.return_value = ["guess4", "guess5", "guess6"]

        # Mock the submit_guess method to always return a rank > 1
        contexto_api.submit_guess.return_value = 100

        # Create the solver with a small max_turns
        solver = Solver(vector_db, cognitive_mirrors, contexto_api, max_turns=3)

        # Solve the puzzle
        result = await solver.solve()

        # Check the result
        assert result["solution"] is None  # No solution found
        assert result["attempts"] == 3
        assert len(result["history"]) == 3

        # Verify that submit_guess was called exactly 3 times
        assert contexto_api.submit_guess.call_count == 3

    def test_propose_candidates(self):
        """Test proposing candidate words."""
        # Mock dependencies
        vector_db = MagicMock()
        cognitive_mirrors = MagicMock()
        contexto_api = MagicMock()

        # Set up the mocks
        vector_db.search.return_value = [("word1", 0.9), ("word2", 0.8), ("word3", 0.7), ("word4", 0.6), ("word5", 0.5), ("word6", 0.4)]

        # Create the solver
        solver = Solver(vector_db, cognitive_mirrors, contexto_api)

        # Test with no history
        candidates = solver.propose_candidates(k=3)
        assert len(candidates) == 3
        assert "word1" in candidates
        assert "word2" in candidates
        assert "word3" in candidates

        # Test with some history
        solver.history = [("word1", 100), ("word4", 50)]
        candidates = solver.propose_candidates(k=3)
        assert len(candidates) == 3
        assert "word1" not in candidates  # Should not include words already guessed

    def test_select_best_candidate(self):
        """Test selecting the best candidate."""
        # Mock dependencies
        vector_db = MagicMock()
        cognitive_mirrors = MagicMock()
        contexto_api = MagicMock()

        # Create the solver
        solver = Solver(vector_db, cognitive_mirrors, contexto_api)

        # Test with no history
        candidates = ["word1", "word2", "word3"]
        best = solver.select_best_candidate(candidates)
        assert best in candidates

        # Test with some history
        solver.history = [("word4", 100), ("word5", 50)]
        best = solver.select_best_candidate(candidates)
        assert best in candidates

    def test_estimate_target_vector(self):
        """Test estimating the target vector."""
        # Mock dependencies
        vector_db = MagicMock()
        cognitive_mirrors = MagicMock()
        contexto_api = MagicMock()

        # Set up the mocks
        vector_db.get_embedding.return_value = np.array([0.1, 0.2, 0.3])

        # Create the solver
        solver = Solver(vector_db, cognitive_mirrors, contexto_api)

        # Test with no history
        vector = solver._estimate_target_vector()
        assert vector is None

        # Test with some history
        solver.history = [("word1", 100), ("word2", 50)]
        vector = solver._estimate_target_vector()
        assert vector is not None
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (3,)  # Should match the shape of the mock embeddings
