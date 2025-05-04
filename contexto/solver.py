"""Solver implementation for Contexto-Crusher."""

import random
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class Solver:
    """Core engine for solving Contexto puzzles."""

    def __init__(self, vector_db, cognitive_mirrors, contexto_api, max_turns: int = 20):
        """Initialize the solver with required components.
        
        Args:
            vector_db: VectorDB instance for word embeddings
            cognitive_mirrors: CognitiveMirrors instance for reflection
            contexto_api: ContextoAPI instance for submitting guesses
            max_turns: Maximum number of turns to attempt
        """
        self.vector_db = vector_db
        self.cognitive_mirrors = cognitive_mirrors
        self.contexto_api = contexto_api
        self.max_turns = max_turns
        self.history: List[Tuple[str, int]] = []

    async def solve(self, initial_word: Optional[str] = None) -> Dict:
        """Solve the current Contexto puzzle.
        
        Args:
            initial_word: Optional starting word
            
        Returns:
            dict: Results including solution word, number of attempts, and history
        """
        # Start with a clean history
        self.history = []
        
        # Try the initial word if provided
        if initial_word:
            rank = await self.contexto_api.submit_guess(initial_word)
            self.history.append((initial_word, rank))
            
            # Check if we got lucky
            if rank == 1:
                return {
                    "solution": initial_word,
                    "attempts": 1,
                    "history": self.history
                }
        
        # Main solving loop
        for turn in range(self.max_turns):
            # Check if we've already found the solution
            if self.history and self.history[-1][1] == 1:
                break
                
            # Propose candidates
            candidates = self.propose_candidates(k=10)
            
            # If we have history, use cognitive mirrors to refine candidates
            if self.history:
                reflection = self.cognitive_mirrors.critic(candidates, self.history)
                candidates = self.cognitive_mirrors.refine(candidates, reflection, self.history)
            
            # Select the best candidate
            best_candidate = self.select_best_candidate(candidates)
            
            # Submit the guess
            rank = await self.contexto_api.submit_guess(best_candidate)
            self.history.append((best_candidate, rank))
            
            # Check if we found the solution
            if rank == 1:
                break
        
        # Prepare the result
        solution = None
        if self.history and self.history[-1][1] == 1:
            solution = self.history[-1][0]
            
        return {
            "solution": solution,
            "attempts": len(self.history),
            "history": self.history
        }

    def propose_candidates(self, k: int = 10) -> List[str]:
        """Propose k candidate words based on current state.
        
        Args:
            k: Number of candidates to propose
            
        Returns:
            List of candidate words
        """
        # Get words we've already tried
        tried_words = {word for word, _ in self.history}
        
        # If we have history, estimate the target vector
        target_vector = self._estimate_target_vector()
        
        if target_vector is not None:
            # Search using the estimated target vector
            search_results = self.vector_db.search(target_vector, limit=k*2)
        else:
            # No history, start with common words or random search
            search_results = self.vector_db.search("the", limit=k*2)
        
        # Filter out words we've already tried
        candidates = []
        for word, _ in search_results:
            if word not in tried_words:
                candidates.append(word)
                if len(candidates) >= k:
                    break
        
        # If we don't have enough candidates, add some random ones
        if len(candidates) < k:
            # This is a simplified approach; in a real implementation,
            # we would have a better strategy for generating additional candidates
            additional_results = self.vector_db.search("random", limit=k*2)
            for word, _ in additional_results:
                if word not in tried_words and word not in candidates:
                    candidates.append(word)
                    if len(candidates) >= k:
                        break
        
        return candidates

    def select_best_candidate(self, candidates: List[str]) -> str:
        """Select the best word from candidates to guess next.
        
        Args:
            candidates: List of candidate words
            
        Returns:
            Selected word to guess
        """
        if not candidates:
            raise ValueError("No candidates provided")
        
        # If we have no history, just pick the first candidate
        if not self.history:
            return candidates[0]
        
        # Get the best rank so far
        best_rank = min(rank for _, rank in self.history)
        
        # If we're doing well (rank < 50), focus on the first candidate
        # which should be the most promising one
        if best_rank < 50:
            return candidates[0]
        
        # Otherwise, add some exploration by occasionally picking a random candidate
        if random.random() < 0.2:  # 20% chance of exploration
            return random.choice(candidates)
        else:
            return candidates[0]

    def _estimate_target_vector(self) -> Optional[np.ndarray]:
        """Estimate the target word's vector based on history.
        
        Returns:
            Estimated target vector or None if not enough history
        """
        if not self.history:
            return None
        
        # Get embeddings for all words in history
        embeddings = []
        weights = []
        
        for word, rank in self.history:
            # Convert rank to weight (higher weight for lower rank)
            # This is a simple inverse relationship; more sophisticated
            # weighting schemes could be used
            weight = 1.0 / (rank + 1)
            
            # Get the embedding
            embedding = self.vector_db.get_embedding(word)
            
            embeddings.append(embedding)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return None
            
        normalized_weights = [w / total_weight for w in weights]
        
        # Compute weighted average
        weighted_sum = np.zeros_like(embeddings[0])
        for embedding, weight in zip(embeddings, normalized_weights):
            weighted_sum += embedding * weight
        
        # Normalize the result
        norm = np.linalg.norm(weighted_sum)
        if norm > 0:
            weighted_sum = weighted_sum / norm
        
        return weighted_sum
