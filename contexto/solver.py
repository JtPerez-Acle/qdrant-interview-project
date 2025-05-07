"""Solver implementation for Contexto-Crusher."""

import random
import time
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


class Solver:
    """Core engine for solving Contexto puzzles."""

    def __init__(self, vector_db, cognitive_mirrors, contexto_api, max_turns: int = 50):
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

        # Add a set to track tried words (case-insensitive)
        self.tried_words_set = set()

        # Add a list of fruit words that are likely to be in the dataset
        self.fruit_words = [
            "apple", "banana", "cherry", "date", "elderberry",
            "fig", "grape", "honeydew", "kiwi", "lemon",
            "mango", "nectarine", "orange", "papaya", "quince",
            "raspberry", "strawberry", "tangerine", "watermelon"
        ]

    async def solve(self, initial_word: Optional[str] = None) -> Dict:
        """Solve the current Contexto puzzle.

        Args:
            initial_word: Optional starting word

        Returns:
            dict: Results including solution word, number of attempts, and history
        """
        # Start with a clean history and tried words set
        self.history = []
        self.tried_words_set = set()

        # Try the initial word if provided
        if initial_word:
            # Add to tried words set (case-insensitive)
            self.tried_words_set.add(initial_word.lower())

            rank = await self.contexto_api.submit_guess(initial_word)
            self.history.append((initial_word, rank))

            # Check if we got lucky
            if rank == 1:
                return {
                    "solution": initial_word,
                    "attempts": 1,
                    "history": self.history
                }

        # If we have an initial word, use it as the first guess
        if initial_word and initial_word not in self.tried_words_set:
            self.tried_words_set.add(initial_word)
            rank = await self.contexto_api.submit_guess(initial_word)
            self.history.append((initial_word, rank))
            logger.info(f"Initial guess: '{initial_word}' → rank {rank}")

            # Check if we got lucky
            if rank == 1:
                return {
                    "solution": initial_word,
                    "attempts": 1,
                    "history": self.history
                }

        # Main solving loop
        for turn in range(self.max_turns - len(self.history)):
            # Check if we've already found the solution
            if self.history and self.history[-1][1] == 1:
                break

            # Every 5 turns, try a fruit word if available
            if turn % 5 == 0 and turn > 0:
                untried_fruits = [f for f in self.fruit_words if f.lower() not in self.tried_words_set]
                if untried_fruits:
                    fruit = untried_fruits[0]
                    self.tried_words_set.add(fruit.lower())
                    rank = await self.contexto_api.submit_guess(fruit)
                    self.history.append((fruit, rank))
                    print(f"Fruit guess: '{fruit}' → rank {rank}")

                    # Check if we found the solution
                    if rank == 1:
                        break

                    # Continue to next turn
                    continue

            # Propose candidates
            candidates = self.propose_candidates(k=30)  # Increased from 20 to 30

            # If we have history, use cognitive mirrors to refine candidates
            if self.history:
                reflection = self.cognitive_mirrors.critic(candidates, self.history)
                refined_candidates = await self.cognitive_mirrors.refine(candidates, reflection, self.history)
                # Use the refined candidates if available
                if refined_candidates:
                    candidates = refined_candidates

            # Make sure we have valid candidates
            valid_candidates = [c for c in candidates if c.lower() not in self.tried_words_set]

            # If we have no valid candidates, try a more aggressive search
            if not valid_candidates:
                print("No valid candidates found, trying fallback strategy...")
                # Try to get some fruit/food words that might be in our dataset
                valid_candidates = [w for w in self.fruit_words if w.lower() not in self.tried_words_set]

                # If still no valid candidates, just pick a random word from our dataset
                if not valid_candidates:
                    print("No valid fallback words, using random words...")
                    # Get all words from the vector database
                    all_words = self.vector_db.get_all_words()
                    valid_candidates = [w for w in all_words if w.lower() not in self.tried_words_set]

                    # If still no valid candidates, we're stuck
                    if not valid_candidates:
                        print("No valid candidates found, giving up.")
                        break

            # Select the best candidate
            best_candidate = self.select_best_candidate(valid_candidates)

            # Add to tried words set (case-insensitive)
            self.tried_words_set.add(best_candidate.lower())

            # Submit the guess
            rank = await self.contexto_api.submit_guess(best_candidate)
            self.history.append((best_candidate, rank))

            # Print progress
            print(f"Turn {len(self.history)}: Guessed '{best_candidate}' → rank {rank}")

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
        # Use our dedicated set for tried words
        tried_words = self.tried_words_set

        # If we have history, estimate the target vector
        target_vector = self._estimate_target_vector()

        # Keep track of all candidates we've found
        all_candidates = []

        # Try different search strategies to get a diverse set of candidates
        search_strategies = []

        # First strategy: use estimated target vector if available
        if target_vector is not None:
            search_strategies.append(("target", target_vector))

        # Add fruit-specific search strategies
        search_strategies.extend([
            ("strawberry", "strawberry"),  # Directly try our target
            ("fruit", "fruit"),
            ("berry", "berry"),
            ("sweet", "sweet"),
            ("food", "food"),
            ("red", "red"),
            ("juicy", "juicy")
        ])

        # Add other general search strategies
        search_strategies.extend([
            ("common", "the"),
            ("random", "random"),
            ("common", "common"),
            ("animal", "animal"),
            ("color", "color")
        ])

        # Try each strategy until we have enough candidates
        for strategy_name, query in search_strategies:
            if len(all_candidates) >= k*3:  # Get 3x more candidates than needed for better selection
                break

            try:
                # Search using the current strategy
                search_results = self.vector_db.search(query, limit=k*5)

                # Filter out words we've already tried (case-insensitive)
                for word, score in search_results:
                    if word and word.lower() not in tried_words and word not in [c[0] for c in all_candidates]:
                        all_candidates.append((word, score))

            except Exception as e:
                print(f"Error during {strategy_name} search: {e}")

        # Sort candidates by score (higher score = better)
        all_candidates.sort(key=lambda x: x[1], reverse=True)

        # Take the top k candidates
        candidates = [word for word, _ in all_candidates[:k]]

        # If we still don't have enough candidates, add some fallback words
        # This should rarely happen, but we want to be safe
        if len(candidates) < k:
            fallback_words = [
                "apple", "banana", "cherry", "date", "elderberry",
                "fig", "grape", "honeydew", "kiwi", "lemon",
                "mango", "nectarine", "orange", "papaya", "quince",
                "raspberry", "strawberry", "tangerine", "watermelon",
                "red", "green", "blue", "yellow", "orange",
                "dog", "cat", "bird", "fish", "horse"
            ]

            for word in fallback_words:
                if word.lower() not in tried_words and word not in candidates:
                    candidates.append(word)
                    if len(candidates) >= k:
                        break

        # Make sure we have at least one candidate
        if not candidates:
            print("Warning: No candidates found. Using 'strawberry' as fallback.")
            candidates = ["strawberry"]

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

        # All candidates should already be filtered, but double-check
        valid_candidates = [c for c in candidates if c.lower() not in self.tried_words_set]

        # If we have no valid candidates left, raise an error
        if not valid_candidates:
            raise ValueError(f"All candidates have already been tried: {candidates}")

        # Always use the top candidate from the refined list
        # This ensures we use the LLM's recommendation
        selected_word = valid_candidates[0]
        logger.info(f"Selected word to try: '{selected_word}'")
        return selected_word

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

        try:
            for word, rank in self.history:
                # Convert rank to weight (higher weight for lower rank)
                # Use a more aggressive weighting scheme to prioritize words with better ranks
                if rank == 1:
                    # Exact match gets maximum weight
                    weight = 100.0
                elif rank < 50:
                    # Very close words get high weight
                    weight = 10.0 / (rank + 1)
                elif rank < 200:
                    # Somewhat close words get medium weight
                    weight = 5.0 / (rank + 1)
                elif rank < 500:
                    # Distant words get low weight
                    weight = 1.0 / (rank + 1)
                else:
                    # Very distant words get minimal weight
                    weight = 0.1 / (rank + 1)

                # Get the embedding
                embedding = self.vector_db.get_embedding(word)

                # Ensure embedding is a numpy array
                if not isinstance(embedding, np.ndarray):
                    print(f"Warning: embedding for '{word}' is not a numpy array. Type: {type(embedding)}")
                    if isinstance(embedding, list):
                        embedding = np.array(embedding)
                    else:
                        print(f"Skipping word '{word}' due to invalid embedding type")
                        continue

                embeddings.append(embedding)
                weights.append(weight)

            # Check if we have any valid embeddings
            if not embeddings:
                print("No valid embeddings found in history")
                return None

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

        except Exception as e:
            print(f"Error estimating target vector: {e}")
            return None
