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

        # No predefined word lists - let the system discover patterns on its own

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

            # Log progress every 5 turns
            if turn % 5 == 0 and turn > 0:
                # Get the best rank so far
                best_word, best_rank = min(self.history, key=lambda x: x[1])
                logger.info(f"Progress after {len(self.history)} guesses: Best word '{best_word}' with rank {best_rank}")

            # Propose candidates
            candidates = self.propose_candidates(k=30)  # Increased from 20 to 30

            # If we have history, use cognitive mirrors with double-loop critique process
            if self.history:
                logger.info("Using cognitive mirrors with double-loop critique process")
                # Process candidates through the cognitive mirrors double-loop
                refined_candidates = await self.cognitive_mirrors.process(candidates, self.history)
                # Use the refined candidates if available
                if refined_candidates:
                    candidates = refined_candidates

            # Make sure we have valid candidates with enhanced filtering
            valid_candidates = self._filter_candidates(candidates)

            # If we have no valid candidates, try a more aggressive search
            if not valid_candidates:
                logger.warning("No valid candidates found, trying fallback strategy...")

                # Try a random sampling from the vector database
                logger.info("Using random sampling from vector database...")
                all_words = self.vector_db.get_all_words()
                # Shuffle the words to get a random sample
                import random
                random.shuffle(all_words)
                # Take the first 200 words and apply enhanced filtering
                pre_filtered = [w for w in all_words[:200] if w.lower() not in self.tried_words_set]
                valid_candidates = self._filter_candidates(pre_filtered)

                # If still no valid candidates, we're stuck
                if not valid_candidates:
                    logger.error("No valid candidates found, giving up.")
                    break

            # Select the best candidate
            best_candidate = self.select_best_candidate(valid_candidates)

            # Add to tried words set (case-insensitive)
            self.tried_words_set.add(best_candidate.lower())

            # Submit the guess
            rank = await self.contexto_api.submit_guess(best_candidate)
            self.history.append((best_candidate, rank))

            # Store the guess result in the vector database for analysis
            try:
                self.vector_db.store_guess_result(best_candidate, rank)

                # Analyze patterns if we have at least 3 guesses
                if len(self.history) >= 3:
                    analysis = self.vector_db.analyze_guess_patterns()
                    if isinstance(analysis, dict) and "error" not in analysis:
                        logger.info(f"Guess pattern analysis: {analysis}")
            except Exception as e:
                logger.error(f"Error storing/analyzing guess result: {e}")

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

        # If we have history, use the best words as search strategies
        if self.history:
            # Sort history by rank (best first)
            sorted_history = sorted(self.history, key=lambda x: x[1])

            # Use the top 3 best words as search strategies
            for i, (word, rank) in enumerate(sorted_history[:3]):
                search_strategies.append((f"best_{i+1}", word))

            # Also try combinations of the best words
            if len(sorted_history) >= 2:
                best_word = sorted_history[0][0]
                second_best = sorted_history[1][0]
                search_strategies.append(("best_combo", f"{best_word} {second_best}"))

        # Add general search strategies based on common semantic fields
        search_strategies.extend([
            ("abstract", "concept"),
            ("concrete", "object"),
            ("action", "action"),
            ("emotion", "feeling"),
            ("time", "time"),
            ("space", "space"),
            ("person", "person"),
            ("place", "place"),
            ("thing", "thing"),
            ("quality", "quality"),
            ("quantity", "quantity")
        ])

        # Try each strategy until we have enough candidates
        strategy_results = {}  # Store results by strategy for analysis

        for strategy_name, query in search_strategies:
            if len(all_candidates) >= k*5:  # Get 5x more candidates than needed for better selection
                break

            try:
                # Search using the current strategy
                search_results = self.vector_db.search(query, limit=k*5)

                # Store results by strategy for logging
                filtered_results = []

                # Filter out words we've already tried (case-insensitive)
                for word, score in search_results:
                    if word and word.lower() not in tried_words and word not in [c[0] for c in all_candidates]:
                        all_candidates.append((word, score))
                        filtered_results.append((word, score))

                # Store filtered results for this strategy
                strategy_results[strategy_name] = filtered_results

                logger.info(f"Strategy '{strategy_name}' found {len(filtered_results)} candidates")

            except Exception as e:
                logger.error(f"Error during {strategy_name} search: {e}")

        # Log the strategies and their top results for analysis
        logger.info("Search strategy results:")
        for strategy, results in strategy_results.items():
            if results:
                top_results = results[:3]  # Show top 3 results
                logger.info(f"  Strategy '{strategy}': {', '.join([f'{word} ({score:.2f})' for word, score in top_results])}")
            else:
                logger.info(f"  Strategy '{strategy}': No results")

        # Sort candidates by score (higher score = better)
        all_candidates.sort(key=lambda x: x[1], reverse=True)

        # Take the top k candidates
        candidates = [word for word, _ in all_candidates[:k]]

        # If we still don't have enough candidates, add some random words from the database
        # This should rarely happen, but we want to be safe
        if len(candidates) < k:
            logger.warning(f"Not enough candidates ({len(candidates)}), adding random words from database")

            # Get all words from the database
            all_words = self.vector_db.get_all_words()

            # Shuffle the words to get a random sample
            import random
            random.shuffle(all_words)

            # Add words that haven't been tried and aren't already in candidates
            for word in all_words:
                if word.lower() not in tried_words and word not in candidates:
                    candidates.append(word)
                    if len(candidates) >= k:
                        break

        # Make sure we have at least one candidate
        if not candidates:
            logger.error("No candidates found. Using a random word from the database.")
            # Get a random word from the database
            all_words = self.vector_db.get_all_words()
            if all_words:
                candidates = [random.choice(all_words)]
            else:
                # If all else fails, use a common word
                candidates = ["the"]

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
                    weight = 20.0 / (rank + 1)
                elif rank < 200:
                    # Somewhat close words get medium weight
                    weight = 10.0 / (rank + 1)
                elif rank < 500:
                    # Distant words get low weight
                    weight = 5.0 / (rank + 1)
                elif rank < 1000:
                    # Very distant words get minimal weight
                    weight = 1.0 / (rank + 1)
                else:
                    # Extremely distant words get negative weight (move away from them)
                    weight = -0.5 / (rank + 1)

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

    def _filter_candidates(self, candidates: List[str]) -> List[str]:
        """Filter candidates to remove abbreviations, very short words, and already tried words.

        Args:
            candidates: List of candidate words to filter

        Returns:
            Filtered list of valid candidates
        """
        if not candidates:
            return []

        # First filter out words we've already tried
        not_tried = [c for c in candidates if c.lower() not in self.tried_words_set]

        # Filter out very short words (less than 3 characters)
        min_length = 3
        length_filtered = [c for c in not_tried if len(c) >= min_length]

        # Filter out likely abbreviations (all uppercase or containing periods)
        abbrev_filtered = []
        for word in length_filtered:
            # Skip words that are all uppercase (likely abbreviations)
            if word.isupper() and len(word) <= 4:
                logger.info(f"Filtering out likely abbreviation: '{word}'")
                continue

            # Skip words with periods (likely abbreviations)
            if '.' in word:
                logger.info(f"Filtering out likely abbreviation with period: '{word}'")
                continue

            # Skip words that are just 2 letters
            if len(word) == 2 and word.isalpha():
                logger.info(f"Filtering out 2-letter word: '{word}'")
                continue

            # Skip words that don't contain at least one vowel (likely not real words)
            if not any(vowel in word.lower() for vowel in 'aeiou'):
                logger.info(f"Filtering out word without vowels: '{word}'")
                continue

            abbrev_filtered.append(word)

        # If filtering removed all candidates, fall back to just filtering tried words
        if not abbrev_filtered and not_tried:
            logger.warning("Filtering removed all candidates, falling back to basic filtering")
            return not_tried

        return abbrev_filtered
