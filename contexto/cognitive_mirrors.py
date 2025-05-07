"""Cognitive Mirrors implementation for recursive reasoning with the curated approach and LLM integration."""

import random
import logging
from typing import List, Optional, Tuple

import numpy as np

# Import LLM integration
from contexto.llm_integration import LLMIntegration

# Configure logging
logger = logging.getLogger(__name__)


class CognitiveMirrors:
    """Recursive reasoning module for refining word guesses."""

    def __init__(self, vector_db, introspection_depth: int = 2, use_llm: bool = True):
        """Initialize the cognitive mirrors module.

        Args:
            vector_db: VectorDB instance for word embeddings
            introspection_depth: Number of reflection iterations
            use_llm: Whether to use LLM integration (default: True)
        """
        self.vector_db = vector_db
        self.introspection_depth = introspection_depth
        self.use_llm = use_llm

        # Initialize LLM integration if enabled
        if self.use_llm:
            # Pass the vector_db to the LLM integration for semantic similarity calculations
            self.llm = LLMIntegration(vector_db=self.vector_db)
            if self.llm.is_available():
                logger.info("LLM integration enabled for cognitive mirrors with vector database access")
            else:
                logger.warning("LLM integration not available. Using traditional cognitive mirrors only.")
                self.use_llm = False
        else:
            logger.info("LLM integration disabled for cognitive mirrors")

    async def process(self, candidates: List[str], history: List[Tuple[str, int]]) -> List[str]:
        """Process candidates through the cognitive mirrors with double-loop critique.

        Args:
            candidates: List of candidate words
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Refined list of candidate words
        """
        logger.info("Starting cognitive mirrors process with double-loop critique")

        # First loop: Initial reflection and refinement
        logger.info("First cognitive loop: Initial reflection")
        initial_reflection = self.critic(candidates, history)
        logger.info(f"Initial reflection:\n{initial_reflection}")

        # First refinement based on initial reflection
        first_refined_candidates = await self.refine(candidates, initial_reflection, history)

        # Second loop: Meta-reflection on the first refinement
        logger.info("Second cognitive loop: Meta-reflection")
        # Create a meta-reflection that critiques the first refinement
        meta_reflection = self._meta_critic(first_refined_candidates, initial_reflection, history)
        logger.info(f"Meta-reflection:\n{meta_reflection}")

        # Final refinement based on meta-reflection
        final_candidates = await self.refine(first_refined_candidates, meta_reflection, history)

        logger.info("Completed cognitive mirrors double-loop process")
        return final_candidates

    def critic(self, candidates: List[str], history: List[Tuple[str, int]]) -> str:
        """Analyze candidates and history to generate reflection with enhanced rank focus.

        Args:
            candidates: List of candidate words
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Reflection text analyzing patterns and suggesting improvements with rank emphasis
        """
        # Get the best word and rank from history
        best_word = None
        best_rank = float('inf')
        if history:
            best_word, best_rank = min(history, key=lambda x: x[1])

        # Generate introspective questions with rank focus
        questions = self.introspect(history)

        # Add rank-specific questions
        if history:
            if best_rank < 100:
                questions.append(f"How can we find words even closer to the target than '{best_word}' (rank {best_rank})?")
            elif best_rank < 500:
                questions.append(f"What semantic variations of '{best_word}' (rank {best_rank}) might get us closer to the target?")
            else:
                questions.append(f"Should we explore completely different semantic areas than '{best_word}' (rank {best_rank})?")

        # Answer the questions
        answers = []
        for question in questions:
            if "semantic basin" in question.lower():
                is_basin, explanation = self._analyze_semantic_basin(history)
                answers.append(f"Q: {question}\nA: {explanation}")
            elif "polysemy" in question.lower() or "multiple meanings" in question.lower():
                has_polysemy, explanation = self._detect_polysemy(history)
                answers.append(f"Q: {question}\nA: {explanation}")
            elif "morphology" in question.lower() or "part of speech" in question.lower():
                should_shift, explanation = self._analyze_morphology(history)
                answers.append(f"Q: {question}\nA: {explanation}")
            elif "closer to the target" in question or "semantic variations" in question or "different semantic areas" in question:
                # Rank-specific questions
                if best_rank < 100:
                    answers.append(f"Q: {question}\nA: We should focus on subtle variations and synonyms of '{best_word}' since we're very close to the target. Look for words with similar meanings but slightly different connotations or contexts.")
                elif best_rank < 500:
                    answers.append(f"Q: {question}\nA: We should explore the semantic neighborhood of '{best_word}' more thoroughly. Consider related concepts, different forms, or words that share key semantic components.")
                else:
                    answers.append(f"Q: {question}\nA: We should try words from completely different semantic domains since our current best rank is still high. The target word might be in a different conceptual space than we've explored so far.")
            else:
                # Generic answer for other questions
                answers.append(f"Q: {question}\nA: Based on the current guesses and rank information, this is worth exploring.")

        # Analyze rank trends with enhanced focus
        rank_trend = self._analyze_rank_trend(history)

        # Perform detailed rank analysis if we have enough history
        detailed_rank_analysis = "Not enough history for detailed rank analysis."
        if history and len(history) >= 3:
            try:
                detailed_rank_analysis = self._analyze_rank_patterns(history)
            except Exception as e:
                logger.error(f"Error in detailed rank analysis: {e}")

        # Generate reflection text with enhanced rank focus
        reflection = f"RANK-FOCUSED Reflection on current search trajectory:\n\n"

        # Add best rank information prominently
        if best_word:
            reflection += f"BEST WORD SO FAR: '{best_word}' with RANK {best_rank}\n\n"

            # Add rank-based strategy guidance
            if best_rank < 50:
                reflection += f"RANK STRATEGY: We're VERY CLOSE to the target! Focus intensely on words similar to '{best_word}'.\n\n"
            elif best_rank < 100:
                reflection += f"RANK STRATEGY: We're CLOSE to the target! Focus on the semantic area around '{best_word}'.\n\n"
            elif best_rank < 500:
                reflection += f"RANK STRATEGY: We're on the right track. Balance between exploring similar words to '{best_word}' and trying some new areas.\n\n"
            else:
                reflection += f"RANK STRATEGY: We're still far from the target. Prioritize exploration of diverse semantic areas.\n\n"

        reflection += f"Current candidates: {', '.join(candidates[:10])}\n\n"
        reflection += f"RANK TREND ANALYSIS: {rank_trend}\n\n"
        reflection += f"DETAILED RANK ANALYSIS: {detailed_rank_analysis}\n\n"
        reflection += "Introspective analysis:\n"
        reflection += "\n".join(answers)
        reflection += "\n\nSuggested actions (RANK-PRIORITIZED):\n"

        # Add suggested actions based on the analyses with rank prioritization
        if history:
            # First, add rank-specific actions
            if best_rank < 100:
                reflection += f"- HIGH PRIORITY: Focus on subtle variations of '{best_word}' to get even closer to the target\n"
            elif best_rank < 500:
                reflection += f"- MEDIUM PRIORITY: Explore the semantic neighborhood of '{best_word}' more thoroughly\n"
            else:
                reflection += f"- LOW PRIORITY: Try words from completely different semantic domains\n"

            # Then add other actions
            basin_result = self._analyze_semantic_basin(history)
            if basin_result[0]:
                reflection += f"- Explore more diverse semantic areas\n"

            polysemy_result = self._detect_polysemy(history)
            if polysemy_result[0]:
                reflection += f"- Consider alternative meanings of key words\n"

            morphology_result = self._analyze_morphology(history)
            if morphology_result[0]:
                reflection += f"- Try different parts of speech (e.g., verbs instead of nouns)\n"

            # Add penalty for semantic areas with poor ranks
            try:
                poor_areas = self._identify_poor_semantic_areas(history)
                if poor_areas:
                    reflection += f"- AVOID these semantic areas that have yielded poor ranks: {poor_areas}\n"
            except Exception as e:
                logger.error(f"Error identifying poor semantic areas: {e}")

        return reflection

    async def refine(self, candidates: List[str], reflection: str, history: List[Tuple[str, int]]) -> List[str]:
        """Refine candidates based on reflection.

        Args:
            candidates: List of candidate words
            reflection: Reflection text from critic
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Refined list of candidate words
        """
        # Log the original candidates for comparison
        logger.info(f"Original candidates (top 5): {', '.join(candidates[:5])}")

        # If LLM integration is enabled and available, use it to refine candidates
        if self.use_llm and hasattr(self, 'llm') and self.llm.is_available():
            logger.info("Using LLM to refine candidates")
            try:
                # Use LLM to refine candidates
                refined_candidates = await self.llm.refine_candidates(candidates, history)
                if refined_candidates and len(refined_candidates) > 0:
                    top_candidate = refined_candidates[0]
                    logger.info(f"LLM refined candidates. Top candidate: {top_candidate}")

                    # Log the semantic similarity between top candidate and best guess so far
                    if history:
                        best_word, _ = min(history, key=lambda x: x[1])
                        try:
                            if hasattr(self, 'vector_db') and self.vector_db:
                                similarity = self.vector_db.get_similarity(top_candidate, best_word)
                                logger.info(f"Semantic similarity between top candidate '{top_candidate}' and best guess '{best_word}': {similarity:.4f}")
                        except Exception as e:
                            logger.error(f"Error calculating similarity: {e}")

                    # Ensure the top candidate is at the beginning of the list
                    if top_candidate in candidates:
                        # Remove it from its current position
                        candidates.remove(top_candidate)
                    # Add it to the beginning
                    candidates.insert(0, top_candidate)

                    # Log the final candidates list
                    logger.info(f"Final candidates after refinement (top 5): {', '.join(candidates[:5])}")

                    return candidates
                else:
                    logger.warning("LLM returned empty candidates list. Using traditional refinement.")
                    return self._traditional_refine(candidates, reflection, history)
            except Exception as e:
                logger.error(f"Error using LLM to refine candidates: {e}")
                logger.info("Falling back to traditional refinement")
                return self._traditional_refine(candidates, reflection, history)
        else:
            # Use traditional refinement
            return self._traditional_refine(candidates, reflection, history)

    def _traditional_refine(self, candidates: List[str], reflection: str, history: List[Tuple[str, int]]) -> List[str]:
        """Traditional method to refine candidates without LLM.

        Args:
            candidates: List of candidate words
            reflection: Reflection text from critic
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Refined list of candidate words
        """
        refined_candidates = []

        # Extract insights from reflection
        explore_diverse = "diverse semantic areas" in reflection
        consider_alternative = "alternative meanings" in reflection
        try_different_pos = "different parts of speech" in reflection

        # Get the best ranked word from history
        best_word = None
        best_rank = float('inf')
        for word, rank in history:
            if rank < best_rank:
                best_word = word
                best_rank = rank

        # Generate new candidates based on insights
        if explore_diverse:
            # Add some diverse candidates by using a broader search
            diverse_words = self._generate_diverse_candidates(history, 3)
            refined_candidates.extend(diverse_words)

        if consider_alternative and best_word:
            # Add candidates that might represent alternative meanings
            alternative_words = self._generate_alternative_meanings(best_word, 3)
            refined_candidates.extend(alternative_words)

        if try_different_pos and best_word:
            # Add candidates with different parts of speech
            pos_words = self._generate_different_pos(best_word, 3)
            refined_candidates.extend(pos_words)

        # Add some of the original candidates
        num_original = min(3, len(candidates))
        refined_candidates.extend(candidates[:num_original])

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for word in refined_candidates:
            if word not in seen:
                seen.add(word)
                unique_candidates.append(word)

        return unique_candidates

    def introspect(self, history: List[Tuple[str, int]]) -> List[str]:
        """Generate dynamic introspective questions based on guess history with enhanced rank focus.

        Args:
            history: List of (word, rank) tuples from previous guesses

        Returns:
            List of introspective questions dynamically generated based on current state
        """
        questions = []

        # Base questions that are always relevant
        questions.append("Are we stuck in a local semantic basin?")

        # Get the best word and rank from history
        if history:
            best_word, best_rank = min(history, key=lambda x: x[1])

            # Add rank-specific questions based on best rank
            if best_rank < 100:
                questions.append(f"What subtle variations of '{best_word}' (rank {best_rank}) might get us even closer to rank 1?")
            elif best_rank < 500:
                questions.append(f"What semantic field does '{best_word}' (rank {best_rank}) belong to, and what related words might be closer to rank 1?")
            else:
                questions.append(f"Should we explore completely different semantic areas than '{best_word}' (rank {best_rank})?")

            # Add polysemy question with context
            questions.append(f"Could the target word have multiple meanings different from what '{best_word}' suggests?")

            # Add morphology question with context
            questions.append(f"Would a different part of speech than '{best_word}' get us closer to rank 1?")
        else:
            # Generic questions if no history yet
            questions.append("Do ranks suggest a polysemous cluster we ignored?")
            questions.append("Should we pivot word morphology (noun → verb)?")

        # Add more specific questions based on recent history
        if len(history) >= 3:
            # Check for rank plateaus
            ranks = [rank for _, rank in history[-3:]]
            if max(ranks) - min(ranks) < 100:
                questions.append(f"Our recent ranks ({', '.join(str(r) for r in ranks)}) show little variation. Are we making sufficient progress?")

            # Check for rank deterioration
            if all(history[-i][1] > history[-i-1][1] for i in range(1, min(3, len(history)))):
                questions.append("Our recent guesses are getting worse ranks. Should we change our strategy completely?")

            # Check for semantic similarity in recent words
            words = [word for word, _ in history[-3:]]
            questions.append(f"Are '{words[0]}', '{words[1]}', and '{words[2]}' too semantically similar to make progress?")

            # Add question about best improvement
            improvements = []
            for i in range(1, len(history)):
                prev_rank = history[i-1][1]
                curr_rank = history[i][1]
                change = prev_rank - curr_rank
                if change > 0:  # Improvement
                    improvements.append((history[i][0], change))

            if improvements:
                best_improvement = max(improvements, key=lambda x: x[1])
                questions.append(f"What made '{best_improvement[0]}' our best improvement (rank change: -{best_improvement[1]}) and how can we build on that?")

        # Add domain-specific questions based on best and worst guesses
        if len(history) >= 5:
            # Get best and worst guesses
            sorted_history = sorted(history, key=lambda x: x[1])
            best_words = [word for word, _ in sorted_history[:2]]
            worst_words = [word for word, _ in sorted_history[-2:]]

            questions.append(f"What semantic domain connects our best guesses ('{best_words[0]}', '{best_words[1]}') that our worst guesses ('{worst_words[0]}', '{worst_words[1]}') lack?")

            # Add question about semantic clusters
            questions.append("Are there distinct semantic clusters in our guesses that suggest different possible target domains?")

        return questions

    def _analyze_semantic_basin(self, history: List[Tuple[str, int]]) -> Tuple[bool, str]:
        """Analyze if we're stuck in a local semantic basin.

        Args:
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Tuple of (is_in_basin, explanation)
        """
        if len(history) < 3:
            return False, "Not enough history to determine if we're in a semantic basin."

        # Check semantic similarity between recent guesses
        recent_words = [word for word, _ in history[-3:]]

        # Calculate average pairwise distance
        distances = []
        for i in range(len(recent_words)):
            for j in range(i+1, len(recent_words)):
                distance = self.vector_db.get_distance(recent_words[i], recent_words[j])
                distances.append(distance)

        avg_distance = sum(distances) / len(distances) if distances else 0

        # Check if words are very similar (small distance)
        if avg_distance < 0.3:  # Threshold for semantic similarity
            return True, f"We appear to be stuck in a semantic basin. The average distance between recent words is {avg_distance:.2f}, which is quite low. We should explore more diverse areas of the semantic space."
        else:
            return False, f"We don't appear to be stuck in a semantic basin. The average distance between recent words is {avg_distance:.2f}, which indicates reasonable diversity."

    def _detect_polysemy(self, history: List[Tuple[str, int]]) -> Tuple[bool, str]:
        """Detect if we're missing polysemous meanings.

        Args:
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Tuple of (has_polysemy, explanation)
        """
        if len(history) < 4:
            return False, "Not enough history to detect polysemy."

        # Look for significant rank jumps followed by worse ranks
        # This might indicate we briefly touched on a different meaning
        has_jump = False
        jump_word = None
        jump_rank = 0

        for i in range(1, len(history)):
            prev_rank = history[i-1][1]
            curr_rank = history[i][1]

            # Check for significant improvement
            if prev_rank - curr_rank > 100:
                # Check if later guesses got worse again
                if i < len(history) - 1 and history[i+1][1] > curr_rank + 50:
                    has_jump = True
                    jump_word = history[i][0]
                    jump_rank = curr_rank
                    break

        if has_jump:
            return True, f"We may have encountered polysemy. The word '{jump_word}' achieved rank {jump_rank}, but subsequent guesses performed worse. This suggests '{jump_word}' might have captured an alternative meaning that we should explore further."
        else:
            return False, "No clear evidence of polysemy in the current guesses."

    def _analyze_morphology(self, history: List[Tuple[str, int]]) -> Tuple[bool, str]:
        """Analyze if we should shift word morphology.

        Args:
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Tuple of (should_shift, explanation)
        """
        if not history:
            return False, "No history to analyze morphology."

        # This is a simplified analysis that assumes most words are nouns
        # In a real implementation, we would use a POS tagger

        # Check if most words end with common noun suffixes
        noun_suffixes = ["tion", "ment", "ity", "ness", "ance", "ence", "er", "or", "ist"]
        verb_suffixes = ["ate", "ize", "ify", "en", "ing", "ed"]

        noun_count = 0
        verb_count = 0

        for word, _ in history:
            if any(word.endswith(suffix) for suffix in noun_suffixes):
                noun_count += 1
            elif any(word.endswith(suffix) for suffix in verb_suffixes):
                verb_count += 1

        # If we've mostly tried nouns, suggest trying verbs
        if noun_count > verb_count and noun_count >= 2:
            return True, f"We've mostly tried nouns ({noun_count} nouns vs {verb_count} verbs). Consider trying more verbs or adjectives to explore different parts of speech."
        # If we've mostly tried verbs, suggest trying nouns
        elif verb_count > noun_count and verb_count >= 2:
            return True, f"We've mostly tried verbs ({verb_count} verbs vs {noun_count} nouns). Consider trying more nouns or adjectives to explore different parts of speech."
        else:
            return False, "We have a good mix of different parts of speech in our guesses."

    def _analyze_rank_trend(self, history: List[Tuple[str, int]]) -> str:
        """Analyze the trend in ranks.

        Args:
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Analysis of rank trend
        """
        if len(history) < 2:
            return "Not enough history to analyze rank trend."

        # Calculate rank changes
        rank_changes = []
        for i in range(1, len(history)):
            prev_rank = history[i-1][1]
            curr_rank = history[i][1]
            change = prev_rank - curr_rank
            rank_changes.append((history[i][0], change))

        # Calculate average change
        avg_change = sum(change for _, change in rank_changes) / len(rank_changes)

        # Find best and worst changes
        best_change = max(rank_changes, key=lambda x: x[1])
        worst_change = min(rank_changes, key=lambda x: x[1])

        # Analyze recent trend (last 3 guesses)
        recent_trend = "stable"
        if len(rank_changes) >= 3:
            recent_changes = [change for _, change in rank_changes[-3:]]
            if all(change > 0 for change in recent_changes):
                recent_trend = "improving"
            elif all(change < 0 for change in recent_changes):
                recent_trend = "worsening"
            elif sum(recent_changes) > 0:
                recent_trend = "fluctuating but generally improving"
            else:
                recent_trend = "fluctuating but generally worsening"

        # Build detailed analysis
        analysis = []

        # Overall trend
        if avg_change > 100:
            analysis.append(f"Ranks are improving rapidly (average improvement: {avg_change:.1f} ranks per guess).")
        elif avg_change > 0:
            analysis.append(f"Ranks are improving gradually (average improvement: {avg_change:.1f} ranks per guess).")
        else:
            analysis.append(f"Ranks are getting worse (average change: {avg_change:.1f} ranks per guess).")

        # Best and worst changes
        analysis.append(f"Best improvement: '{best_change[0]}' improved by {best_change[1]} ranks.")
        analysis.append(f"Worst change: '{worst_change[0]}' changed by {worst_change[1]} ranks.")

        # Recent trend
        analysis.append(f"Recent trend (last {min(3, len(rank_changes))} guesses): {recent_trend}.")

        # Recommendation
        if avg_change > 0:
            if recent_trend in ["improving", "fluctuating but generally improving"]:
                analysis.append("Recommendation: Continue with the current strategy.")
            else:
                analysis.append("Recommendation: Recent guesses are not as effective. Consider returning to semantic areas of earlier successful guesses.")
        else:
            analysis.append("Recommendation: Change strategy significantly. Try words from completely different semantic fields.")

        return "\n".join(analysis)

    def _generate_diverse_candidates(self, history: List[Tuple[str, int]], count: int) -> List[str]:
        """Generate diverse candidates that are different from history.

        Args:
            history: List of (word, rank) tuples from previous guesses
            count: Number of candidates to generate

        Returns:
            List of diverse candidate words
        """
        if not history:
            return []

        # Get the best word from history
        best_word = min(history, key=lambda x: x[1])[0]

        # Get embedding for the best word
        best_embedding = self.vector_db.get_embedding(best_word)

        # Add some random noise to the embedding to explore different areas
        noise = np.random.normal(0, 0.1, best_embedding.shape)
        diverse_embedding = best_embedding + noise

        # Normalize the embedding
        diverse_embedding = diverse_embedding / np.linalg.norm(diverse_embedding)

        # Search for similar words to this diverse embedding
        diverse_results = self.vector_db.search(diverse_embedding.tolist(), limit=count*2)

        # Filter out words that are already in history
        history_words = {word for word, _ in history}
        diverse_candidates = [word for word, _ in diverse_results if word not in history_words]

        return diverse_candidates[:count]

    def _generate_alternative_meanings(self, word: str, count: int) -> List[str]:
        """Generate words that might represent alternative meanings of a word.

        Args:
            word: Word to find alternative meanings for
            count: Number of alternatives to generate

        Returns:
            List of words with potentially different meanings
        """
        # This is a simplified implementation
        # In a real implementation, we might use a dictionary API or WordNet

        # Get embedding for the word
        embedding = self.vector_db.get_embedding(word)

        # Search for similar words
        similar_words = self.vector_db.search(word, limit=20)

        # Filter to keep only words with moderate similarity (not too similar, not too different)
        # This might help find words with alternative meanings
        alternative_words = []
        for similar_word, score in similar_words:
            if 0.5 < score < 0.8:  # Moderate similarity range
                alternative_words.append(similar_word)

        # If we don't have enough, add some random words from the similar list
        if len(alternative_words) < count:
            remaining = [w for w, _ in similar_words if w not in alternative_words]
            alternative_words.extend(remaining[:count - len(alternative_words)])

        return alternative_words[:count]

    def _generate_different_pos(self, word: str, count: int) -> List[str]:
        """Generate words with different parts of speech.

        Args:
            word: Base word
            count: Number of words to generate

        Returns:
            List of words with different parts of speech
        """
        # This is a simplified implementation
        # In a real implementation, we would use a POS tagger and morphological analyzer

        # Common verb-forming suffixes
        verb_forms = [word + "ing", word + "ed", word + "s", word + "ify", word + "ize"]

        # Common noun-forming suffixes
        noun_forms = [word + "er", word + "or", word + "ion", word + "ment", word + "ness"]

        # Common adjective-forming suffixes
        adj_forms = [word + "ful", word + "less", word + "ish", word + "y", word + "al"]

        # Combine all forms and return the requested number
        all_forms = verb_forms + noun_forms + adj_forms
        return all_forms[:count]

    def _meta_critic(self, candidates: List[str], initial_reflection: str, history: List[Tuple[str, int]]) -> str:
        """Generate a meta-reflection that critiques the first refinement with enhanced rank analysis.

        This is the second loop of the cognitive mirrors process, where we reflect on our
        initial reflection and refinement to further improve our understanding, with a strong
        emphasis on rank information to guide our decision-making.

        Args:
            candidates: List of candidate words after first refinement
            initial_reflection: The reflection text from the first loop
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Meta-reflection text with enhanced rank analysis
        """
        # Get the best word and rank from history
        if not history:
            return "No history available for meta-reflection."

        best_word, best_rank = min(history, key=lambda x: x[1])

        # Get the top candidate after first refinement
        top_candidate = candidates[0] if candidates else None

        # Calculate semantic similarity between top candidate and best word
        similarity = 0.0
        if top_candidate and best_word:
            try:
                similarity = self.vector_db.get_similarity(top_candidate, best_word)
            except Exception as e:
                logger.error(f"Error calculating similarity in meta-critic: {e}")

        # Perform detailed rank analysis
        rank_analysis = self._analyze_rank_patterns(history)

        # Analyze semantic clusters in relation to ranks
        semantic_clusters = self._analyze_semantic_clusters_by_rank(history)

        # Analyze the initial reflection
        reflection_analysis = []

        # Check if the initial reflection mentioned semantic basin
        if "semantic basin" in initial_reflection:
            reflection_analysis.append("The initial reflection identified a potential semantic basin.")

            # Check if the top candidate is semantically diverse from best word
            if similarity < 0.5:
                reflection_analysis.append(f"The top candidate '{top_candidate}' is semantically diverse from the best word '{best_word}' (similarity: {similarity:.2f}), which is good for exploring new areas.")
            else:
                reflection_analysis.append(f"The top candidate '{top_candidate}' is still semantically similar to the best word '{best_word}' (similarity: {similarity:.2f}). We should explore more diverse areas.")

        # Check if the initial reflection mentioned polysemy
        if "polysemy" in initial_reflection or "alternative meanings" in initial_reflection:
            reflection_analysis.append("The initial reflection suggested exploring alternative meanings.")

            # Check if the top candidate might represent an alternative meaning
            if 0.3 < similarity < 0.7:
                reflection_analysis.append(f"The top candidate '{top_candidate}' has moderate similarity to '{best_word}' (similarity: {similarity:.2f}), which might indicate an alternative meaning.")
            else:
                reflection_analysis.append(f"The top candidate '{top_candidate}' is either too similar or too different from '{best_word}' (similarity: {similarity:.2f}). It might not represent an effective exploration of alternative meanings.")

        # Check if the initial reflection mentioned morphology
        if "parts of speech" in initial_reflection or "morphology" in initial_reflection:
            reflection_analysis.append("The initial reflection suggested trying different parts of speech.")

            # This is a simplified check - in a real implementation, we would use a POS tagger
            noun_suffixes = ["tion", "ment", "ity", "ness", "ance", "ence", "er", "or", "ist"]
            verb_suffixes = ["ate", "ize", "ify", "en", "ing", "ed"]

            best_is_noun = any(best_word.endswith(suffix) for suffix in noun_suffixes)
            best_is_verb = any(best_word.endswith(suffix) for suffix in verb_suffixes)

            top_is_noun = top_candidate and any(top_candidate.endswith(suffix) for suffix in noun_suffixes)
            top_is_verb = top_candidate and any(top_candidate.endswith(suffix) for suffix in verb_suffixes)

            if (best_is_noun and top_is_verb) or (best_is_verb and top_is_noun):
                reflection_analysis.append(f"The top candidate '{top_candidate}' appears to have a different part of speech than '{best_word}', which is good for exploring morphological variations.")
            else:
                reflection_analysis.append(f"The top candidate '{top_candidate}' appears to have the same part of speech as '{best_word}'. We might want to try words with different parts of speech.")

        # Add detailed rank analysis
        reflection_analysis.append(f"Detailed rank analysis: {rank_analysis}")

        # Add semantic cluster analysis
        if semantic_clusters:
            reflection_analysis.append(f"Semantic cluster analysis: {semantic_clusters}")

        # Generate meta-reflection with enhanced rank focus
        meta_reflection = "Meta-reflection on the cognitive process with ENHANCED RANK ANALYSIS:\n\n"
        meta_reflection += f"Initial top candidate after first refinement: '{top_candidate}'\n"
        meta_reflection += f"Best word so far: '{best_word}' with rank {best_rank}\n\n"
        meta_reflection += "Analysis of initial reflection:\n"
        meta_reflection += "\n".join(f"- {analysis}" for analysis in reflection_analysis)
        meta_reflection += "\n\nRank-based insights:\n"

        # Add rank-based insights
        meta_reflection += self._generate_rank_based_insights(history, top_candidate)

        meta_reflection += "\n\nSuggested meta-actions (RANK-PRIORITIZED):\n"

        # Add suggested meta-actions based on the analysis with stronger rank emphasis
        if best_rank < 50:
            meta_reflection += "- CRITICAL: We're very close to the target! Focus intensely on fine-tuning within the current semantic area.\n"
            meta_reflection += "- Prioritize words that are slight variations or closely related to our best guesses.\n"
        elif best_rank < 100:
            meta_reflection += "- HIGH PRIORITY: We're getting close! Focus on fine-tuning within the current semantic area.\n"
            meta_reflection += "- Explore words that are semantically similar to our best guesses but with subtle variations.\n"
        elif best_rank < 500:
            meta_reflection += "- MEDIUM PRIORITY: We're on the right track. Balance between exploration and exploitation.\n"
            meta_reflection += "- Consider both similar words to our best guesses and some moderately different options.\n"
        else:
            meta_reflection += "- LOW PRIORITY: We're still far from the target. Prioritize exploration of diverse semantic areas.\n"
            meta_reflection += "- Try words from completely different semantic fields than our current guesses.\n"

        # Add similarity-based suggestions with rank context
        if similarity > 0.7 and best_rank > 200:
            meta_reflection += "- The top candidate is too semantically similar to our best word, but our best rank is still high. Explore more diverse candidates.\n"
        elif similarity < 0.3 and best_rank < 200:
            meta_reflection += "- The top candidate is too semantically different from our best word, which has a good rank. Stay closer to the semantic area of our best guesses.\n"

        # Add trend-based suggestions
        if len(history) >= 5 and all(history[-i][1] > history[-i-1][1] for i in range(1, min(3, len(history)))):
            meta_reflection += "- WARNING: Recent guesses are getting worse. Consider a more radical shift in strategy.\n"
            meta_reflection += "- Avoid the semantic areas of recent guesses that led to poor ranks.\n"

        # Add penalty for semantic areas with poor ranks
        poor_areas = self._identify_poor_semantic_areas(history)
        if poor_areas:
            meta_reflection += f"- AVOID these semantic areas that have yielded poor ranks: {poor_areas}\n"

        return meta_reflection

    def _analyze_rank_patterns(self, history: List[Tuple[str, int]]) -> str:
        """Perform a detailed analysis of rank patterns to guide word selection.

        Args:
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Detailed analysis of rank patterns
        """
        if len(history) < 3:
            return "Not enough history to analyze rank patterns."

        # Sort history by rank (best first)
        sorted_history = sorted(history, key=lambda x: x[1])

        # Calculate rank distribution
        ranks = [rank for _, rank in history]
        min_rank = min(ranks)
        max_rank = max(ranks)
        avg_rank = sum(ranks) / len(ranks)

        # Analyze rank improvements
        improvements = []
        for i in range(1, len(history)):
            prev_rank = history[i-1][1]
            curr_rank = history[i][1]
            change = prev_rank - curr_rank
            if change > 1000:  # Significant improvement
                improvements.append((history[i][0], change))

        # Analyze rank deteriorations
        deteriorations = []
        for i in range(1, len(history)):
            prev_rank = history[i-1][1]
            curr_rank = history[i][1]
            change = curr_rank - prev_rank
            if change > 1000:  # Significant deterioration
                deteriorations.append((history[i][0], change))

        # Build analysis
        analysis = []
        analysis.append(f"Rank range: {min_rank} to {max_rank} (average: {avg_rank:.1f})")

        if improvements:
            analysis.append(f"Significant improvements: {', '.join([f'{word} (-{change})' for word, change in improvements])}")

        if deteriorations:
            analysis.append(f"Significant deteriorations: {', '.join([f'{word} (+{change})' for word, change in deteriorations])}")

        # Analyze if we're getting closer
        recent_ranks = [rank for _, rank in history[-3:]]
        if min(recent_ranks) < min(ranks[:-3]) if len(ranks) > 3 else float('inf'):
            analysis.append("We're making progress with recent guesses.")
        else:
            analysis.append("Recent guesses haven't improved our best rank.")

        return " ".join(analysis)

    def _analyze_semantic_clusters_by_rank(self, history: List[Tuple[str, int]]) -> str:
        """Analyze semantic clusters in relation to their ranks.

        Args:
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Analysis of semantic clusters by rank
        """
        if len(history) < 5:
            return "Not enough history to analyze semantic clusters."

        # Sort history by rank
        sorted_history = sorted(history, key=lambda x: x[1])

        # Try to identify semantic clusters
        clusters = []

        # Get the top 5 words
        top_words = sorted_history[:min(5, len(sorted_history))]

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(top_words)):
            for j in range(i+1, len(top_words)):
                word1, rank1 = top_words[i]
                word2, rank2 = top_words[j]
                try:
                    similarity = self.vector_db.get_similarity(word1, word2)
                    similarities.append((word1, word2, similarity))
                except Exception as e:
                    logger.error(f"Error calculating similarity: {e}")

        # Find highly similar pairs
        similar_pairs = [(w1, w2) for w1, w2, sim in similarities if sim > 0.6]

        # Build clusters based on similar pairs
        if similar_pairs:
            cluster_words = set()
            for w1, w2 in similar_pairs:
                cluster_words.add(w1)
                cluster_words.add(w2)

            if cluster_words:
                avg_rank = sum([rank for word, rank in sorted_history if word in cluster_words]) / len(cluster_words)
                clusters.append((list(cluster_words), avg_rank))

        # Build analysis
        if not clusters:
            return "No clear semantic clusters identified among top-ranked words."

        analysis = []
        for cluster_words, avg_rank in clusters:
            analysis.append(f"Semantic cluster with average rank {avg_rank:.1f}: {', '.join(cluster_words)}")

        return " ".join(analysis)

    def _generate_rank_based_insights(self, history: List[Tuple[str, int]], top_candidate: str) -> str:
        """Generate insights based on rank patterns to guide word selection.

        Args:
            history: List of (word, rank) tuples from previous guesses
            top_candidate: The top candidate word after first refinement

        Returns:
            Rank-based insights
        """
        if not history:
            return "No history available for rank-based insights."

        # Sort history by rank
        sorted_history = sorted(history, key=lambda x: x[1])
        best_word, best_rank = sorted_history[0]

        # Calculate rank improvements over time
        improvements = []
        for i in range(1, len(history)):
            prev_rank = history[i-1][1]
            curr_rank = history[i][1]
            change = prev_rank - curr_rank
            improvements.append(change)

        # Calculate average improvement
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        # Build insights
        insights = []

        # Insight based on best rank
        if best_rank < 50:
            insights.append(f"Our best word '{best_word}' has rank {best_rank}, which is VERY CLOSE to the target. We should focus on words very similar to this.")
        elif best_rank < 100:
            insights.append(f"Our best word '{best_word}' has rank {best_rank}, which is CLOSE to the target. We should focus on this semantic area.")
        elif best_rank < 500:
            insights.append(f"Our best word '{best_word}' has rank {best_rank}, which is SOMEWHAT CLOSE to the target. We should explore this semantic area more.")
        else:
            insights.append(f"Our best word '{best_word}' has rank {best_rank}, which is STILL FAR from the target. We need to explore different semantic areas.")

        # Insight based on improvement trend
        if avg_improvement > 100:
            insights.append(f"We're making good progress with an average rank improvement of {avg_improvement:.1f} per guess.")
        elif avg_improvement > 0:
            insights.append(f"We're making slow progress with an average rank improvement of {avg_improvement:.1f} per guess.")
        else:
            insights.append(f"We're NOT making progress with an average rank change of {avg_improvement:.1f} per guess. We need a new strategy.")

        # Insight based on top candidate
        if top_candidate:
            # Try to predict the rank of the top candidate based on similarity to other words
            predicted_rank = self._predict_candidate_rank(top_candidate, history)
            if predicted_rank:
                insights.append(f"Based on similarity patterns, the top candidate '{top_candidate}' might achieve a rank around {predicted_rank}.")

                if predicted_rank < best_rank:
                    insights.append(f"This would be an improvement over our current best rank of {best_rank}.")
                else:
                    insights.append(f"This would NOT improve our current best rank of {best_rank}. Consider a different candidate.")

        return "\n- ".join([""] + insights)

    def _predict_candidate_rank(self, candidate: str, history: List[Tuple[str, int]]) -> Optional[int]:
        """Predict the potential rank of a candidate based on similarity to previous guesses.

        Args:
            candidate: Candidate word to predict rank for
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Predicted rank or None if prediction isn't possible
        """
        if not history or not candidate:
            return None

        try:
            # Calculate similarities between candidate and all history words
            similarities = []
            for word, rank in history:
                similarity = self.vector_db.get_similarity(candidate, word)
                similarities.append((word, rank, similarity))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[2], reverse=True)

            # Use the top 3 most similar words to predict rank
            top_similarities = similarities[:min(3, len(similarities))]

            # Weighted average based on similarity
            total_weight = sum(sim for _, _, sim in top_similarities)
            if total_weight == 0:
                return None

            predicted_rank = sum(rank * sim / total_weight for _, rank, sim in top_similarities)

            return int(predicted_rank)
        except Exception as e:
            logger.error(f"Error predicting candidate rank: {e}")
            return None

    def _identify_poor_semantic_areas(self, history: List[Tuple[str, int]]) -> str:
        """Identify semantic areas that have yielded poor ranks.

        Args:
            history: List of (word, rank) tuples from previous guesses

        Returns:
            Description of semantic areas to avoid
        """
        if len(history) < 5:
            return ""

        # Sort history by rank (worst first)
        sorted_history = sorted(history, key=lambda x: x[1], reverse=True)

        # Get the worst 3 words
        worst_words = [word for word, _ in sorted_history[:min(3, len(sorted_history))]]

        # Try to find common semantic themes
        common_themes = []

        # This is a simplified implementation
        # In a real implementation, we might use topic modeling or more sophisticated NLP

        # Check for common suffixes
        noun_suffixes = ["tion", "ment", "ity", "ness", "ance", "ence", "er", "or", "ist"]
        verb_suffixes = ["ate", "ize", "ify", "en", "ing", "ed"]
        adj_suffixes = ["ful", "less", "ish", "y", "al"]

        # Count suffix types
        noun_count = sum(1 for word in worst_words if any(word.endswith(suffix) for suffix in noun_suffixes))
        verb_count = sum(1 for word in worst_words if any(word.endswith(suffix) for suffix in verb_suffixes))
        adj_count = sum(1 for word in worst_words if any(word.endswith(suffix) for suffix in adj_suffixes))

        # Identify predominant part of speech
        if noun_count > verb_count and noun_count > adj_count:
            common_themes.append("nouns")
        elif verb_count > noun_count and verb_count > adj_count:
            common_themes.append("verbs")
        elif adj_count > noun_count and adj_count > verb_count:
            common_themes.append("adjectives")

        # Try to identify semantic similarity
        try:
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(worst_words)):
                for j in range(i+1, len(worst_words)):
                    similarity = self.vector_db.get_similarity(worst_words[i], worst_words[j])
                    similarities.append(similarity)

            # Check if words are semantically similar
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            if avg_similarity > 0.5:
                common_themes.append(f"words similar to {', '.join(worst_words)}")
        except Exception as e:
            logger.error(f"Error identifying poor semantic areas: {e}")

        return ", ".join(common_themes) if common_themes else f"words similar to {', '.join(worst_words)}"
