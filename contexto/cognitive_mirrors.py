"""Cognitive Mirrors implementation for recursive reasoning."""

import random
from typing import List, Optional, Tuple

import numpy as np


class CognitiveMirrors:
    """Recursive reasoning module for refining word guesses."""

    def __init__(self, vector_db, introspection_depth: int = 2):
        """Initialize the cognitive mirrors module.
        
        Args:
            vector_db: VectorDB instance for word embeddings
            introspection_depth: Number of reflection iterations
        """
        self.vector_db = vector_db
        self.introspection_depth = introspection_depth

    def critic(self, candidates: List[str], history: List[Tuple[str, int]]) -> str:
        """Analyze candidates and history to generate reflection.
        
        Args:
            candidates: List of candidate words
            history: List of (word, rank) tuples from previous guesses
            
        Returns:
            Reflection text analyzing patterns and suggesting improvements
        """
        # Generate introspective questions
        questions = self.introspect(history)
        
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
            else:
                # Generic answer for other questions
                answers.append(f"Q: {question}\nA: Based on the current guesses, this is worth exploring.")
        
        # Analyze rank trends
        rank_trend = self._analyze_rank_trend(history)
        
        # Generate reflection text
        reflection = f"Reflection on current search trajectory:\n\n"
        reflection += f"Current candidates: {', '.join(candidates)}\n\n"
        reflection += f"Rank trend analysis: {rank_trend}\n\n"
        reflection += "Introspective analysis:\n"
        reflection += "\n".join(answers)
        reflection += "\n\nSuggested actions:\n"
        
        # Add suggested actions based on the analyses
        if history:
            basin_result = self._analyze_semantic_basin(history)
            if basin_result[0]:
                reflection += f"- Explore more diverse semantic areas\n"
            
            polysemy_result = self._detect_polysemy(history)
            if polysemy_result[0]:
                reflection += f"- Consider alternative meanings of key words\n"
            
            morphology_result = self._analyze_morphology(history)
            if morphology_result[0]:
                reflection += f"- Try different parts of speech (e.g., verbs instead of nouns)\n"
        
        return reflection

    def refine(self, candidates: List[str], reflection: str, history: List[Tuple[str, int]]) -> List[str]:
        """Refine candidates based on reflection.
        
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
        """Generate introspective questions based on guess history.
        
        Args:
            history: List of (word, rank) tuples from previous guesses
            
        Returns:
            List of introspective questions
        """
        questions = [
            "Are we stuck in a local semantic basin?",
            "Do ranks suggest a polysemous cluster we ignored?",
            "Should we pivot word morphology (noun → verb)?"
        ]
        
        # Add more specific questions based on history
        if len(history) >= 3:
            # Check for rank plateaus
            ranks = [rank for _, rank in history[-3:]]
            if max(ranks) - min(ranks) < 100:
                questions.append("Are we making sufficient progress in ranks?")
            
            # Check for semantic similarity
            words = [word for word, _ in history[-3:]]
            questions.append(f"Are '{words[0]}', '{words[1]}', and '{words[2]}' too semantically similar?")
        
        # Add domain-specific questions
        if history:
            questions.append(f"Is there a domain shift we're missing from '{history[0][0]}'?")
        
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
            rank_changes.append(change)
        
        # Calculate average change
        avg_change = sum(rank_changes) / len(rank_changes)
        
        if avg_change > 100:
            return f"Ranks are improving rapidly (average improvement: {avg_change:.1f} ranks per guess). We're on the right track."
        elif avg_change > 0:
            return f"Ranks are improving gradually (average improvement: {avg_change:.1f} ranks per guess). We're making progress but might need to adjust our strategy."
        else:
            return f"Ranks are getting worse (average change: {avg_change:.1f} ranks per guess). We need to reconsider our approach."

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
        
        # Combine all forms
        all_forms = verb_forms + noun_forms + adj_forms
        
        # Filter to keep only valid words (those that exist in our vector DB)
        valid_forms = []
        for form in all_forms:
            try:
                # Check if we can get an embedding for this word
                self.vector_db.get_embedding(form)
                valid_forms.append(form)
            except:
                # Word doesn't exist in our vocabulary
                pass
        
        # If we don't have enough valid forms, search for related words
        if len(valid_forms) < count:
            related_words = self.vector_db.search(word, limit=10)
            valid_forms.extend([w for w, _ in related_words if w not in valid_forms])
        
        return valid_forms[:count]
