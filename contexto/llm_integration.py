"""LLM integration for Contexto-Crusher using OpenAI's GPT models."""

import os
import logging
import json
from typing import List, Dict, Tuple, Optional
import openai
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


class LLMIntegration:
    """LLM integration for enhancing word selection in Contexto-Crusher."""

    def __init__(self, model: str = "gpt-4o"):
        """Initialize the LLM integration.

        Args:
            model: The OpenAI model to use (default: gpt-4o)
        """
        self.model = model
        self.api_key = openai.api_key

        if not self.api_key:
            logger.warning("OpenAI API key not found. LLM integration will not work.")
            logger.warning("Please set the OPENAI_API_KEY environment variable in a .env file.")
        else:
            logger.info(f"LLM integration initialized with model: {model}")

    def is_available(self) -> bool:
        """Check if the LLM integration is available.

        Returns:
            True if the API key is set, False otherwise
        """
        return bool(self.api_key)

    async def analyze_guesses(
        self,
        candidates: List[str],
        history: List[Tuple[str, int]]
    ) -> Dict:
        """Analyze previous guesses and suggest the best next word.

        Args:
            candidates: List of candidate words to choose from
            history: List of (word, rank) tuples representing previous guesses

        Returns:
            Dictionary with analysis results and recommended next word
        """
        if not self.is_available():
            logger.warning("LLM integration not available. Skipping analysis.")
            return {
                "recommendation": candidates[0] if candidates else None,
                "reasoning": "LLM integration not available. Using default selection.",
                "analysis": {}
            }

        # Get the best ranked word from history
        best_word = None
        best_rank = float('inf')
        for word, rank in history:
            if rank < best_rank:
                best_word = word
                best_rank = rank

        # Sort history by rank
        sorted_history = sorted(history, key=lambda x: x[1])

        # Format the sorted history for the prompt
        history_text = "\n".join([f"- '{word}' → rank {rank}" for word, rank in sorted_history])

        # Format the candidates for the prompt
        candidates_text = ", ".join([f"'{word}'" for word in candidates])

        # Create the prompt
        prompt = f"""You are an expert at solving Contexto.me word puzzles.

In Contexto, each guess is ranked based on how semantically close it is to the hidden target word. The lower the rank, the closer the word is to the target (rank 1 is the target word).

Previous guesses (sorted by rank, closest first):
{history_text}

The best guess so far is '{best_word}' with rank {best_rank}.

Based on these guesses, analyze the semantic patterns and identify the likely domain or concept of the target word.

Think about:
1. What semantic field or category might the target word belong to?
2. Is the target word likely to be a noun, verb, adjective, or another part of speech?
3. What concepts are suggested by the words with lower ranks?
4. Are there any patterns in the rankings that suggest a particular direction?

Available candidates for the next guess:
{candidates_text}

IMPORTANT: Your recommendation MUST be one of the candidate words listed above EXACTLY as written.
Do not suggest words that are not in this list. Do not modify or change the spelling of any candidate word.

Please provide:
1. A detailed analysis of the previous guesses and what they suggest about the target word
2. Identification of any semantic patterns or clusters
3. Your recommendation for the best next word to guess from the candidates list (must be one of the words listed above)
4. A clear explanation of your reasoning

Format your response as JSON with the following structure:
{{
  "analysis": {{
    "patterns": "Identified semantic patterns",
    "domain": "Likely domain or concept",
    "observations": "Additional observations"
  }},
  "recommendation": "your_recommended_word",
  "reasoning": "Detailed explanation for your recommendation"
}}
"""

        try:
            # Call the OpenAI API using the new client format (openai >= 1.0.0)
            # Note: The standard OpenAI client is not async, so we need to use it synchronously
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at solving Contexto.me word puzzles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            # Extract the response content
            content = response.choices[0].message.content

            # Parse the JSON response
            result = json.loads(content)

            # Ensure the recommendation is in the candidates list
            recommendation = result.get("recommendation")
            if not recommendation or recommendation not in candidates:
                logger.warning(f"LLM recommended '{recommendation}' which is not in the candidates list. Finding closest match.")

                # Try to find the closest match in the candidates list
                if recommendation:
                    # Check if any candidate contains the recommendation as a substring
                    substring_matches = [c for c in candidates if recommendation.lower() in c.lower()]
                    if substring_matches:
                        result["recommendation"] = substring_matches[0]
                        logger.info(f"Found substring match: '{substring_matches[0]}'")
                    else:
                        # Otherwise, use the first candidate
                        result["recommendation"] = candidates[0] if candidates else None
                        logger.warning(f"Using first candidate: '{result['recommendation']}'")
                else:
                    result["recommendation"] = candidates[0] if candidates else None
                    logger.warning(f"No recommendation provided. Using first candidate: '{result['recommendation']}'")

                # Update the reasoning to reflect the change
                result["reasoning"] = f"Original recommendation was not in candidates list. Using '{result['recommendation']}' instead. " + result.get("reasoning", "")

            return result

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {
                "recommendation": candidates[0] if candidates else None,
                "reasoning": f"Error calling OpenAI API: {e}",
                "analysis": {}
            }

    async def refine_candidates(
        self,
        candidates: List[str],
        history: List[Tuple[str, int]],
        target_embedding: Optional[List[float]] = None
    ) -> List[str]:
        """Refine the list of candidates based on previous guesses.

        Args:
            candidates: List of candidate words to refine
            history: List of (word, rank) tuples representing previous guesses
            target_embedding: Optional estimated target embedding

        Returns:
            Refined list of candidates
        """
        if not self.is_available() or not candidates:
            return candidates

        try:
            # Get the analysis and recommendation
            result = await self.analyze_guesses(candidates, history)

            # Get the recommended word
            recommended = result.get("recommendation")

            if recommended and recommended in candidates:
                # Move the recommended word to the front of the list
                candidates.remove(recommended)
                candidates.insert(0, recommended)

                logger.info(f"LLM recommended word: {recommended}")
                logger.info(f"Reasoning: {result.get('reasoning', 'No reasoning provided')}")

            return candidates

        except Exception as e:
            logger.error(f"Error refining candidates: {e}")
            return candidates
