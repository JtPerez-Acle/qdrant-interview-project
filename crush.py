#!/usr/bin/env python
"""Command-line interface for solving the daily Contexto puzzle using the curated word list."""

import argparse
import asyncio
import os
import time
import logging
from typing import Optional

from contexto.cognitive_mirrors import CognitiveMirrors
from contexto.contexto_api import ContextoAPI
from contexto.solver import Solver
from contexto.vector_db import VectorDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main(initial_word: Optional[str] = None, max_turns: int = 20, headless: bool = True,
           force_rebuild: bool = False, word_list: str = "data/20k.txt"):
    """Solve the daily Contexto puzzle using the curated word list.

    Args:
        initial_word: Optional starting word
        max_turns: Maximum number of turns to attempt
        headless: Whether to run the browser in headless mode
        force_rebuild: Whether to force rebuild the vector index
        word_list: Path to word list file
    """
    logger.info("Contexto-Crusher 🚀")
    logger.info("-------------------")

    # Initialize components
    logger.info("Initializing vector database...")
    vector_db = VectorDB(collection_name="words_curated", path="./data/vector_index")

    # Initialize the client with force_new if force_rebuild is True
    if force_rebuild:
        logger.warning("Forcing rebuild of vector index...")
        vector_db.initialize_client(force_new=True)

    # Check if the collection exists, if not, build it
    if force_rebuild or not vector_db.collection_exists():
        logger.warning(f"Building index from {word_list}...")
        # Load the model if not already loaded
        if vector_db.model is None:
            vector_db.load_model()
        # Build the index
        vector_db.build_index(word_list)

    logger.info("Initializing cognitive mirrors...")
    cognitive_mirrors = CognitiveMirrors(vector_db)

    logger.info("Initializing browser...")
    contexto_api = ContextoAPI(headless=headless)
    await contexto_api.start()

    logger.info("Navigating to Contexto.me...")
    await contexto_api.navigate_to_daily()

    # Create solver
    solver = Solver(vector_db, cognitive_mirrors, contexto_api, max_turns=max_turns)

    # Solve the puzzle
    logger.info("\nSolving the puzzle...")
    start_time = time.time()
    result = await solver.solve(initial_word=initial_word)
    end_time = time.time()

    # Print results
    logger.info("\nResults:")
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
    logger.info(f"Attempts: {result['attempts']}")

    if result["solution"]:
        logger.info(f"Solution: {result['solution']} 🎉")
    else:
        logger.info("No solution found within the maximum number of turns.")

    logger.info("\nGuess history:")
    for i, (word, rank) in enumerate(result["history"], 1):
        logger.info(f"{i}. Guessed: \"{word}\" → rank {rank}")

    # Close the browser
    await contexto_api.stop()

    # Close the vector database
    vector_db.close()
    logger.info("Vector database closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the daily Contexto puzzle using the curated word list.")
    parser.add_argument("--initial-word", type=str, help="Initial word to start with")
    parser.add_argument("--max-turns", type=int, default=50, help="Maximum number of turns (default: 50)")
    parser.add_argument("--no-headless", action="store_true", help="Run with visible browser (show browser window)")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild the vector index")
    parser.add_argument("--word-list", type=str, default="data/20k.txt", help="Path to word list file")

    args = parser.parse_args()

    # If no initial word is provided, use a good starting word
    initial_word = args.initial_word
    if not initial_word:
        # Choose a balanced, common word as the starting point
        initial_word = "thing"  # A neutral, common word that's often semantically central

    # Print a message if running in non-headless mode
    if args.no_headless:
        print("\nRunning in visible browser mode. You will see the browser window.\n")

    # Run the main function
    asyncio.run(main(
        initial_word=initial_word,
        max_turns=args.max_turns,
        headless=not args.no_headless,
        force_rebuild=args.force_rebuild,
        word_list=args.word_list
    ))
