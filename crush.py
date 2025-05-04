#!/usr/bin/env python
"""Command-line interface for solving the daily Contexto puzzle."""

import argparse
import asyncio
import os
import time
from typing import Optional

from contexto.cognitive_mirrors import CognitiveMirrors
from contexto.contexto_api import ContextoAPI
from contexto.solver import Solver
from contexto.vector_db import VectorDB


async def main(initial_word: Optional[str] = None, max_turns: int = 20, headless: bool = True):
    """Solve the daily Contexto puzzle.
    
    Args:
        initial_word: Optional starting word
        max_turns: Maximum number of turns to attempt
        headless: Whether to run the browser in headless mode
    """
    print("Contexto-Crusher 🚀")
    print("-------------------")
    
    # Initialize components
    print("Initializing vector database...")
    vector_db = VectorDB(collection_name="words", path="./data/vector_index")
    
    print("Initializing cognitive mirrors...")
    cognitive_mirrors = CognitiveMirrors(vector_db)
    
    print("Initializing browser...")
    contexto_api = ContextoAPI(headless=headless)
    await contexto_api.start()
    
    print("Navigating to Contexto.me...")
    await contexto_api.navigate_to_daily()
    
    # Create solver
    solver = Solver(vector_db, cognitive_mirrors, contexto_api, max_turns=max_turns)
    
    # Solve the puzzle
    print("\nSolving the puzzle...")
    start_time = time.time()
    result = await solver.solve(initial_word=initial_word)
    end_time = time.time()
    
    # Print results
    print("\nResults:")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Attempts: {result['attempts']}")
    
    if result["solution"]:
        print(f"Solution: {result['solution']} 🎉")
    else:
        print("No solution found within the maximum number of turns.")
    
    print("\nGuess history:")
    for i, (word, rank) in enumerate(result["history"], 1):
        print(f"{i}. Guessed: \"{word}\" → rank {rank}")
    
    # Close the browser
    await contexto_api.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the daily Contexto puzzle.")
    parser.add_argument("--initial-word", type=str, help="Initial word to start with")
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum number of turns (default: 20)")
    parser.add_argument("--no-headless", action="store_true", help="Run with visible browser")
    
    args = parser.parse_args()
    
    # Run the main function
    asyncio.run(main(
        initial_word=args.initial_word,
        max_turns=args.max_turns,
        headless=not args.no_headless
    ))
