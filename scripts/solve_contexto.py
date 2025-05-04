#!/usr/bin/env python
"""
Main script for solving Contexto puzzles.

This script provides a user-friendly interface for solving Contexto puzzles
using the Contexto-Crusher system.
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

from contexto.vector_db import VectorDB
from contexto.cognitive_mirrors import CognitiveMirrors
from contexto.contexto_api import ContextoAPI
from contexto.solver import Solver


async def solve_daily_puzzle(
    vector_db_path: str,
    collection_name: str = "words",
    use_docker: bool = False,
    qdrant_url: str = "http://localhost:6333",
    max_turns: int = 50,
    initial_word: Optional[str] = None,
    headless: bool = False,
    batch_size: Optional[int] = None
):
    """
    Solve the daily Contexto puzzle.

    Args:
        vector_db_path: Path to the vector database
        collection_name: Name of the collection in the vector database
        use_docker: Whether to use Qdrant in Docker mode
        qdrant_url: URL of the Qdrant server if using Docker
        max_turns: Maximum number of turns to attempt
        initial_word: Optional starting word
        headless: Whether to run the browser in headless mode
        batch_size: Batch size for embedding generation
    """
    print("=" * 80)
    print("Contexto-Crusher - Daily Puzzle Solver")
    print("=" * 80)

    # Check if the vector database exists
    if not os.path.exists(vector_db_path):
        print(f"Vector database not found at {vector_db_path}")
        print("Please run scripts/build_index.py first to build the vector database")
        return

    # Initialize the vector database
    print(f"Initializing vector database from {vector_db_path}...")
    if use_docker:
        print(f"Using Qdrant in Docker mode with URL: {qdrant_url}")
        vector_db = VectorDB(
            collection_name=collection_name,
            path=vector_db_path,
            use_docker=True,
            url=qdrant_url,
            batch_size=batch_size
        )
    else:
        print("Using Qdrant in local mode")
        vector_db = VectorDB(
            collection_name=collection_name,
            path=vector_db_path,
            batch_size=batch_size
        )

    # Initialize the cognitive mirrors
    print("Initializing cognitive mirrors...")
    cognitive_mirrors = CognitiveMirrors(vector_db)

    # Initialize the Contexto API
    print("Initializing Contexto API...")
    contexto_api = ContextoAPI(headless=headless)

    # Start the Contexto API
    print("Starting Contexto API...")
    await contexto_api.start()

    try:
        # Navigate to the daily puzzle
        print("Navigating to the daily puzzle...")
        await contexto_api.navigate_to_daily()

        # Initialize the solver
        print("Initializing solver...")
        solver = Solver(vector_db, cognitive_mirrors, contexto_api, max_turns=max_turns)

        # Solve the puzzle
        print("\nSolving the puzzle...")
        result = await solver.solve(initial_word=initial_word)

        # Print the result
        if result["solution"]:
            print(f"\n✅ Success! Solved in {result['attempts']} attempts.")
            print(f"Solution: {result['solution']}")
        else:
            print(f"\n❌ Failed to solve the puzzle in {result['attempts']} attempts.")

        print("\nGuess history:")
        for i, (word, rank) in enumerate(result["history"], 1):
            print(f"{i}. Guessed: \"{word}\" → rank {rank}")

    finally:
        # Stop the Contexto API
        print("\nStopping Contexto API...")
        await contexto_api.stop()


async def solve_historical_puzzle(
    vector_db_path: str,
    date: str,
    collection_name: str = "words",
    use_docker: bool = False,
    qdrant_url: str = "http://localhost:6333",
    max_turns: int = 50,
    initial_word: Optional[str] = None,
    headless: bool = False,
    batch_size: Optional[int] = None
):
    """
    Solve a historical Contexto puzzle.

    Args:
        vector_db_path: Path to the vector database
        date: Date of the historical puzzle in YYYY-MM-DD format
        collection_name: Name of the collection in the vector database
        use_docker: Whether to use Qdrant in Docker mode
        qdrant_url: URL of the Qdrant server if using Docker
        max_turns: Maximum number of turns to attempt
        initial_word: Optional starting word
        headless: Whether to run the browser in headless mode
        batch_size: Batch size for embedding generation
    """
    print("=" * 80)
    print(f"Contexto-Crusher - Historical Puzzle Solver ({date})")
    print("=" * 80)

    # Check if the vector database exists
    if not os.path.exists(vector_db_path):
        print(f"Vector database not found at {vector_db_path}")
        print("Please run scripts/build_index.py first to build the vector database")
        return

    # Initialize the vector database
    print(f"Initializing vector database from {vector_db_path}...")
    if use_docker:
        print(f"Using Qdrant in Docker mode with URL: {qdrant_url}")
        vector_db = VectorDB(
            collection_name=collection_name,
            path=vector_db_path,
            use_docker=True,
            url=qdrant_url,
            batch_size=batch_size
        )
    else:
        print("Using Qdrant in local mode")
        vector_db = VectorDB(
            collection_name=collection_name,
            path=vector_db_path,
            batch_size=batch_size
        )

    # Initialize the cognitive mirrors
    print("Initializing cognitive mirrors...")
    cognitive_mirrors = CognitiveMirrors(vector_db)

    # Initialize the Contexto API
    print("Initializing Contexto API...")
    contexto_api = ContextoAPI(headless=headless)

    # Start the Contexto API
    print("Starting Contexto API...")
    await contexto_api.start()

    try:
        # Navigate to the historical puzzle
        print(f"Navigating to the historical puzzle for {date}...")
        await contexto_api.navigate_to_historical(date)

        # Initialize the solver
        print("Initializing solver...")
        solver = Solver(vector_db, cognitive_mirrors, contexto_api, max_turns=max_turns)

        # Solve the puzzle
        print("\nSolving the puzzle...")
        result = await solver.solve(initial_word=initial_word)

        # Print the result
        if result["solution"]:
            print(f"\n✅ Success! Solved in {result['attempts']} attempts.")
            print(f"Solution: {result['solution']}")
        else:
            print(f"\n❌ Failed to solve the puzzle in {result['attempts']} attempts.")

        print("\nGuess history:")
        for i, (word, rank) in enumerate(result["history"], 1):
            print(f"{i}. Guessed: \"{word}\" → rank {rank}")

    finally:
        # Stop the Contexto API
        print("\nStopping Contexto API...")
        await contexto_api.stop()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Solve Contexto puzzles using Contexto-Crusher.")
    parser.add_argument("--vector-db", type=str, default="data/vector_index", help="Path to the vector database")
    parser.add_argument("--collection", type=str, default="words", help="Name of the collection in the vector database")
    parser.add_argument("--use-docker", action="store_true", help="Use Qdrant in Docker instead of local mode")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333", help="URL of the Qdrant server if using Docker")
    parser.add_argument("--max-turns", type=int, default=50, help="Maximum number of turns to attempt")
    parser.add_argument("--initial-word", type=str, help="Optional starting word")
    parser.add_argument("--headless", action="store_true", help="Run the browser in headless mode")
    parser.add_argument("--batch-size", type=int, help="Batch size for embedding generation")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Daily puzzle command
    daily_parser = subparsers.add_parser("daily", help="Solve the daily Contexto puzzle")
    
    # Historical puzzle command
    historical_parser = subparsers.add_parser("historical", help="Solve a historical Contexto puzzle")
    historical_parser.add_argument("date", type=str, help="Date of the historical puzzle in YYYY-MM-DD format")
    
    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate command
    if args.command == "daily":
        asyncio.run(solve_daily_puzzle(
            vector_db_path=args.vector_db,
            collection_name=args.collection,
            use_docker=args.use_docker,
            qdrant_url=args.qdrant_url,
            max_turns=args.max_turns,
            initial_word=args.initial_word,
            headless=args.headless,
            batch_size=args.batch_size
        ))
    elif args.command == "historical":
        asyncio.run(solve_historical_puzzle(
            vector_db_path=args.vector_db,
            date=args.date,
            collection_name=args.collection,
            use_docker=args.use_docker,
            qdrant_url=args.qdrant_url,
            max_turns=args.max_turns,
            initial_word=args.initial_word,
            headless=args.headless,
            batch_size=args.batch_size
        ))


if __name__ == "__main__":
    main()
