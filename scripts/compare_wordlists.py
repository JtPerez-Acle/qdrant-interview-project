#!/usr/bin/env python
"""Script to compare performance between full and curated word lists.

This script evaluates the performance of the Contexto-Crusher system
using both the full vocabulary and the curated word list.
"""

import argparse
import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Tuple

from contexto.cognitive_mirrors import CognitiveMirrors
from contexto.contexto_api import ContextoAPI
from contexto.solver import Solver
from contexto.vector_db import VectorDB


async def solve_with_wordlist(
    solution: str,
    collection_name: str,
    max_turns: int = 20,
    initial_word: Optional[str] = None
) -> Dict:
    """Solve a puzzle with a specific word list.

    Args:
        solution: The target solution word
        collection_name: Name of the Qdrant collection to use
        max_turns: Maximum number of turns to attempt
        initial_word: Optional starting word

    Returns:
        Dictionary with results
    """
    # Create a mock ContextoAPI that returns ranks based on the solution
    class MockContextoAPI:
        async def start(self):
            return True

        async def stop(self):
            return True

        async def navigate_to_daily(self):
            return True

        async def submit_guess(self, word: str) -> int:
            # If the word matches the solution, return 1
            if word.lower() == solution.lower():
                return 1

            # Otherwise, use the vector database to estimate the rank
            # This is a simplification, but works for testing
            results = vector_db.search(solution, limit=10000)
            words = [w for w, _ in results]
            
            try:
                # Return the rank of the word in the results
                return words.index(word.lower()) + 1
            except ValueError:
                # If the word is not in the results, return a high rank
                return 9999

    # Initialize components
    print(f"Initializing vector database with collection: {collection_name}...")
    vector_db = VectorDB(collection_name=collection_name, path="./data/vector_index")
    
    print("Initializing cognitive mirrors...")
    cognitive_mirrors = CognitiveMirrors(vector_db)
    
    print("Initializing mock Contexto API...")
    contexto_api = MockContextoAPI()
    
    # Create solver
    solver = Solver(vector_db, cognitive_mirrors, contexto_api, max_turns=max_turns)
    
    # Solve the puzzle
    print(f"\nSolving for target word: {solution}...")
    start_time = time.time()
    result = await solver.solve(initial_word=initial_word)
    end_time = time.time()
    
    # Add timing information
    result["time_taken"] = end_time - start_time
    
    return result


async def compare_wordlists(
    solutions: List[str],
    max_turns: int = 20,
    initial_word: Optional[str] = None
) -> Dict:
    """Compare performance between full and curated word lists.

    Args:
        solutions: List of solution words to test
        max_turns: Maximum number of turns to attempt
        initial_word: Optional starting word

    Returns:
        Dictionary with comparison results
    """
    results = {
        "full": [],
        "curated": [],
        "summary": {}
    }
    
    # Test each solution with both word lists
    for solution in solutions:
        print(f"\n{'='*50}")
        print(f"Testing solution: {solution}")
        print(f"{'='*50}")
        
        # Test with full vocabulary
        print("\nTesting with full vocabulary...")
        full_result = await solve_with_wordlist(
            solution=solution,
            collection_name="words",
            max_turns=max_turns,
            initial_word=initial_word
        )
        results["full"].append({
            "solution": solution,
            "attempts": full_result["attempts"],
            "time_taken": full_result["time_taken"],
            "history": full_result["history"]
        })
        
        # Test with curated word list
        print("\nTesting with curated word list...")
        curated_result = await solve_with_wordlist(
            solution=solution,
            collection_name="words_curated",
            max_turns=max_turns,
            initial_word=initial_word
        )
        results["curated"].append({
            "solution": solution,
            "attempts": curated_result["attempts"],
            "time_taken": curated_result["time_taken"],
            "history": curated_result["history"]
        })
        
        # Print comparison
        print("\nComparison:")
        print(f"Full vocabulary: {full_result['attempts']} attempts in {full_result['time_taken']:.2f} seconds")
        print(f"Curated word list: {curated_result['attempts']} attempts in {curated_result['time_taken']:.2f} seconds")
        
        # Calculate improvement
        attempts_diff = full_result["attempts"] - curated_result["attempts"]
        time_diff = full_result["time_taken"] - curated_result["time_taken"]
        
        if attempts_diff > 0:
            print(f"Curated list used {attempts_diff} fewer attempts! 🎉")
        elif attempts_diff < 0:
            print(f"Full vocabulary used {abs(attempts_diff)} fewer attempts.")
        else:
            print("Both used the same number of attempts.")
            
        if time_diff > 0:
            print(f"Curated list was {time_diff:.2f} seconds faster! 🚀")
        elif time_diff < 0:
            print(f"Full vocabulary was {abs(time_diff):.2f} seconds faster.")
        else:
            print("Both took the same amount of time.")
    
    # Calculate summary statistics
    full_attempts = [r["attempts"] for r in results["full"]]
    curated_attempts = [r["attempts"] for r in results["curated"]]
    full_times = [r["time_taken"] for r in results["full"]]
    curated_times = [r["time_taken"] for r in results["curated"]]
    
    results["summary"] = {
        "full": {
            "avg_attempts": sum(full_attempts) / len(full_attempts),
            "avg_time": sum(full_times) / len(full_times),
            "min_attempts": min(full_attempts),
            "max_attempts": max(full_attempts)
        },
        "curated": {
            "avg_attempts": sum(curated_attempts) / len(curated_attempts),
            "avg_time": sum(curated_times) / len(curated_times),
            "min_attempts": min(curated_attempts),
            "max_attempts": max(curated_attempts)
        }
    }
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare performance between full and curated word lists.")
    parser.add_argument("--solutions", type=str, nargs="+", help="Solution words to test")
    parser.add_argument("--historical", type=str, default="data/historical_solutions.json", 
                        help="Path to historical solutions file")
    parser.add_argument("--num-tests", type=int, default=5, 
                        help="Number of historical solutions to test (if --solutions not provided)")
    parser.add_argument("--max-turns", type=int, default=20, 
                        help="Maximum number of turns to attempt")
    parser.add_argument("--initial-word", type=str, 
                        help="Initial word to start with")
    parser.add_argument("--output", type=str, default="results/comparison_results.json", 
                        help="Path to save the results")
    
    args = parser.parse_args()
    
    # Get solutions to test
    solutions = args.solutions
    if not solutions:
        # Try to load from historical solutions
        if os.path.exists(args.historical):
            try:
                with open(args.historical, "r") as f:
                    historical = json.load(f)
                
                # Get the most recent solutions
                solutions = list(historical.values())[-args.num_tests:]
                print(f"Using {len(solutions)} recent historical solutions for testing")
            except Exception as e:
                print(f"Error loading historical solutions: {e}")
                solutions = ["apple", "house", "happy", "computer", "friend"]
        else:
            # Use default solutions
            solutions = ["apple", "house", "happy", "computer", "friend"]
    
    # Run the comparison
    results = asyncio.run(compare_wordlists(
        solutions=solutions,
        max_turns=args.max_turns,
        initial_word=args.initial_word
    ))
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("\nFull vocabulary:")
    print(f"Average attempts: {results['summary']['full']['avg_attempts']:.2f}")
    print(f"Average time: {results['summary']['full']['avg_time']:.2f} seconds")
    print(f"Min attempts: {results['summary']['full']['min_attempts']}")
    print(f"Max attempts: {results['summary']['full']['max_attempts']}")
    
    print("\nCurated word list:")
    print(f"Average attempts: {results['summary']['curated']['avg_attempts']:.2f}")
    print(f"Average time: {results['summary']['curated']['avg_time']:.2f} seconds")
    print(f"Min attempts: {results['summary']['curated']['min_attempts']}")
    print(f"Max attempts: {results['summary']['curated']['max_attempts']}")
    
    # Calculate improvement
    attempts_improvement = results['summary']['full']['avg_attempts'] - results['summary']['curated']['avg_attempts']
    time_improvement = results['summary']['full']['avg_time'] - results['summary']['curated']['avg_time']
    
    print("\nImprovement:")
    if attempts_improvement > 0:
        print(f"Curated list used {attempts_improvement:.2f} fewer attempts on average! 🎉")
    elif attempts_improvement < 0:
        print(f"Full vocabulary used {abs(attempts_improvement):.2f} fewer attempts on average.")
    else:
        print("Both used the same number of attempts on average.")
        
    if time_improvement > 0:
        print(f"Curated list was {time_improvement:.2f} seconds faster on average! 🚀")
    elif time_improvement < 0:
        print(f"Full vocabulary was {abs(time_improvement):.2f} seconds faster on average.")
    else:
        print("Both took the same amount of time on average.")
    
    # Save the results
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
