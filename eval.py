#!/usr/bin/env python
"""Evaluation script for Contexto-Crusher."""

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from contexto.cognitive_mirrors import CognitiveMirrors
from contexto.contexto_api import ContextoAPI
from contexto.solver import Solver
from contexto.vector_db import VectorDB


async def evaluate_solver(
    solver: Solver,
    target_words: List[str],
    verbose: bool = False
) -> Dict:
    """Evaluate solver performance on multiple puzzles.
    
    Args:
        solver: Solver instance
        target_words: List of puzzle target words
        verbose: Whether to print detailed results
        
    Returns:
        dict: Evaluation metrics
    """
    results = []
    total_time = 0
    
    for i, target_word in enumerate(target_words, 1):
        if verbose:
            print(f"Puzzle {i}/{len(target_words)}: Target word = {target_word}")
        
        # Mock the contexto_api to return ranks based on the target word
        async def mock_submit_guess(word):
            if word == target_word:
                return 1
            else:
                # Calculate a mock rank based on vector similarity
                embedding1 = solver.vector_db.get_embedding(word)
                embedding2 = solver.vector_db.get_embedding(target_word)
                similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                # Convert similarity to rank (higher similarity = lower rank)
                rank = int(1000 * (1 - similarity))
                return max(2, rank)  # Ensure rank is at least 2 (1 is reserved for the target)
        
        # Replace the submit_guess method with our mock
        solver.contexto_api.submit_guess = mock_submit_guess
        
        # Solve the puzzle
        start_time = time.time()
        result = await solver.solve()
        end_time = time.time()
        
        # Record the result
        solve_time = end_time - start_time
        total_time += solve_time
        
        results.append({
            "target": target_word,
            "solution": result["solution"],
            "attempts": result["attempts"],
            "time": solve_time,
            "history": result["history"]
        })
        
        if verbose:
            print(f"  Solution: {result['solution']}")
            print(f"  Attempts: {result['attempts']}")
            print(f"  Time: {solve_time:.2f} seconds")
            print()
    
    # Calculate metrics
    attempts = [r["attempts"] for r in results]
    times = [r["time"] for r in results]
    
    metrics = {
        "results": results,
        "mean_attempts": statistics.mean(attempts),
        "median_attempts": statistics.median(attempts),
        "p95_attempts": np.percentile(attempts, 95),
        "min_attempts": min(attempts),
        "max_attempts": max(attempts),
        "mean_time": statistics.mean(times),
        "total_time": total_time,
        "success_rate": sum(1 for r in results if r["solution"] is not None) / len(results)
    }
    
    return metrics


async def run_evaluation(
    days: int = 100,
    ablate: bool = False,
    output: str = "results.json",
    verbose: bool = False
):
    """Run the evaluation on historical puzzles.
    
    Args:
        days: Number of historical days to evaluate
        ablate: Whether to run ablation studies
        output: Output file for results
        verbose: Whether to print detailed results
    """
    print("Contexto-Crusher Evaluation 🚀")
    print("-----------------------------")
    
    # Initialize components
    print("Initializing vector database...")
    vector_db = VectorDB(collection_name="words", path="./data/vector_index")
    
    print("Initializing cognitive mirrors...")
    cognitive_mirrors = CognitiveMirrors(vector_db)
    
    print("Initializing browser (mock)...")
    contexto_api = ContextoAPI(headless=True)
    
    # Create solver
    solver = Solver(vector_db, cognitive_mirrors, contexto_api)
    
    # Load historical puzzles
    # In a real implementation, we would load actual historical puzzles
    # For this example, we'll use a small set of sample words
    sample_words = [
        "paper", "book", "computer", "language", "algorithm",
        "science", "mathematics", "physics", "chemistry", "biology"
    ]
    
    # Limit to the requested number of days
    target_words = sample_words[:min(days, len(sample_words))]
    
    # Run the evaluation
    print(f"\nEvaluating solver on {len(target_words)} historical puzzles...")
    metrics = await evaluate_solver(solver, target_words, verbose=verbose)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Mean attempts: {metrics['mean_attempts']:.2f}")
    print(f"Median attempts: {metrics['median_attempts']:.2f}")
    print(f"95th percentile attempts: {metrics['p95_attempts']:.2f}")
    print(f"Min attempts: {metrics['min_attempts']}")
    print(f"Max attempts: {metrics['max_attempts']}")
    print(f"Success rate: {metrics['success_rate'] * 100:.2f}%")
    print(f"Mean time per puzzle: {metrics['mean_time']:.2f} seconds")
    print(f"Total time: {metrics['total_time']:.2f} seconds")
    
    # Save results
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output}")
    
    # Run ablation studies if requested
    if ablate:
        print("\nRunning ablation studies...")
        
        # Ablation 1: Disable Cognitive Mirrors
        print("\nAblation 1: Disable Cognitive Mirrors")
        # Create a dummy cognitive mirrors that doesn't do any refinement
        dummy_cm = CognitiveMirrors(vector_db)
        dummy_cm.refine = lambda candidates, reflection, history: candidates
        
        solver_no_cm = Solver(vector_db, dummy_cm, contexto_api)
        metrics_no_cm = await evaluate_solver(solver_no_cm, target_words, verbose=False)
        
        print(f"Mean attempts (no CM): {metrics_no_cm['mean_attempts']:.2f}")
        print(f"Mean attempts (with CM): {metrics['mean_attempts']:.2f}")
        print(f"Difference: {metrics_no_cm['mean_attempts'] - metrics['mean_attempts']:.2f}")
        
        # Ablation 2: Vary introspection depth
        print("\nAblation 2: Vary Introspection Depth")
        depths = [1, 2, 3]
        depth_results = []
        
        for depth in depths:
            print(f"Testing introspection depth = {depth}")
            cm_depth = CognitiveMirrors(vector_db, introspection_depth=depth)
            solver_depth = Solver(vector_db, cm_depth, contexto_api)
            metrics_depth = await evaluate_solver(solver_depth, target_words, verbose=False)
            depth_results.append((depth, metrics_depth["mean_attempts"]))
            print(f"Mean attempts: {metrics_depth['mean_attempts']:.2f}")
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.bar([str(d) for d, _ in depth_results], [a for _, a in depth_results])
        plt.xlabel("Introspection Depth")
        plt.ylabel("Mean Attempts")
        plt.title("Effect of Introspection Depth on Performance")
        plt.savefig("ablation_results.png")
        print("\nAblation results plot saved to ablation_results.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate solver performance on historical puzzles.")
    parser.add_argument("--days", type=int, default=10, help="Number of historical days to evaluate (default: 10)")
    parser.add_argument("--ablate", action="store_true", help="Run ablation studies")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for results (default: results.json)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    
    args = parser.parse_args()
    
    # Run the evaluation
    asyncio.run(run_evaluation(
        days=args.days,
        ablate=args.ablate,
        output=args.output,
        verbose=args.verbose
    ))
