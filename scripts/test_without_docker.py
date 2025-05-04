#!/usr/bin/env python
"""
Simple test script for Contexto-Crusher without Docker.

This script tests the core functionality:
1. Creates a small test dataset
2. Builds a vector index using local Qdrant
3. Tests the solver with a mock target word
"""

import asyncio
import os
import shutil
import sys
import tempfile
from typing import List, Tuple

import numpy as np

# Import our modules
from contexto.vector_db import VectorDB
from contexto.cognitive_mirrors import CognitiveMirrors
from contexto.solver import Solver


class MockContextoAPI:
    """Mock API for testing the solver."""
    
    def __init__(self, target_word: str, vector_db: VectorDB):
        """Initialize with a target word and vector database."""
        self.target_word = target_word
        self.vector_db = vector_db
        self.guesses = []
    
    async def start(self) -> bool:
        """Mock start method."""
        return True
    
    async def stop(self) -> bool:
        """Mock stop method."""
        return True
    
    async def navigate_to_daily(self) -> bool:
        """Mock navigation method."""
        return True
    
    async def submit_guess(self, word: str) -> int:
        """Submit a guess and get the rank."""
        self.guesses.append(word)
        
        if word.lower() == self.target_word.lower():
            return 1
        
        # Calculate similarity-based rank
        try:
            embedding1 = self.vector_db.get_embedding(word)
            embedding2 = self.vector_db.get_embedding(self.target_word)
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            # Convert similarity to rank (higher similarity = lower rank)
            rank = max(2, int(1000 * (1 - similarity)))
            return rank
        except Exception as e:
            print(f"Error calculating rank: {e}")
            return 500  # Default middle rank
    
    async def get_history(self) -> List[Tuple[str, int]]:
        """Get the history of guesses."""
        history = []
        for word in self.guesses:
            rank = await self.submit_guess(word)
            history.append((word, rank))
        return history


def create_test_dataset(temp_dir: str) -> str:
    """Create a small test dataset."""
    test_words = [
        "apple", "banana", "cherry", "date", "elderberry",
        "fig", "grape", "honeydew", "kiwi", "lemon",
        "mango", "nectarine", "orange", "papaya", "quince",
        "raspberry", "strawberry", "tangerine", "watermelon"
    ]
    
    dataset_path = os.path.join(temp_dir, "test_words.txt")
    with open(dataset_path, "w") as f:
        f.write("\n".join(test_words))
    
    print(f"Created test dataset with {len(test_words)} words at {dataset_path}")
    return dataset_path


async def run_test():
    """Run the test pipeline without Docker."""
    # Create a temporary directory for test data
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Create test dataset
        dataset_path = create_test_dataset(temp_dir)
        
        # Initialize VectorDB with local mode
        vector_db = VectorDB(
            collection_name="test_collection",
            path=os.path.join(temp_dir, "vector_index"),
            batch_size=8  # Small batch size for testing
        )
        
        # Build the index
        print("Building vector index...")
        count = vector_db.build_index(dataset_path)
        print(f"Indexed {count} words")
        
        # Choose a target word from the dataset
        target_word = "strawberry"  # A word from our test dataset
        print(f"Target word: {target_word}")
        
        # Initialize components for the solver
        cognitive_mirrors = CognitiveMirrors(vector_db)
        mock_api = MockContextoAPI(target_word, vector_db)
        
        # Create solver
        solver = Solver(vector_db, cognitive_mirrors, mock_api, max_turns=10)
        
        # Solve the puzzle
        print("\nSolving the puzzle...")
        result = await solver.solve()
        
        # Check the result
        if result["solution"] == target_word:
            print(f"✅ Success! Solved in {result['attempts']} attempts.")
            print("\nGuess history:")
            for i, (word, rank) in enumerate(result["history"], 1):
                print(f"{i}. Guessed: \"{word}\" → rank {rank}")
            return True
        else:
            print(f"❌ Failed to solve. Got {result['solution']} instead of {target_word}.")
            print(f"Attempts: {result['attempts']}")
            return False
    
    except Exception as e:
        print(f"Error during test: {e}")
        return False
    
    finally:
        # Clean up
        print("\nCleaning up...")
        print(f"Removing temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Cleanup complete.")


def main():
    """Main function."""
    print("=" * 80)
    print("Contexto-Crusher Simple Test (No Docker)")
    print("=" * 80)
    
    # Run the test pipeline
    success = asyncio.run(run_test())
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
