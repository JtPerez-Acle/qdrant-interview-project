#!/usr/bin/env python
"""
End-to-end test script for Contexto-Crusher.

This script tests the entire pipeline:
1. Starts Qdrant in Docker
2. Creates a small test dataset
3. Embeds and uploads the words
4. Tests the solver with a mock target word
5. Cleans up everything at the end
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# Import our modules
from contexto.vector_db import VectorDB
from contexto.cognitive_mirrors import CognitiveMirrors
from contexto.solver import Solver


class MockContextoAPI:
    """Mock API for testing the solver."""

    def __init__(self, target_word: str, vector_db: VectorDB):
        """Initialize with a target word and vector database.

        Args:
            target_word: The word to guess
            vector_db: VectorDB instance for calculating similarity
        """
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
        """Submit a guess and get the rank.

        Args:
            word: Word to guess

        Returns:
            Rank of the guessed word (1 is the target word)
        """
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
        """Get the history of guesses.

        Returns:
            List of (word, rank) tuples
        """
        history = []
        # Calculate ranks without adding to guesses
        for word in self.guesses:
            if word.lower() == self.target_word.lower():
                rank = 1
            else:
                try:
                    embedding1 = self.vector_db.get_embedding(word)
                    embedding2 = self.vector_db.get_embedding(self.target_word)
                    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                    # Convert similarity to rank (higher similarity = lower rank)
                    rank = max(2, int(1000 * (1 - similarity)))
                except Exception as e:
                    print(f"Error calculating rank: {e}")
                    rank = 500  # Default middle rank

            history.append((word, rank))

        return history


def check_docker_installed() -> bool:
    """Check if Docker is installed and running."""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_docker_permissions() -> bool:
    """Check if the user has permissions to use Docker."""
    try:
        # Try to run a simple Docker command that requires daemon access
        subprocess.run(["docker", "info"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def start_qdrant_docker(container_name: str = "qdrant-test") -> bool:
    """Start Qdrant in Docker for testing.

    Args:
        container_name: Name for the Docker container

    Returns:
        True if successful, False otherwise
    """
    # Check if container already exists and remove it
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if container_name in result.stdout:
        print(f"Removing existing container '{container_name}'...")
        subprocess.run(["docker", "rm", "-f", container_name], check=True)

    # Start a new container
    cmd = [
        "docker", "run", "-d", "--name", container_name,
        "-p", "6333:6333", "-p", "6334:6334",
        "qdrant/qdrant"
    ]

    try:
        print("Starting Qdrant in Docker...")
        subprocess.run(cmd, check=True)

        # Wait for Qdrant to be ready
        print("Waiting for Qdrant to be ready...")
        max_retries = 30  # Increased from 10 to 30
        retry_delay = 5   # Increased from 2 to 5 seconds

        for i in range(max_retries):
            try:
                import requests
                # Try the health check endpoint instead of readiness
                response = requests.get("http://localhost:6333/")
                if response.status_code == 200:
                    print("Qdrant is ready!")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                print(f"Error checking Qdrant status: {e}")

            # Show container logs if we're having trouble
            if i > 0 and i % 5 == 0:
                print("\nChecking container logs:")
                try:
                    logs = subprocess.run(
                        ["docker", "logs", container_name, "--tail", "10"],
                        capture_output=True, text=True, check=False
                    )
                    if logs.stdout:
                        print("Container logs:")
                        print(logs.stdout)
                except Exception as e:
                    print(f"Error getting container logs: {e}")

            print(f"Waiting for Qdrant to start (attempt {i+1}/{max_retries}, {retry_delay}s delay)...")
            time.sleep(retry_delay)

        print("\nTimed out waiting for Qdrant to start.")
        print("Possible issues:")
        print("1. Docker might not have enough resources (memory/CPU)")
        print("2. Network port 6333 might be blocked or in use")
        print("3. The container might be failing to initialize properly")

        # Show final container logs
        try:
            logs = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True, text=True, check=False
            )
            if logs.stdout:
                print("\nFull container logs:")
                print(logs.stdout)
        except Exception as e:
            print(f"Error getting container logs: {e}")

        return False

    except subprocess.CalledProcessError as e:
        print(f"Error starting Qdrant container: {e}")
        return False


def stop_qdrant_docker(container_name: str = "qdrant-test") -> bool:
    """Stop and remove the Qdrant Docker container.

    Args:
        container_name: Name of the Docker container

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Stopping and removing container '{container_name}'...")
        subprocess.run(["docker", "rm", "-f", container_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error stopping container: {e}")
        return False


def create_test_dataset(temp_dir: str) -> str:
    """Create a small test dataset.

    Args:
        temp_dir: Directory to create the dataset in

    Returns:
        Path to the created dataset file
    """
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


async def run_test_pipeline(cleanup: bool = True) -> bool:
    """Run the end-to-end test pipeline.

    Args:
        cleanup: Whether to clean up after the test

    Returns:
        True if all tests pass, False otherwise
    """
    # Create a temporary directory for test data
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")

    try:
        # Check if Docker is installed
        if not check_docker_installed():
            print("❌ Docker is not installed. Please install Docker and try again.")
            print("   Visit https://docs.docker.com/get-docker/ for installation instructions.")
            return False

        # Check if user has permissions to use Docker
        if not check_docker_permissions():
            print("❌ Permission denied when trying to access Docker.")
            print("\nTo fix this issue, you have several options:")
            print("1. Run the script with sudo:")
            print("   sudo PYTHONPATH=. python scripts/test_pipeline.py")
            print("\n2. Add your user to the docker group (recommended):")
            print("   sudo usermod -aG docker $USER")
            print("   Then log out and log back in for the changes to take effect.")
            print("\n3. Use rootless Docker:")
            print("   https://docs.docker.com/engine/security/rootless/")
            return False

        # Start Qdrant in Docker
        if not start_qdrant_docker():
            print("❌ Failed to start Qdrant in Docker.")
            return False

        # Create test dataset
        dataset_path = create_test_dataset(temp_dir)

        # Initialize VectorDB with Docker mode
        vector_db = VectorDB(
            collection_name="test_collection",
            path=os.path.join(temp_dir, "vector_index"),
            use_docker=True,
            url="http://localhost:6333",
            batch_size=8  # Small batch size for testing
        )

        # Build the index
        print("Building vector index...")
        count = vector_db.build_index(dataset_path)
        print(f"Indexed {count} words")

        # Choose a target word from the dataset
        with open(dataset_path, "r") as f:
            words = [line.strip() for line in f]
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
            success = True
        else:
            print(f"❌ Failed to solve. Got {result['solution']} instead of {target_word}.")
            print(f"Attempts: {result['attempts']}")
            success = False

        return success

    except Exception as e:
        print(f"Error during test: {e}")
        return False

    finally:
        # Clean up
        if cleanup:
            print("\nCleaning up...")
            stop_qdrant_docker()

            print(f"Removing temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

            print("Cleanup complete.")


async def run_test_pipeline_local(cleanup: bool = True) -> bool:
    """Run the end-to-end test pipeline without Docker.

    Args:
        cleanup: Whether to clean up after the test

    Returns:
        True if all tests pass, False otherwise
    """
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
        with open(dataset_path, "r") as f:
            words = [line.strip() for line in f]
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
            success = True
        else:
            print(f"❌ Failed to solve. Got {result['solution']} instead of {target_word}.")
            print(f"Attempts: {result['attempts']}")
            success = False

        return success

    except Exception as e:
        print(f"Error during test: {e}")
        return False

    finally:
        # Clean up
        if cleanup:
            print("\nCleaning up...")
            print(f"Removing temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("Cleanup complete.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run end-to-end test for Contexto-Crusher.")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up after the test")
    parser.add_argument("--no-docker", action="store_true", help="Run without Docker (local mode)")

    args = parser.parse_args()

    print("=" * 80)
    print("Contexto-Crusher End-to-End Test")
    print("=" * 80)

    # Run the test pipeline
    if args.no_docker:
        print("Running in local mode (without Docker)...")
        success = asyncio.run(run_test_pipeline_local(cleanup=not args.no_cleanup))
    else:
        # Check if Docker is available
        if not check_docker_installed() or not check_docker_permissions():
            print("Docker is not available or you don't have permissions.")
            print("Falling back to local mode...")
            success = asyncio.run(run_test_pipeline_local(cleanup=not args.no_cleanup))
        else:
            print("Running with Docker...")
            success = asyncio.run(run_test_pipeline(cleanup=not args.no_cleanup))

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
