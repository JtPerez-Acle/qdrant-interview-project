#!/usr/bin/env python
"""Script to build the vector index for Contexto-Crusher."""

import argparse
import os
import time
from typing import List

import requests
from tqdm import tqdm

from contexto.vector_db import VectorDB


def download_word_list(url: str, output_path: str) -> int:
    """Download a word list from a URL.

    Args:
        url: URL to download from
        output_path: Path to save the word list

    Returns:
        Number of words downloaded
    """
    print(f"Downloading word list from {url}...")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download the word list
    response = requests.get(url)
    response.raise_for_status()

    # Parse the word list (assuming one word per line)
    words = response.text.splitlines()

    # Filter out empty lines and non-alphabetic words
    words = [word.strip().lower() for word in words if word.strip().isalpha()]

    # Remove duplicates
    words = list(set(words))

    # Save the word list
    with open(output_path, "w") as f:
        f.write("\n".join(words))

    print(f"Downloaded {len(words)} words to {output_path}")
    return len(words)


def build_index(word_list_path: str, vector_db: VectorDB) -> int:
    """Build the vector index from a word list.

    Args:
        word_list_path: Path to the word list
        vector_db: VectorDB instance

    Returns:
        Number of words indexed
    """
    print(f"Building vector index from {word_list_path}...")
    print("This process will:")
    print("1. Load the sentence transformer model")
    print("2. Read all words from the word list")
    print("3. Create embeddings for each word (this may take a while)")
    print("4. Store the embeddings in the Qdrant vector database")
    print("\nStarting the process now...\n")

    start_time = time.time()
    count = vector_db.build_index(word_list_path)
    end_time = time.time()

    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    print(f"\nIndexing complete!")
    print(f"Indexed {count} words in {minutes} minutes and {seconds} seconds")
    print(f"Average time per word: {total_time / count:.4f} seconds")
    print(f"Vector index stored at: {vector_db.path}")
    return count


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build the vector index for Contexto-Crusher.")
    parser.add_argument("--word-list", type=str, default="data/word_list.txt", help="Path to the word list")
    parser.add_argument("--download", action="store_true", help="Download the word list")
    parser.add_argument("--url", type=str, default="https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt", help="URL to download the word list from")
    parser.add_argument("--index-path", type=str, default="data/vector_index", help="Path to store the vector index")
    parser.add_argument("--use-docker", action="store_true", help="Use Qdrant in Docker instead of local mode")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333", help="URL of the Qdrant server if using Docker")
    parser.add_argument("--force-local", action="store_true", help="Force using local mode even for large collections")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding generation (default: 64 for GPU, 32 for CPU)")

    args = parser.parse_args()

    # Download the word list if requested
    if args.download:
        download_word_list(args.url, args.word_list)

    # Check if the word list exists
    if not os.path.exists(args.word_list):
        print(f"Word list not found at {args.word_list}")
        print("Use --download to download a word list")
        return

    # Set environment variable if forcing local mode
    if args.force_local:
        os.environ["QDRANT_FORCE_LOCAL"] = "1"
        print("Forcing local mode for Qdrant (not recommended for large collections)")

    # Create the vector database
    if args.use_docker:
        print(f"Using Qdrant in Docker mode with URL: {args.qdrant_url}")
        vector_db = VectorDB(
            collection_name="words",
            path=args.index_path,
            use_docker=True,
            url=args.qdrant_url,
            batch_size=args.batch_size
        )
    else:
        print("Using Qdrant in local mode")
        vector_db = VectorDB(
            collection_name="words",
            path=args.index_path,
            batch_size=args.batch_size
        )

    # Build the index
    build_index(args.word_list, vector_db)


if __name__ == "__main__":
    main()
