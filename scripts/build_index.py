#!/usr/bin/env python
"""Script to build the curated vector index for Contexto-Crusher."""

import argparse
import os
import time
import sys
import logging
from typing import List

import requests
from tqdm import tqdm

from contexto.vector_db import VectorDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_word_frequency_list(url: str, output_path: str) -> int:
    """Download a word frequency list.

    Args:
        url: URL to download from
        output_path: Path to save the word list

    Returns:
        Number of words downloaded
    """
    logger.info(f"Downloading word frequency data from {url}...")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Download the word list
    response = requests.get(url)
    response.raise_for_status()

    # Parse the word frequency data
    # Format depends on the source, this assumes a CSV-like format: word,frequency
    words = []
    for line in response.text.splitlines():
        if ',' in line:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                word = parts[0].lower()
                if word.isalpha():
                    words.append(word)

    # Remove duplicates
    words = list(set(words))

    # Save the word list
    with open(output_path, "w") as f:
        f.write("\n".join(words))

    logger.info(f"Downloaded {len(words)} words to {output_path}")
    return len(words)


def build_index(word_list_path: str, vector_db: VectorDB) -> int:
    """Build the vector index from a curated word list.

    Args:
        word_list_path: Path to the word list
        vector_db: VectorDB instance

    Returns:
        Number of words indexed
    """
    logger.info(f"Building vector index from {word_list_path}...")
    logger.info("This process will:")
    logger.info("1. Load the sentence transformer model")
    logger.info("2. Read all words from the curated word list")
    logger.info("3. Create embeddings for each word")
    logger.info("4. Store the embeddings in the Qdrant vector database")
    logger.info("Starting the process now...")

    start_time = time.time()
    count = vector_db.build_index(word_list_path)
    end_time = time.time()

    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    logger.info(f"Indexing complete!")
    logger.info(f"Indexed {count} words in {minutes} minutes and {seconds} seconds")
    logger.info(f"Average time per word: {total_time / count:.4f} seconds")
    logger.info(f"Vector index stored at: {vector_db.path}")
    return count


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build the curated vector index for Contexto-Crusher.")
    parser.add_argument("--word-list", type=str, default="data/20k.txt",
                        help="Path to the word list (default: data/20k.txt)")
    parser.add_argument("--download", action="store_true",
                        help="Download the word frequency list")
    parser.add_argument("--url", type=str,
                        default="https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt",
                        help="URL to download the word frequency list from")
    parser.add_argument("--index-path", type=str, default="data/vector_index",
                        help="Path to store the vector index")
    parser.add_argument("--use-docker", action="store_true",
                        help="Use Qdrant in Docker instead of local mode")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333",
                        help="URL of the Qdrant server if using Docker")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for embedding generation (default: 64)")
    parser.add_argument("--max-words", type=int, default=20000,
                        help="Maximum number of words to include (default: 20000)")
    parser.add_argument("--historical", type=str, default="data/historical_solutions.json",
                        help="Path to historical solutions file")

    args = parser.parse_args()

    # Download the word list if requested
    if args.download:
        download_word_frequency_list(args.url, args.word_list)

    # Check if the word list exists
    if not os.path.exists(args.word_list):
        logger.error(f"Word list not found at {args.word_list}")
        logger.info("Use --download to download a word frequency list")
        return

    # Create the vector database
    if args.use_docker:
        logger.info(f"Using Qdrant in Docker mode with URL: {args.qdrant_url}")
        vector_db = VectorDB(
            collection_name="words_curated",
            path=args.index_path,
            use_docker=True,
            url=args.qdrant_url,
            batch_size=args.batch_size
        )
    else:
        logger.info("Using Qdrant in local mode")
        vector_db = VectorDB(
            collection_name="words_curated",
            path=args.index_path,
            batch_size=args.batch_size
        )

    # Build the index
    build_index(args.word_list, vector_db)


if __name__ == "__main__":
    main()
