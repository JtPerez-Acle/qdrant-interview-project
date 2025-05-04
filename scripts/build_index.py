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
    
    start_time = time.time()
    count = vector_db.build_index(word_list_path)
    end_time = time.time()
    
    print(f"Indexed {count} words in {end_time - start_time:.2f} seconds")
    return count


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build the vector index for Contexto-Crusher.")
    parser.add_argument("--word-list", type=str, default="data/word_list.txt", help="Path to the word list")
    parser.add_argument("--download", action="store_true", help="Download the word list")
    parser.add_argument("--url", type=str, default="https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt", help="URL to download the word list from")
    parser.add_argument("--index-path", type=str, default="data/vector_index", help="Path to store the vector index")
    
    args = parser.parse_args()
    
    # Download the word list if requested
    if args.download:
        download_word_list(args.url, args.word_list)
    
    # Check if the word list exists
    if not os.path.exists(args.word_list):
        print(f"Word list not found at {args.word_list}")
        print("Use --download to download a word list")
        return
    
    # Create the vector database
    vector_db = VectorDB(collection_name="words", path=args.index_path)
    
    # Build the index
    build_index(args.word_list, vector_db)


if __name__ == "__main__":
    main()
