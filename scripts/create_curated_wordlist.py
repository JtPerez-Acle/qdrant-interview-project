#!/usr/bin/env python
"""Script to create a curated word list for Contexto-Crusher.

This script generates a smaller, more focused word list for use with Contexto-Crusher.
It can filter words based on frequency, length, and other criteria to create
a more efficient vocabulary for solving Contexto puzzles.
"""

import argparse
import os
import re
import json
from typing import Dict, List, Set, Tuple
import requests
from tqdm import tqdm


def download_word_frequency_list(url: str) -> Dict[str, int]:
    """Download a word frequency list.

    Args:
        url: URL to download from

    Returns:
        Dictionary mapping words to their frequency rank
    """
    print(f"Downloading word frequency data from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    # Parse the word frequency data
    # Format depends on the source, this assumes a CSV-like format: word,frequency
    word_freq = {}
    for line in response.text.splitlines():
        if ',' in line:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                word = parts[0].lower()
                try:
                    # Some lists use rank instead of raw frequency
                    rank = int(parts[1])
                    word_freq[word] = rank
                except ValueError:
                    # Skip lines with non-integer frequency/rank
                    continue

    print(f"Downloaded frequency data for {len(word_freq)} words")
    return word_freq


def load_existing_word_list(file_path: str) -> Set[str]:
    """Load an existing word list.

    Args:
        file_path: Path to the word list file

    Returns:
        Set of words
    """
    if not os.path.exists(file_path):
        print(f"Word list file not found: {file_path}")
        return set()

    print(f"Loading existing word list from {file_path}...")
    with open(file_path, 'r') as f:
        words = {line.strip().lower() for line in f if line.strip()}

    print(f"Loaded {len(words)} words from existing list")
    return words


def load_historical_solutions(file_path: str) -> Set[str]:
    """Load historical Contexto solutions.

    Args:
        file_path: Path to the historical solutions file

    Returns:
        Set of solution words
    """
    if not os.path.exists(file_path):
        print(f"Historical solutions file not found: {file_path}")
        return set()

    print(f"Loading historical solutions from {file_path}...")
    solutions = set()

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract solutions based on the file format
        # This assumes a format like: {"date": "solution", ...}
        if isinstance(data, dict):
            solutions = {word.lower() for word in data.values() if isinstance(word, str)}
        elif isinstance(data, list):
            solutions = {item.lower() for item in data if isinstance(item, str)}
            
    except json.JSONDecodeError:
        # Try plain text format (one word per line)
        with open(file_path, 'r') as f:
            solutions = {line.strip().lower() for line in f if line.strip()}
            
    print(f"Loaded {len(solutions)} historical solutions")
    return solutions


def filter_words(words: Set[str], 
                 word_freq: Dict[str, int], 
                 historical_solutions: Set[str],
                 min_length: int = 3, 
                 max_length: int = 12,
                 max_words: int = 10000) -> List[str]:
    """Filter words based on various criteria.

    Args:
        words: Set of words to filter
        word_freq: Dictionary mapping words to their frequency rank
        historical_solutions: Set of historical solution words
        min_length: Minimum word length
        max_length: Maximum word length
        max_words: Maximum number of words to include

    Returns:
        Filtered list of words
    """
    print("Filtering words...")
    
    # Filter by length
    length_filtered = {word for word in words 
                      if min_length <= len(word) <= max_length}
    print(f"After length filtering: {len(length_filtered)} words")
    
    # Prioritize historical solutions
    final_words = list(historical_solutions)
    print(f"Added {len(final_words)} historical solutions")
    
    # Add high-frequency words
    freq_words = []
    for word in length_filtered:
        if word in word_freq:
            freq_words.append((word, word_freq[word]))
    
    # Sort by frequency (lower rank = higher frequency)
    freq_words.sort(key=lambda x: x[1])
    
    # Add words until we reach the limit
    for word, _ in freq_words:
        if word not in final_words:  # Avoid duplicates
            final_words.append(word)
            if len(final_words) >= max_words:
                break
    
    print(f"Final word list contains {len(final_words)} words")
    return final_words


def save_word_list(words: List[str], output_path: str) -> None:
    """Save the word list to a file.

    Args:
        words: List of words to save
        output_path: Path to save the word list
    """
    print(f"Saving word list to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the word list
    with open(output_path, 'w') as f:
        f.write('\n'.join(words))
    
    print(f"Saved {len(words)} words to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create a curated word list for Contexto-Crusher.")
    parser.add_argument("--input", type=str, default="data/word_list.txt", 
                        help="Path to the input word list")
    parser.add_argument("--output", type=str, default="data/curated_word_list.txt", 
                        help="Path to save the curated word list")
    parser.add_argument("--freq-url", type=str, 
                        default="https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt", 
                        help="URL to download word frequency data")
    parser.add_argument("--historical", type=str, default="data/historical_solutions.json", 
                        help="Path to historical Contexto solutions")
    parser.add_argument("--min-length", type=int, default=3, 
                        help="Minimum word length")
    parser.add_argument("--max-length", type=int, default=12, 
                        help="Maximum word length")
    parser.add_argument("--max-words", type=int, default=10000, 
                        help="Maximum number of words in the curated list")
    
    args = parser.parse_args()
    
    # Load existing word list
    words = load_existing_word_list(args.input)
    
    # Download word frequency data
    word_freq = download_word_frequency_list(args.freq_url)
    
    # Load historical solutions
    historical_solutions = load_historical_solutions(args.historical)
    
    # Filter words
    curated_words = filter_words(
        words, 
        word_freq, 
        historical_solutions,
        args.min_length, 
        args.max_length,
        args.max_words
    )
    
    # Save the curated word list
    save_word_list(curated_words, args.output)
    
    print("Curated word list creation complete!")


if __name__ == "__main__":
    main()
