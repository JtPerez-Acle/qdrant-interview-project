#!/usr/bin/env python
"""Script to download historical Contexto solutions.

This script scrapes historical Contexto puzzles to build a database of past solutions.
This data can be used to improve the curated word list for better solving performance.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from playwright.async_api import async_playwright
from tqdm import tqdm


async def get_solution_for_date(date_str: str, headless: bool = True) -> Optional[str]:
    """Get the solution for a specific date.

    Args:
        date_str: Date string in YYYY-MM-DD format
        headless: Whether to run the browser in headless mode

    Returns:
        Solution word or None if not found
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()

        # Navigate to the historical puzzle
        url = f"https://contexto.me/#{date_str}"
        await page.goto(url)

        try:
            # Wait for the page to load
            await page.wait_for_selector("input[type='text']", timeout=10000)

            # Check if we're on the correct page
            page_title = await page.title()
            if date_str not in page_title:
                print(f"Warning: Could not verify that we're on the correct historical puzzle page for {date_str}")

            # Click the "Give up" button to reveal the solution
            give_up_button = await page.query_selector("button.give-up-button")
            if give_up_button:
                await give_up_button.click()

                # Wait for the confirmation dialog
                confirm_button = await page.wait_for_selector("button.confirm-button", timeout=5000)
                if confirm_button:
                    await confirm_button.click()

                    # Wait for the solution to appear
                    await page.wait_for_selector(".solution-word", timeout=5000)

                    # Extract the solution
                    solution_element = await page.query_selector(".solution-word")
                    if solution_element:
                        solution = await solution_element.text_content()
                        return solution.strip().lower()

        except Exception as e:
            print(f"Error getting solution for {date_str}: {e}")

        finally:
            await browser.close()

        return None


async def download_solutions(start_date: str, end_date: str, output_path: str, headless: bool = True) -> Dict[str, str]:
    """Download solutions for a range of dates.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Path to save the solutions
        headless: Whether to run the browser in headless mode

    Returns:
        Dictionary mapping dates to solutions
    """
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate list of dates
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    # Load existing solutions if available
    solutions = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                solutions = json.load(f)
            print(f"Loaded {len(solutions)} existing solutions from {output_path}")
        except json.JSONDecodeError:
            print(f"Error loading existing solutions from {output_path}")

    # Download solutions for each date
    for date_str in tqdm(dates, desc="Downloading solutions"):
        # Skip if we already have the solution
        if date_str in solutions:
            continue

        # Get the solution
        solution = await get_solution_for_date(date_str, headless)
        if solution:
            solutions[date_str] = solution
            print(f"Found solution for {date_str}: {solution}")

            # Save after each successful download
            with open(output_path, "w") as f:
                json.dump(solutions, f, indent=2)
        else:
            print(f"Could not find solution for {date_str}")

        # Add a delay to avoid rate limiting
        await asyncio.sleep(2)

    # Final save
    with open(output_path, "w") as f:
        json.dump(solutions, f, indent=2)

    print(f"Downloaded {len(solutions)} solutions to {output_path}")
    return solutions


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download historical Contexto solutions.")
    parser.add_argument("--start-date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"), 
                        help="End date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--output", type=str, default="data/historical_solutions.json", 
                        help="Path to save the solutions")
    parser.add_argument("--no-headless", action="store_true", help="Run with visible browser")
    parser.add_argument("--days", type=int, default=30, 
                        help="Number of days to download (if start-date not provided)")

    args = parser.parse_args()

    # Calculate start date if not provided
    if not args.start_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=args.days)
        args.start_date = start_date.strftime("%Y-%m-%d")

    print(f"Downloading solutions from {args.start_date} to {args.end_date}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Download solutions
    asyncio.run(download_solutions(
        args.start_date,
        args.end_date,
        args.output,
        headless=not args.no_headless
    ))


if __name__ == "__main__":
    main()
