"""Contexto API implementation using Playwright."""

import re
import time
from typing import List, Optional, Tuple

from playwright.async_api import async_playwright, Browser, Page


class ContextoAPI:
    """Headless browser interface for interacting with Contexto.me."""

    def __init__(self, headless: bool = True):
        """Initialize the Contexto API client.

        Args:
            headless: Whether to run the browser in headless mode
        """
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def _launch_browser(self):
        """Launch the browser and return browser and page objects.

        Returns:
            Tuple of (browser, page)
        """
        # Launch the browser
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=self.headless)

        # Create a new context and page
        context = await browser.new_context()
        page = await context.new_page()

        return browser, page

    async def start(self) -> bool:
        """Start the browser session.

        Returns:
            True if successful
        """
        try:
            # Launch the browser
            self.browser, self.page = await self._launch_browser()
            return True
        except Exception as e:
            print(f"Error starting browser: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the browser session.

        Returns:
            True if successful
        """
        try:
            if self.browser:
                await self.browser.close()

            self.page = None
            self.browser = None

            return True
        except Exception as e:
            print(f"Error stopping browser: {e}")
            return False

    async def navigate_to_daily(self) -> bool:
        """Navigate to the daily puzzle.

        Returns:
            True if successful
        """
        try:
            if not self.page:
                return False

            # Navigate to Contexto.me
            await self.page.goto("https://contexto.me/")

            # Wait for the page to load
            await self.page.wait_for_selector("input[type='text']")

            return True
        except Exception as e:
            print(f"Error navigating to daily puzzle: {e}")
            return False

    async def navigate_to_historical(self, date: str) -> bool:
        """Navigate to a historical puzzle.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            True if successful
        """
        try:
            if not self.page:
                return False

            # Validate date format
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
                print(f"Invalid date format: {date}. Expected format: YYYY-MM-DD")
                return False

            # Navigate to the historical puzzle
            url = f"https://contexto.me/#{date}"
            await self.page.goto(url)

            # Wait for the page to load
            await self.page.wait_for_selector("input[type='text']")

            # Check if we're on the correct page
            page_title = await self.page.title()
            if date not in page_title:
                print(f"Warning: Could not verify that we're on the correct historical puzzle page for {date}")

            return True
        except Exception as e:
            print(f"Error navigating to historical puzzle for {date}: {e}")
            return False

    async def submit_guess(self, word: str) -> int:
        """Submit a guess and get the rank.

        Args:
            word: Word to guess

        Returns:
            Rank of the guessed word (1 is the target word)
        """
        try:
            if not self.page:
                return -1

            # Find the input field and submit button
            input_field = await self.page.query_selector("input[type='text']")
            submit_button = await self.page.query_selector("button[type='submit']")

            if not input_field or not submit_button:
                print("Could not find input field or submit button")
                return -1

            # Enter the word and submit
            await input_field.fill(word)
            await submit_button.click()

            # Wait for the result to appear
            result_element = await self.page.wait_for_selector(".result-item:last-child .result-rank")

            if not result_element:
                print("Could not find result element")
                return -1

            # Extract the rank
            rank_text = await result_element.text_content()

            # Parse the rank (remove any non-numeric characters)
            rank = int(re.sub(r'[^0-9]', '', rank_text))

            # Add a small delay to avoid rate limiting
            await self.page.wait_for_timeout(1000)

            return rank
        except Exception as e:
            print(f"Error submitting guess: {e}")
            return -1

    async def get_history(self) -> List[Tuple[str, int]]:
        """Get the history of guesses from the current session.

        Returns:
            List of (word, rank) tuples
        """
        try:
            if not self.page:
                return []

            # Find all word elements
            word_elements = await self.page.query_selector_all(".result-item .result-word")

            # Find all rank elements
            rank_elements = await self.page.query_selector_all(".result-item .result-rank")

            # Extract words and ranks
            words = []
            for element in word_elements:
                word_text = await element.text_content()
                words.append(word_text.strip())

            ranks = []
            for element in rank_elements:
                rank_text = await element.text_content()
                # Parse the rank (remove any non-numeric characters)
                rank = int(re.sub(r'[^0-9]', '', rank_text))
                ranks.append(rank)

            # Combine words and ranks
            history = list(zip(words, ranks))

            return history
        except Exception as e:
            print(f"Error getting history: {e}")
            return []
