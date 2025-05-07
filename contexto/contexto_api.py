"""Contexto API implementation using Playwright for the curated approach."""

import re
import time
import logging
from typing import List, Optional, Tuple

from playwright.async_api import async_playwright, Browser, Page

# Configure logging
logger = logging.getLogger(__name__)


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
            logger.info("Launching browser...")
            self.browser, self.page = await self._launch_browser()
            logger.info("Browser launched successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting browser: {e}")
            return False

    async def stop(self) -> bool:
        """Stop the browser session.

        Returns:
            True if successful
        """
        try:
            if self.browser:
                logger.info("Closing browser...")
                await self.browser.close()

            self.page = None
            self.browser = None
            logger.info("Browser closed successfully")

            return True
        except Exception as e:
            logger.error(f"Error stopping browser: {e}")
            return False

    async def navigate_to_daily(self) -> bool:
        """Navigate to the daily puzzle.

        Returns:
            True if successful
        """
        try:
            if not self.page:
                logger.error("No page available for navigation")
                return False

            # Navigate to Contexto.me with increased timeout
            logger.info("Navigating to Contexto.me...")
            try:
                # First try with a shorter timeout
                await self.page.goto("https://contexto.me/", timeout=30000)  # 30 seconds
            except Exception as e:
                logger.warning(f"Navigation timeout with 30s, trying to continue anyway: {e}")
                # Even if we get a timeout, the page might still be usable

                # Take a screenshot to see the current state
                try:
                    await self.page.screenshot(path="contexto_navigation_timeout.png")
                    logger.info("Screenshot saved as contexto_navigation_timeout.png")
                except Exception:
                    pass

                # Try to reload the page with a shorter timeout
                try:
                    logger.info("Trying to reload the page...")
                    await self.page.reload(timeout=15000)  # 15 seconds
                except Exception as e:
                    logger.warning(f"Reload timeout, continuing anyway: {e}")

                # Wait a moment for any scripts to load
                await self.page.wait_for_timeout(5000)

            # Wait for the page to load
            logger.info("Waiting for page to load...")

            # Use the exact selector for the input field
            try:
                await self.page.wait_for_selector("input.word[type='text']", timeout=5000)
            except Exception:
                try:
                    # Try the exact JS path
                    await self.page.wait_for_selector("#root > div > main > form > input", timeout=5000)
                except Exception:
                    try:
                        # Fallback to more generic selectors
                        await self.page.wait_for_selector("input[type='text']", timeout=5000)
                    except Exception as e:
                        logger.error(f"Could not find input field: {e}")
                        # Take a screenshot to debug
                        await self.page.screenshot(path="contexto_load_debug.png")
                        logger.info("Debug screenshot saved as contexto_load_debug.png")
                        return False

            logger.info("Page loaded successfully")

            # Check if we need to handle any cookie consent or popup
            try:
                cookie_button = await self.page.query_selector("button.cookie-consent-button")
                if cookie_button:
                    await cookie_button.click()
                    logger.info("Clicked cookie consent button")
            except Exception:
                pass

            # Check if we need to close any modal or popup
            try:
                close_button = await self.page.query_selector("button.close-modal")
                if close_button:
                    await close_button.click()
                    logger.info("Closed modal popup")
            except Exception:
                pass

            return True
        except Exception as e:
            logger.error(f"Error navigating to daily puzzle: {e}")
            # Take a screenshot to debug
            try:
                await self.page.screenshot(path="contexto_error_debug.png")
                logger.info("Debug screenshot saved as contexto_error_debug.png")
            except Exception:
                pass
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
                logger.error("No page available for submitting guess")
                return -1

            logger.info(f"Submitting guess: '{word}'")

            # Find the input field and submit button
            # Use the exact selector for the input field
            input_field = await self.page.query_selector("input.word[type='text']")
            if not input_field:
                # Try the exact JS path
                input_field = await self.page.query_selector("#root > div > main > form > input")
            if not input_field:
                # Fallback to more generic selectors
                input_field = await self.page.query_selector("input[type='text']")

            # For the submit button, we'll try to find the form and use Enter key instead
            # since the user mentioned we can use enterkeyhint="send"
            submit_button = None
            form = await self.page.query_selector("form")

            if not input_field:
                logger.error("Could not find input field")
                return -1

            # Enter the word and submit using Enter key
            await input_field.fill(word)
            await input_field.press("Enter")
            logger.info("Guess submitted, waiting for result...")

            # Wait for the result to appear
            try:
                # Wait a moment for the result to appear
                await self.page.wait_for_timeout(2000)

                # Take a screenshot after submitting the guess
                await self.page.screenshot(path="contexto_after_submit.png")
                logger.info("Screenshot saved as contexto_after_submit.png")

                # Try to find the result using various selectors
                selectors_to_try = [
                    # Exact selectors provided by the user
                    ".message > div:nth-child(1) > div:nth-child(1) > div:nth-child(2)",
                    ".guess-history > div:nth-child(1) > div:nth-child(2)",
                    # XPath selectors
                    "xpath=/html/body/div[1]/div/main/div[4]/div/div/div[2]",
                    "xpath=/html/body/div[1]/div/main/div[5]/div/div[2]",
                    # Previous selectors as fallback
                    ".row > span:nth-child(2)",
                    "#root > div > main > div.message > div > div > div.row > span:nth-child(2)",
                    ".row span:last-child",
                    "div.row span:last-child",
                    ".row span",
                    "div.row span",
                    "div[class*='row'] span:last-child",
                    "div[class*='guess'] span:last-child"
                ]

                result_element = None
                for selector in selectors_to_try:
                    try:
                        # Handle XPath selectors differently
                        if selector.startswith("xpath="):
                            xpath = selector.replace("xpath=", "")
                            result_element = await self.page.wait_for_selector(f"xpath={xpath}", timeout=2000)
                        else:
                            result_element = await self.page.wait_for_selector(selector, timeout=2000)

                        if result_element:
                            logger.info(f"Found result element with selector: {selector}")

                            # Get the text content
                            text_content = await result_element.text_content()
                            logger.info(f"Element text content: {text_content}")

                            break
                    except Exception as e:
                        logger.debug(f"Selector {selector} failed: {e}")
                        continue

                # If we found a result element, extract the rank
                if result_element:
                    try:
                        rank_text = await result_element.text_content()
                        # Parse the rank (remove any non-numeric characters)
                        # Extract numbers from the text (e.g., "feature358" -> 358)
                        import re
                        numbers = re.findall(r'\d+', rank_text)
                        if numbers:
                            rank = int(numbers[-1])  # Use the last number found
                            logger.info(f"Received rank: {rank}")
                        else:
                            # If no numbers found, try to extract using regex
                            rank_str = re.sub(r'[^0-9]', '', rank_text)
                            if rank_str:
                                rank = int(rank_str)
                                logger.info(f"Received rank (using regex): {rank}")
                            else:
                                logger.error(f"No numbers found in text: '{rank_text}'")
                                raise ValueError(f"No numbers found in text: '{rank_text}'")

                        # Add a small delay to avoid rate limiting
                        await self.page.wait_for_timeout(1000)

                        return rank
                    except Exception as e:
                        logger.error(f"Error extracting rank from element: {e}")

                # If we still can't find the element, try to get all text on the page
                if not result_element:
                    logger.warning("Could not find result element with any selector, trying to extract all text")
                    # Try to get all text on the page
                    all_text = await self.page.evaluate("() => document.body.innerText")
                    logger.info(f"Page text: {all_text}")

                    # Try to find a number in the text that might be the rank
                    import re
                    rank_matches = re.findall(r'\b\d+\b', all_text)
                    if rank_matches:
                        # Use the first number found as the rank
                        try:
                            rank = int(rank_matches[0])
                            logger.info(f"Found potential rank in page text: {rank}")
                            return rank
                        except ValueError:
                            pass
            except Exception as e:
                logger.error(f"Error waiting for result: {e}")

            # Take a screenshot to debug
            await self.page.screenshot(path="contexto_debug.png")
            logger.info("Debug screenshot saved as contexto_debug.png")

            # If we couldn't find a rank, return a default value
            # Using 999 instead of -1 to avoid breaking the solver's logic
            return 999
        except Exception as e:
            logger.error(f"Error submitting guess: {e}")
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
                import re
                numbers = re.findall(r'\d+', rank_text)
                if numbers:
                    rank = int(numbers[-1])  # Use the last number found
                else:
                    # If no numbers found, try to extract using regex
                    rank_str = re.sub(r'[^0-9]', '', rank_text)
                    if rank_str:
                        rank = int(rank_str)
                    else:
                        logger.warning(f"No numbers found in rank text: '{rank_text}', using 999")
                        rank = 999
                ranks.append(rank)

            # Combine words and ranks
            history = list(zip(words, ranks))

            return history
        except Exception as e:
            print(f"Error getting history: {e}")
            return []
