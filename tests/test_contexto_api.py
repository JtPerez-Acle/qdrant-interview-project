"""Tests for the ContextoAPI class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# We'll implement this class later
from contexto.contexto_api import ContextoAPI


class TestContextoAPI:
    """Test suite for the ContextoAPI class."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test that the ContextoAPI class initializes correctly."""
        api = ContextoAPI(headless=True)
        assert api.headless is True
        assert api.browser is None
        assert api.page is None

    @pytest.mark.asyncio
    async def test_start(self):
        """Test that the browser session starts correctly."""
        # For this test, we'll just check that the method doesn't raise an exception
        # and returns the expected value when mocked properly
        api = ContextoAPI(headless=True)

        # Mock the internal methods to avoid actual browser launch
        with patch.object(api, '_launch_browser', return_value=(AsyncMock(), AsyncMock())):
            result = await api.start()

            # Verify the result
            assert result is True
            assert api.browser is not None
            assert api.page is not None

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test that the browser session stops correctly."""
        api = ContextoAPI(headless=True)
        mock_browser = AsyncMock()
        mock_browser.close = AsyncMock()
        api.browser = mock_browser
        api.page = AsyncMock()

        result = await api.stop()

        assert result is True
        mock_browser.close.assert_called_once()
        assert api.page is None
        assert api.browser is None

    @pytest.mark.asyncio
    async def test_navigate_to_daily(self):
        """Test that navigation to the daily puzzle works."""
        api = ContextoAPI(headless=True)
        api.page = AsyncMock()

        result = await api.navigate_to_daily()

        assert result is True
        api.page.goto.assert_called_once_with("https://contexto.me/")
        api.page.wait_for_selector.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_guess_correct(self):
        """Test submitting a correct guess."""
        api = ContextoAPI(headless=True)
        api.page = AsyncMock()

        # Mock the input field and submit button
        mock_input = AsyncMock()
        mock_button = AsyncMock()
        api.page.query_selector.side_effect = [mock_input, mock_button]

        # Mock the result element with rank 1 (correct guess)
        mock_result = AsyncMock()
        mock_result.text_content.return_value = "1"
        api.page.wait_for_selector.return_value = mock_result

        rank = await api.submit_guess("target")

        assert rank == 1
        mock_input.fill.assert_called_once_with("target")
        mock_button.click.assert_called_once()
        api.page.wait_for_selector.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_guess_incorrect(self):
        """Test submitting an incorrect guess."""
        api = ContextoAPI(headless=True)
        api.page = AsyncMock()

        # Mock the input field and submit button
        mock_input = AsyncMock()
        mock_button = AsyncMock()
        api.page.query_selector.side_effect = [mock_input, mock_button]

        # Mock the result element with rank 42 (incorrect guess)
        mock_result = AsyncMock()
        mock_result.text_content.return_value = "42"
        api.page.wait_for_selector.return_value = mock_result

        rank = await api.submit_guess("wrong")

        assert rank == 42
        mock_input.fill.assert_called_once_with("wrong")
        mock_button.click.assert_called_once()
        api.page.wait_for_selector.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_history(self):
        """Test retrieving guess history."""
        api = ContextoAPI(headless=True)
        api.page = AsyncMock()

        # Mock history elements
        mock_words = [AsyncMock(), AsyncMock()]
        mock_words[0].text_content.return_value = "word1"
        mock_words[1].text_content.return_value = "word2"

        mock_ranks = [AsyncMock(), AsyncMock()]
        mock_ranks[0].text_content.return_value = "10"
        mock_ranks[1].text_content.return_value = "5"

        api.page.query_selector_all.side_effect = [mock_words, mock_ranks]

        history = await api.get_history()

        assert history == [("word1", 10), ("word2", 5)]
        assert api.page.query_selector_all.call_count == 2
