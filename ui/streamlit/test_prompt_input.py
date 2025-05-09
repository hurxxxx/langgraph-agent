#!/usr/bin/env python3
"""
Playwright tests for direct prompt input in the Streamlit UI.

This script contains automated tests that directly input various prompts into the
Streamlit UI and verify the responses. It tests the end-to-end functionality
of the multi-agent supervisor system through the UI.

Usage:
    python -m pytest test_prompt_input.py
"""

import os
import sys
import time
import pytest
import pytest_asyncio
import asyncio
from playwright.async_api import async_playwright, Page, expect

# Test constants
STREAMLIT_URL = "http://localhost:8501"
TIMEOUT = 60000  # 60 seconds
SCREENSHOT_DIR = "test_screenshots"


@pytest.fixture(scope="module")
def ensure_screenshot_dir():
    """Ensure the screenshot directory exists."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)


@pytest_asyncio.fixture(scope="module")
async def browser():
    """Fixture to create and close a browser instance."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # 헤드리스 모드로 변경
        yield browser
        await browser.close()


@pytest_asyncio.fixture(scope="function")
async def page(browser, ensure_screenshot_dir):
    """Fixture to create a new page for each test."""
    page = await browser.new_page()
    await page.goto(STREAMLIT_URL)

    # Wait for Streamlit to load
    await page.wait_for_selector("h1:has-text('Multi-Agent Supervisor Test Interface')")

    yield page
    await page.close()


async def take_screenshot(page, name):
    """Take a screenshot and save it to the screenshots directory."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{SCREENSHOT_DIR}/{name}_{timestamp}.png"
    await page.screenshot(path=filename, full_page=True)
    print(f"Screenshot saved to {filename}")


@pytest.mark.asyncio
async def test_simple_search_prompt(page):
    """Test a simple search prompt."""
    test_name = "simple_search"

    # Enter a simple search query
    await page.fill("textarea[placeholder='Enter your test query here...']",
                   "What are the latest advancements in quantum computing?")

    # Configure for a simple test with search agent only
    await page.click("text=Standard")  # Ensure standard supervisor is selected
    await page.click("text=Search Agent")  # Ensure search agent is checked
    await page.uncheck("text=Image Generation Agent")  # Uncheck image agent

    # Take screenshot of configuration
    await take_screenshot(page, f"{test_name}_config")

    # Execute the test
    await page.click("button:has-text('Execute Test')")

    # Wait for test to start processing
    await page.wait_for_selector("text=Processing query...", timeout=5000)

    # Take screenshot during processing
    await take_screenshot(page, f"{test_name}_processing")

    # Wait for test to complete
    try:
        await page.wait_for_selector("text=Test completed successfully", timeout=TIMEOUT)
        test_completed = True
    except:
        test_completed = False
        await take_screenshot(page, f"{test_name}_timeout")

    assert test_completed, "Test did not complete successfully"

    # Switch to Results tab
    await page.click("button:has-text('Results')")

    # Take screenshot of results
    await take_screenshot(page, f"{test_name}_results")

    # Check that results are displayed
    await expect(page.locator("text=Query: What are the latest advancements in quantum computing?")).to_be_visible()

    # Check that search agent was used
    await page.click("button:has-text('Agent Outputs')")
    await expect(page.locator("button:has-text('search_agent Output')")).to_be_visible()

    # Take screenshot of agent outputs
    await take_screenshot(page, f"{test_name}_agent_outputs")


@pytest.mark.asyncio
async def test_complex_multi_agent_prompt(page):
    """Test a complex prompt that requires multiple agents."""
    test_name = "complex_multi_agent"

    # Enter a complex query that requires multiple agents
    await page.fill("textarea[placeholder='Enter your test query here...']",
                   "Research the impact of quantum computing on cryptography and generate an image visualizing quantum encryption.")

    # Configure for a complex test with multiple agents
    await page.click("text=Standard")  # Ensure standard supervisor is selected
    await page.click("text=Search Agent")  # Ensure search agent is checked
    await page.check("text=Image Generation Agent")  # Ensure image agent is checked

    # Select DALL-E as the image provider
    await page.click("text=DALL-E")  # Select DALL-E from the dropdown

    # Take screenshot of configuration
    await take_screenshot(page, f"{test_name}_config")

    # Execute the test
    await page.click("button:has-text('Execute Test')")

    # Wait for test to start processing
    await page.wait_for_selector("text=Processing query...", timeout=5000)

    # Take screenshot during processing
    await take_screenshot(page, f"{test_name}_processing")

    # Wait for test to complete
    try:
        await page.wait_for_selector("text=Test completed successfully", timeout=TIMEOUT)
        test_completed = True
    except:
        test_completed = False
        await take_screenshot(page, f"{test_name}_timeout")

    assert test_completed, "Test did not complete successfully"

    # Switch to Results tab
    await page.click("button:has-text('Results')")

    # Take screenshot of results
    await take_screenshot(page, f"{test_name}_results")

    # Check that results are displayed
    await expect(page.locator("text=Query: Research the impact of quantum computing")).to_be_visible()

    # Check that both agents were used
    await page.click("button:has-text('Agent Outputs')")
    await expect(page.locator("button:has-text('search_agent Output')")).to_be_visible()
    await expect(page.locator("button:has-text('image_generation_agent Output')")).to_be_visible()

    # Take screenshot of agent outputs
    await take_screenshot(page, f"{test_name}_agent_outputs")


@pytest.mark.asyncio
async def test_parallel_supervisor_prompt(page):
    """Test a prompt using the parallel supervisor."""
    test_name = "parallel_supervisor"

    # Enter a query suitable for parallel processing
    await page.fill("textarea[placeholder='Enter your test query here...']",
                   "Find information about artificial intelligence in healthcare and generate an image of a medical AI system.")

    # Configure for parallel supervisor
    await page.click("text=Supervisor Type")
    await page.click("text=Parallel")

    # Ensure both agents are checked
    await page.check("text=Search Agent")  # Ensure search agent is checked
    await page.check("text=Image Generation Agent")  # Ensure image agent is checked

    # Select DALL-E as the image provider
    await page.click("text=DALL-E")  # Select DALL-E from the dropdown

    # Take screenshot of configuration
    await take_screenshot(page, f"{test_name}_config")

    # Execute the test
    await page.click("button:has-text('Execute Test')")

    # Wait for test to start processing
    await page.wait_for_selector("text=Processing query...", timeout=5000)

    # Take screenshot during processing
    await take_screenshot(page, f"{test_name}_processing")

    # Wait for test to complete
    try:
        await page.wait_for_selector("text=Test completed successfully", timeout=TIMEOUT)
        test_completed = True
    except:
        test_completed = False
        await take_screenshot(page, f"{test_name}_timeout")

    assert test_completed, "Test did not complete successfully"

    # Switch to Results tab
    await page.click("button:has-text('Results')")

    # Take screenshot of results
    await take_screenshot(page, f"{test_name}_results")

    # Check that results are displayed
    await expect(page.locator("text=Query: Find information about artificial intelligence")).to_be_visible()

    # Check that both agents were used
    await page.click("button:has-text('Agent Outputs')")
    await expect(page.locator("button:has-text('search_agent Output')")).to_be_visible()
    await expect(page.locator("button:has-text('image_generation_agent Output')")).to_be_visible()

    # Take screenshot of agent outputs
    await take_screenshot(page, f"{test_name}_agent_outputs")


if __name__ == "__main__":
    # This allows running the tests directly with python
    pytest.main(["-xvs", "--asyncio-mode=strict", __file__])
