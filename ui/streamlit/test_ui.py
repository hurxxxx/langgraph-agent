#!/usr/bin/env python3
"""
Playwright tests for the Streamlit UI.

This script contains automated tests for the Streamlit UI using Playwright.
It tests various aspects of the UI functionality and interaction.

Usage:
    python -m pytest test_ui.py
"""

import os
import sys
import time
import pytest
import asyncio
from playwright.async_api import async_playwright, Page, expect

# Test constants
STREAMLIT_URL = "http://localhost:8501"
TIMEOUT = 30000  # 30 seconds


@pytest.fixture(scope="module")
async def browser():
    """Fixture to create and close a browser instance."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        yield browser
        await browser.close()


@pytest.fixture(scope="function")
async def page(browser):
    """Fixture to create a new page for each test."""
    page = await browser.new_page()
    await page.goto(STREAMLIT_URL)
    
    # Wait for Streamlit to load
    await page.wait_for_selector("h1:has-text('Multi-Agent Supervisor Test Interface')")
    
    yield page
    await page.close()


async def test_page_title(page):
    """Test that the page title is correct."""
    title = await page.title()
    assert "Multi-Agent Supervisor Test Interface" in title


async def test_sidebar_elements(page):
    """Test that the sidebar contains all expected elements."""
    # Check for query input
    await expect(page.locator("text=Test Query")).to_be_visible()
    
    # Check for supervisor configuration
    await expect(page.locator("text=Supervisor Configuration")).to_be_visible()
    await expect(page.locator("text=Supervisor Type")).to_be_visible()
    
    # Check for LLM configuration
    await expect(page.locator("text=LLM Configuration")).to_be_visible()
    await expect(page.locator("text=LLM Provider")).to_be_visible()
    
    # Check for agent selection
    await expect(page.locator("text=Agents")).to_be_visible()
    await expect(page.locator("text=Search Agent")).to_be_visible()
    await expect(page.locator("text=Image Generation Agent")).to_be_visible()
    
    # Check for execute button
    await expect(page.locator("button:has-text('Execute Test')")).to_be_visible()


async def test_tabs(page):
    """Test that all tabs are present and can be selected."""
    # Check that tabs exist
    await expect(page.locator("button:has-text('Execution Monitor')")).to_be_visible()
    await expect(page.locator("button:has-text('Results')")).to_be_visible()
    await expect(page.locator("button:has-text('Quality Evaluation')")).to_be_visible()
    
    # Click on each tab and verify content
    await page.click("button:has-text('Results')")
    await expect(page.locator("h2:has-text('Test Results')")).to_be_visible()
    
    await page.click("button:has-text('Quality Evaluation')")
    await expect(page.locator("h2:has-text('Quality Evaluation')")).to_be_visible()
    
    await page.click("button:has-text('Execution Monitor')")
    await expect(page.locator("h2:has-text('Execution Monitor')")).to_be_visible()


async def test_simple_query_execution(page):
    """Test executing a simple query."""
    # Enter a test query
    await page.fill("textarea[placeholder='Enter your test query here...']", "What is quantum computing?")
    
    # Configure for a simple test
    await page.click("text=Standard")  # Ensure standard supervisor is selected
    await page.click("text=Search Agent")  # Ensure search agent is checked
    await page.uncheck("text=Image Generation Agent")  # Uncheck image agent
    
    # Execute the test
    await page.click("button:has-text('Execute Test')")
    
    # Wait for test to complete (look for log message)
    try:
        await page.wait_for_selector("text=Test completed successfully", timeout=TIMEOUT)
        test_completed = True
    except:
        test_completed = False
    
    assert test_completed, "Test did not complete successfully"
    
    # Switch to Results tab
    await page.click("button:has-text('Results')")
    
    # Check that results are displayed
    await expect(page.locator("text=Query: What is quantum computing?")).to_be_visible()
    
    # Check that execution metrics are displayed
    await expect(page.locator("text=Execution Time")).to_be_visible()
    await expect(page.locator("text=Total Agents")).to_be_visible()


async def test_complex_query_execution(page):
    """Test executing a complex query that uses multiple agents."""
    # Enter a test query
    await page.fill("textarea[placeholder='Enter your test query here...']", 
                   "Research quantum computing and generate an image of a quantum computer")
    
    # Configure for a complex test
    await page.click("text=Standard")  # Ensure standard supervisor is selected
    await page.click("text=Search Agent")  # Ensure search agent is checked
    await page.click("text=Image Generation Agent")  # Ensure image agent is checked
    
    # Execute the test
    await page.click("button:has-text('Execute Test')")
    
    # Wait for test to complete (look for log message)
    try:
        await page.wait_for_selector("text=Test completed successfully", timeout=TIMEOUT)
        test_completed = True
    except:
        test_completed = False
    
    assert test_completed, "Test did not complete successfully"
    
    # Switch to Results tab
    await page.click("button:has-text('Results')")
    
    # Check that results are displayed
    await expect(page.locator("text=Query: Research quantum computing")).to_be_visible()
    
    # Check that both agents were used
    await page.click("button:has-text('Agent Outputs')")
    await expect(page.locator("button:has-text('search_agent Output')")).to_be_visible()
    await expect(page.locator("button:has-text('image_generation_agent Output')")).to_be_visible()


async def test_quality_evaluation(page):
    """Test the quality evaluation functionality."""
    # First run a test to have results to evaluate
    await page.fill("textarea[placeholder='Enter your test query here...']", "What is quantum computing?")
    await page.click("button:has-text('Execute Test')")
    
    # Wait for test to complete
    try:
        await page.wait_for_selector("text=Test completed successfully", timeout=TIMEOUT)
    except:
        pytest.fail("Test did not complete successfully")
    
    # Switch to Quality Evaluation tab
    await page.click("button:has-text('Quality Evaluation')")
    
    # Check that evaluation form is displayed
    await expect(page.locator("h3:has-text('Evaluation Form')")).to_be_visible()
    
    # Fill out evaluation form
    await page.fill("textarea[placeholder='Enter any additional comments or observations...']", 
                   "The response was accurate and comprehensive.")
    
    # Submit evaluation
    await page.click("button:has-text('Submit Evaluation')")
    
    # Check for success message
    await expect(page.locator("text=Evaluation submitted successfully!")).to_be_visible()


async def test_parallel_supervisor(page):
    """Test using the parallel supervisor."""
    # Enter a test query
    await page.fill("textarea[placeholder='Enter your test query here...']", 
                   "Research quantum computing and generate an image of a quantum computer")
    
    # Configure for parallel supervisor
    await page.click("text=Supervisor Type")
    await page.click("text=Parallel")
    
    # Ensure both agents are checked
    await page.click("text=Search Agent")  # Ensure search agent is checked
    await page.click("text=Image Generation Agent")  # Ensure image agent is checked
    
    # Execute the test
    await page.click("button:has-text('Execute Test')")
    
    # Wait for test to complete (look for log message)
    try:
        await page.wait_for_selector("text=Test completed successfully", timeout=TIMEOUT)
        test_completed = True
    except:
        test_completed = False
    
    assert test_completed, "Test did not complete successfully"
    
    # Switch to Results tab
    await page.click("button:has-text('Results')")
    
    # Check that results are displayed
    await expect(page.locator("text=Query: Research quantum computing")).to_be_visible()


if __name__ == "__main__":
    # This allows running the tests directly with python
    asyncio.run(pytest.main(["-xvs", __file__]))
