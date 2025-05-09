#!/usr/bin/env python3
"""
Run UI Tests Script

This script starts the Streamlit app in a separate process and then runs the Playwright tests.
It ensures that the Streamlit app is running before executing the tests.

Usage:
    python run_ui_tests.py
"""

import os
import sys
import time
import subprocess
import signal
import asyncio
import pytest
import requests
from urllib.error import URLError

# Constants
STREAMLIT_PORT = 8501
STREAMLIT_URL = f"http://localhost:{STREAMLIT_PORT}"
MAX_STARTUP_TIME = 30  # Maximum time to wait for Streamlit to start (seconds)


def is_streamlit_running():
    """Check if Streamlit is running by making a request to the server."""
    try:
        response = requests.get(STREAMLIT_URL)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def start_streamlit():
    """Start the Streamlit app in a separate process."""
    print("Starting Streamlit app...")

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Start Streamlit in a separate process
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", os.path.join(script_dir, "app.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for Streamlit to start
    start_time = time.time()
    while not is_streamlit_running():
        if time.time() - start_time > MAX_STARTUP_TIME:
            print("Timed out waiting for Streamlit to start")
            streamlit_process.terminate()
            sys.exit(1)

        print("Waiting for Streamlit to start...")
        time.sleep(1)

    print(f"Streamlit app is running at {STREAMLIT_URL}")
    return streamlit_process


async def run_tests():
    """Run the Playwright tests."""
    print("Running Playwright tests...")

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Run the tests
    test_result = await asyncio.create_subprocess_exec(
        "python", "-m", "pytest",
        os.path.join(script_dir, "test_prompt_input.py"),
        "-v", "--asyncio-mode=strict",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await test_result.communicate()

    print("Test output:")
    print(stdout.decode())

    if stderr:
        print("Test errors:")
        print(stderr.decode())

    return test_result.returncode


async def main():
    """Main function to run the tests."""
    # Start Streamlit
    streamlit_process = start_streamlit()

    try:
        # Run the tests
        test_result = await run_tests()

        # Print test summary
        if test_result == 0:
            print("All tests passed!")
        else:
            print(f"Tests failed with exit code {test_result}")

        return test_result
    finally:
        # Terminate Streamlit process
        print("Terminating Streamlit process...")
        streamlit_process.terminate()
        streamlit_process.wait()


if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
