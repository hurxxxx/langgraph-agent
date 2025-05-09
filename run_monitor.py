#!/usr/bin/env python
"""
Run the Multi-Agent Supervisor System with the monitoring UI.

This script starts the FastAPI server and opens the monitoring UI in a web browser.
It also sets up the necessary directory structure for document and image generation.
"""

import os
import sys
import time
import webbrowser
import subprocess
from threading import Thread

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.utils.directory_setup import setup_all_directories

def open_browser():
    """Open the browser after a short delay to ensure the server is running."""
    time.sleep(2)
    webbrowser.open("http://localhost:8000")

def main():
    """Main function to run the application with the monitoring UI."""
    print("Starting Multi-Agent Supervisor System with monitoring UI...")

    # Set up all required directories
    setup_all_directories(verbose=True)

    # Start the browser in a separate thread
    browser_thread = Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Run the FastAPI application
    try:
        # Add the src directory to the path
        sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

        # Import and run the application
        from src.app import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {str(e)}")

if __name__ == "__main__":
    main()
