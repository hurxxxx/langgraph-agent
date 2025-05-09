"""
Run Server Script for Multi-Agent Supervisor System

This script runs the FastAPI server for the multi-agent supervisor system.
It also sets up the necessary directory structure for document and image generation.
"""

import os
import uvicorn
from dotenv import load_dotenv
from src.utils.directory_setup import setup_all_directories

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    print("Starting Multi-Agent Supervisor API server...")

    # Set up all required directories
    setup_all_directories(verbose=True)

    print("API will be available at http://localhost:8000")
    print("API documentation will be available at http://localhost:8000/docs")

    # Run the API server
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
