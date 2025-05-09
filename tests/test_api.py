"""
Test script for the API endpoints.

This script tests the API endpoints for the multi-agent system.
"""

import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_api():
    """
    Test the API endpoints.
    """
    print("Testing API endpoints...")
    
    # Load environment variables
    load_dotenv()
    
    # API URL
    api_url = "http://localhost:8000"
    
    # Test health endpoint
    print("\nTesting health endpoint...")
    try:
        response = requests.get(f"{api_url}/health")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test agents endpoint
    print("\nTesting agents endpoint...")
    try:
        response = requests.get(f"{api_url}/agents")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test query endpoint (non-streaming)
    print("\nTesting query endpoint (non-streaming)...")
    try:
        query = "What is LangGraph?"
        response = requests.post(
            f"{api_url}/query",
            json={"query": query, "stream": False}
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test query endpoint (streaming)
    print("\nTesting query endpoint (streaming)...")
    try:
        query = "Tell me about the latest developments in AI"
        response = requests.post(
            f"{api_url}/query",
            json={"query": query, "stream": True},
            stream=True
        )
        print(f"Status code: {response.status_code}")
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        print("Stream completed.")
                        break
                    try:
                        chunk = json.loads(data)
                        print(f"Received chunk: {json.dumps(chunk, indent=2)}")
                    except json.JSONDecodeError:
                        print(f"Invalid JSON: {data}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    test_api()
