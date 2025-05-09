"""
Test script for the API endpoints.

This script tests the API endpoints for the multi-agent system, including:
- Health check
- Agent listing
- Basic query processing
- Streaming responses
- Image generation with GPT-Image-1
- Complex parallel processing
"""

import os
import sys
import json
import time
import requests
import argparse
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_api(test_type=None, api_url="http://localhost:8000"):
    """
    Test the API endpoints.

    Args:
        test_type: Type of test to run (basic, image, complex, parallel, all)
        api_url: API URL to test
    """
    print("Testing API endpoints...")

    # Load environment variables
    load_dotenv()

    # Test health endpoint
    print("\nTesting health endpoint...")
    try:
        response = requests.get(f"{api_url}/health")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("API server may not be running. Please start the server and try again.")
        return False

    # Test agents endpoint
    print("\nTesting agents endpoint...")
    try:
        response = requests.get(f"{api_url}/agents")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {str(e)}")

    # Run specific tests based on test_type
    if test_type in [None, "all", "basic"]:
        test_basic_query(api_url)

    if test_type in [None, "all", "streaming"]:
        test_streaming_query(api_url)

    if test_type in [None, "all", "image"]:
        test_image_generation(api_url)

    if test_type in [None, "all", "complex"]:
        test_complex_query(api_url)

    if test_type in [None, "all", "parallel"]:
        test_parallel_processing(api_url)

    return True

def test_basic_query(api_url):
    """Test basic query processing."""
    print("\n--- Testing basic query processing ---")
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

def test_streaming_query(api_url):
    """Test streaming query processing."""
    print("\n--- Testing streaming query processing ---")
    try:
        query = "Tell me about the latest developments in AI"
        response = requests.post(
            f"{api_url}/query",
            json={"query": query, "stream": True},
            stream=True
        )
        print(f"Status code: {response.status_code}")

        # Process streaming response
        print("\nStreaming response:")
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data = line_str[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'messages' in chunk and len(chunk['messages']) > 0:
                            content = chunk['messages'][-1].get('content', '')
                            if content:
                                print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {data}")
        print("\n")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_image_generation(api_url):
    """Test image generation with GPT-Image-1."""
    print("\n--- Testing image generation with GPT-Image-1 ---")
    try:
        query = "Generate an image of a futuristic city with flying cars and neon lights."
        response = requests.post(
            f"{api_url}/query",
            json={"query": query, "stream": False}
        )
        print(f"Status code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        # Check if the response contains an image URL
        content = result.get("response", "")
        if "http" in content and (".png" in content or ".jpg" in content):
            print("✅ Image URL found in response")
        else:
            print("❌ No image URL found in response")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_complex_query(api_url):
    """Test complex query with multiple tasks."""
    print("\n--- Testing complex query with multiple tasks ---")
    try:
        query = "I need information about renewable energy sources and their impact on climate change. Also, generate an image showing solar panels in a green field."
        response = requests.post(
            f"{api_url}/query",
            json={"query": query, "stream": True, "use_parallel": True},
            stream=True
        )
        print(f"Status code: {response.status_code}")

        # Process streaming response
        print("\nStreaming response:")
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data = line_str[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'messages' in chunk and len(chunk['messages']) > 0:
                            content = chunk['messages'][-1].get('content', '')
                            if content:
                                print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {data}")
        print("\n")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_parallel_processing(api_url):
    """Test parallel processing of multiple tasks."""
    print("\n--- Testing parallel processing of multiple tasks ---")
    try:
        query = "Find information about the history of artificial intelligence, the current state of quantum computing, and create an image that represents the future of technology."
        response = requests.post(
            f"{api_url}/query",
            json={"query": query, "stream": True, "use_parallel": True},
            stream=True
        )
        print(f"Status code: {response.status_code}")

        # Process streaming response
        print("\nStreaming response:")
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data = line_str[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'messages' in chunk and len(chunk['messages']) > 0:
                            content = chunk['messages'][-1].get('content', '')
                            if content:
                                print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {data}")
        print("\n")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the multi-agent system API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--test-type", choices=["basic", "streaming", "image", "complex", "parallel", "all"],
                        default="all", help="Type of test to run")
    args = parser.parse_args()

    test_api(test_type=args.test_type, api_url=args.api_url)
