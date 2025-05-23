"""API Client Example for Multi-Agent Supervisor System"""

import requests
import json
import sseclient


def non_streaming_request(query, base_url="http://localhost:8000"):
    """Make a non-streaming request to the API."""
    url = f"{base_url}/query"

    payload = {
        "query": query,
        "stream": False,
        "context": {}
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def streaming_request(query, base_url="http://localhost:8000"):
    """Make a streaming request to the API."""
    url = f"{base_url}/query"

    payload = {
        "query": query,
        "stream": True,
        "context": {}
    }

    response = requests.post(
        url,
        json=payload,
        stream=True,
        headers={"Accept": "text/event-stream"}
    )

    if response.status_code == 200:
        client = sseclient.SSEClient(response)

        for i, event in enumerate(client.events()):
            if event.data == "[DONE]":
                print("\nStream complete")
                break

            try:
                data = json.loads(event.data)
                print(f"\nChunk {i+1}:")

                if "messages" in data and data["messages"]:
                    latest_message = data["messages"][-1]
                    print(f"Role: {latest_message.get('role', 'unknown')}")
                    print(f"Content: {latest_message.get('content', '')}")

                if "next_agent" in data:
                    print(f"Current Agent: {data['next_agent']}")

            except json.JSONDecodeError:
                print(f"Error parsing JSON: {event.data}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def list_agents(base_url="http://localhost:8000"):
    """List all available agents."""
    url = f"{base_url}/agents"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def health_check(base_url="http://localhost:8000"):
    """Check the health of the API."""
    url = f"{base_url}/health"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def main():
    """Main function to run the examples."""
    base_url = "http://localhost:8000"

    print("Checking API health...")
    health = health_check(base_url)
    if health:
        print(f"API health: {health}")
    else:
        print("API is not available. Make sure the server is running.")
        return

    print("\nListing available agents...")
    agents = list_agents(base_url)
    if agents:
        print(f"Available agents: {agents}")

    korean_news_query = "오늘의 한국 주요 뉴스를 섹션별로 정리해서 보고서를 만들아줘"

    print("\n=== Non-Streaming Example ===")
    print(f"Query: {korean_news_query}")

    response = non_streaming_request(korean_news_query, base_url)
    if response:
        print("\nResponse:")
        print(response["response"])

    print("\n=== Streaming Example ===")
    print(f"Query: {korean_news_query}")

    streaming_request(korean_news_query, base_url)


if __name__ == "__main__":
    main()
