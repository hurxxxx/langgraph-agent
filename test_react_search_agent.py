"""
Test script for the ReAct search agent.
"""

import os
from dotenv import load_dotenv
from src.agents.react_search_agent import ReactSearchAgent

# Load environment variables
load_dotenv()

def test_react_search_agent():
    """Test the ReAct search agent with different configurations."""
    # Test 1: Regular query without time period
    print("\n=== Test 1: Regular query without time period ===")
    search_agent = ReactSearchAgent(
        max_results=3,
        streaming=True
    )

    query = "What are the latest advancements in quantum computing?"
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    updated_state = search_agent(state)

    # Print the response
    if "messages" in updated_state and len(updated_state["messages"]) > 0:
        print("\nResponse (excerpt):")
        last_message = updated_state["messages"][-1]
        if hasattr(last_message, "content"):
            response = last_message.content
        else:
            response = str(last_message)
        print(response[:300] + "..." if len(response) > 300 else response)
    else:
        print("\nNo response received.")

    # Test 2: Query with time period (1 day)
    print("\n=== Test 2: Query with time period (1 day) ===")
    search_agent = ReactSearchAgent(
        max_results=3,
        streaming=True,
        time_period="1d"
    )

    query = "What are the latest advancements in quantum computing?"
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    updated_state = search_agent(state)

    # Print the response
    if "messages" in updated_state and len(updated_state["messages"]) > 0:
        print("\nResponse (excerpt):")
        last_message = updated_state["messages"][-1]
        if hasattr(last_message, "content"):
            response = last_message.content
        else:
            response = str(last_message)
        print(response[:300] + "..." if len(response) > 300 else response)
    else:
        print("\nNo response received.")

    # Test 3: News query
    print("\n=== Test 3: News query ===")
    search_agent = ReactSearchAgent(
        max_results=3,
        streaming=True,
        news_only=True,
        time_period="1d"
    )

    query = "What is today's news about AI?"
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    updated_state = search_agent(state)

    # Print the response
    if "messages" in updated_state and len(updated_state["messages"]) > 0:
        print("\nResponse (excerpt):")
        last_message = updated_state["messages"][-1]
        if hasattr(last_message, "content"):
            response = last_message.content
        else:
            response = str(last_message)
        print(response[:300] + "..." if len(response) > 300 else response)
    else:
        print("\nNo response received.")

    # Test 4: Streaming example
    print("\n=== Test 4: Streaming example ===")
    search_agent = ReactSearchAgent(
        max_results=3,
        streaming=True
    )

    query = "What is the capital of France and what are some interesting facts about it?"
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    print("\nStreaming response (first few chunks):")
    chunk_count = 0
    max_chunks = 3  # Limit to first 3 chunks for demonstration

    try:
        for chunk in search_agent.stream(state, stream_mode="values"):
            chunk_count += 1
            if chunk_count <= max_chunks:
                print(f"\nChunk {chunk_count}:")
                if "messages" in chunk and chunk["messages"]:
                    last_message = chunk["messages"][-1]
                    if hasattr(last_message, "content"):
                        content = last_message.content
                    else:
                        content = str(last_message)
                    print(content[:100] + "..." if content and len(content) > 100 else content)
                else:
                    print(str(chunk)[:100] + "..." if len(str(chunk)) > 100 else str(chunk))

            if chunk_count == max_chunks:
                print("\n... (more chunks available) ...")
                break
    except Exception as e:
        print(f"Error during streaming: {str(e)}")

    return updated_state

if __name__ == "__main__":
    test_react_search_agent()
