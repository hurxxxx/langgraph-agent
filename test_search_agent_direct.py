"""
Direct test script for the search agent.
"""

import os
from dotenv import load_dotenv
from src.agents.search_agent import SearchAgent, SearchAgentConfig

# Load environment variables
load_dotenv()

import asyncio

async def test_streaming():
    """Test the search agent's streaming functionality."""
    print("\n=== Test 3: Streaming functionality ===")
    config = SearchAgentConfig(
        providers=["serper"],
        max_results=3,
        streaming=True
    )

    search_agent = SearchAgent(config)
    query = "What are the top AI research labs in the world?"
    print(f"Query: {query}")

    # Create input for the agent
    agent_input = {"messages": [{"role": "user", "content": query}]}

    # Stream the agent's response
    print("\nStreaming response (chunks):")
    async for chunk in search_agent.astream(agent_input):
        if "messages" in chunk and chunk["messages"]:
            last_message = chunk["messages"][-1]
            if hasattr(last_message, "content"):
                print(f"Chunk received: {len(last_message.content)} characters")
            else:
                print(f"Chunk received: {len(str(last_message))} characters")

    print("\nStreaming completed")

def test_search_agent_direct():
    """Test the search agent directly with different configurations."""
    # Test 1: Korean news query
    print("\n=== Test 1: Korean news query ===")
    config = SearchAgentConfig(
        providers=["serper", "tavily"],  # Use both Serper and Tavily
        max_results=5,
        time_period="1d",  # Last 24 hours
        news_only=True,
        region="kr"  # Set region to Korea
    )

    search_agent = SearchAgent(config)
    query = "What is the latest news in Korea today?"
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    updated_state = search_agent(state)
    print("\nResponse (excerpt):")
    last_message = updated_state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    print(content[:500] + "..." if len(content) > 500 else content)

    # Test 2: Technical query with longer time period
    print("\n=== Test 2: Technical query with longer time period ===")
    config = SearchAgentConfig(
        providers=["serper", "tavily"],
        max_results=5,
        time_period="1m",  # Last month
        news_only=False
    )

    search_agent = SearchAgent(config)
    query = "What are the latest advancements in quantum computing?"
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    updated_state = search_agent(state)
    print("\nResponse (excerpt):")
    last_message = updated_state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    print(content[:500] + "..." if len(content) > 500 else content)

    # Test 3: Streaming functionality
    asyncio.run(test_streaming())

    return updated_state

if __name__ == "__main__":
    test_search_agent_direct()
