"""
Test script for the search agent.
"""

import os
from dotenv import load_dotenv
from src.agents.search_agent import SearchAgent, SearchAgentConfig

# Load environment variables
load_dotenv()

def test_search_agent():
    """Test the search agent with different time period configurations."""
    # Test 1: Regular query without time period
    print("\n=== Test 1: Regular query without time period ===")
    config = SearchAgentConfig(
        providers=["serper"],
        max_results=3
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
    print(content[:300] + "..." if len(content) > 300 else content)

    # Test 2: Query with explicit time period (1 day)
    print("\n=== Test 2: Query with explicit time period (1 day) ===")
    config = SearchAgentConfig(
        providers=["serper"],
        max_results=3,
        time_period="1d"  # Set explicit time period
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
    print(content[:300] + "..." if len(content) > 300 else content)

    # Test 3: Time-based query with auto detection
    print("\n=== Test 3: Time-based query with auto detection ===")
    config = SearchAgentConfig(
        providers=["serper"],
        max_results=3,
        news_only=True,
        time_period="1d"
    )

    search_agent = SearchAgent(config)
    query = "What is today's news about AI?"
    print(f"Query: {query}")

    state = {
        "messages": [{"role": "user", "content": query}],
        "agent_outputs": {}
    }

    updated_state = search_agent(state)
    print("\nResponse (excerpt):")
    last_message = updated_state["messages"][-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    print(content[:300] + "..." if len(content) > 300 else content)

    return updated_state

if __name__ == "__main__":
    test_search_agent()
