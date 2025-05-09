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
        default_provider="serper",
        max_results=3,
        parallel_search=False,
        evaluate_results=True,
        additional_queries=False,
        optimize_query=True,
        detect_time_references=False,
        auto_set_time_period=False
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
    print(updated_state["messages"][-1]["content"][:300] + "..."
          if len(updated_state["messages"][-1]["content"]) > 300
          else updated_state["messages"][-1]["content"])

    # Test 2: Query with explicit time period (1 day)
    print("\n=== Test 2: Query with explicit time period (1 day) ===")
    config = SearchAgentConfig(
        providers=["serper"],
        default_provider="serper",
        max_results=3,
        parallel_search=False,
        evaluate_results=True,
        additional_queries=False,
        optimize_query=True,
        detect_time_references=False,
        auto_set_time_period=False,
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
    print(updated_state["messages"][-1]["content"][:300] + "..."
          if len(updated_state["messages"][-1]["content"]) > 300
          else updated_state["messages"][-1]["content"])

    # Test 3: Time-based query with auto detection
    print("\n=== Test 3: Time-based query with auto detection ===")
    config = SearchAgentConfig(
        providers=["serper"],
        default_provider="serper",
        max_results=3,
        parallel_search=False,
        evaluate_results=True,
        additional_queries=False,
        optimize_query=True,
        detect_time_references=True,
        auto_set_time_period=True
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
    print(updated_state["messages"][-1]["content"][:300] + "..."
          if len(updated_state["messages"][-1]["content"]) > 300
          else updated_state["messages"][-1]["content"])

    return updated_state

if __name__ == "__main__":
    test_search_agent()
