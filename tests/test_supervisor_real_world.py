#!/usr/bin/env python3
"""
Real-world test for the Multi-Agent Supervisor System.

This script tests the supervisor with real agents and API calls to verify its functionality
in a real-world scenario. It uses the actual implementation of specialized agents rather than mocks.

Usage:
    python test_supervisor_real_world.py [--mode {standard,mcp,crew,autogen,langgraph}] [--query QUERY]

Options:
    --mode      Supervisor mode to use (default: mcp)
    --query     Query to test with (default: predefined complex query)
"""

import os
import sys
import argparse
import json
import time
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the supervisor and agents
from supervisor.supervisor import Supervisor, SupervisorConfig
from supervisor.parallel_supervisor import ParallelSupervisor, ParallelSupervisorConfig
from agents.search_agent import SearchAgent, SearchAgentConfig
from agents.image_generation_agent import ImageGenerationAgent, ImageGenerationAgentConfig
from agents.quality_agent import QualityAgent, QualityAgentConfig
from agents.vector_storage_agent import VectorStorageAgent, VectorStorageAgentConfig


def initialize_agents():
    """
    Initialize all specialized agents with real configurations.

    Returns:
        dict: Dictionary of agent functions keyed by agent name
    """
    # Initialize search agent
    search_agent = SearchAgent(
        config=SearchAgentConfig(
            provider="serper",
            max_results=3
        )
    )

    # Initialize image generation agent
    image_agent = ImageGenerationAgent(
        config=ImageGenerationAgentConfig(
            provider="dalle",
            dalle_model="dall-e-3",
            image_size="1024x1024"
        )
    )

    # Initialize quality agent
    quality_agent = QualityAgent(
        config=QualityAgentConfig()
    )

    # Initialize vector storage agent
    vector_agent = VectorStorageAgent(
        config=VectorStorageAgentConfig(
            store_type="chroma",
            collection_name="test_collection",
            persist_directory="./test_vector_db"
        )
    )

    return {
        "search_agent": search_agent,
        "image_generation_agent": image_agent,
        "quality_agent": quality_agent,
        "vector_storage_agent": vector_agent
    }


def test_supervisor(mode="mcp", query=None):
    """
    Test the supervisor with real agents.

    Args:
        mode: Supervisor mode to use
        query: Query to test with
    """
    # Default query if none provided
    if query is None:
        query = (
            "Research the latest advancements in quantum computing, focusing on recent breakthroughs "
            "and potential applications. Generate an image that illustrates quantum entanglement. "
            "Evaluate the quality and accuracy of the information gathered. Finally, store the research "
            "in a vector database for future reference."
        )

    print(f"\n=== Testing Supervisor in {mode.upper()} mode ===")
    print(f"Query: {query}")
    print("\nInitializing agents...")

    # Initialize agents
    agents = initialize_agents()

    # Initialize supervisor
    if mode == "parallel":
        supervisor = ParallelSupervisor(
            config=ParallelSupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=True
            ),
            agents=agents
        )
    else:
        supervisor = Supervisor(
            config=SupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=True,
                mcp_mode=mode,
                complexity_threshold=0.6  # Lower threshold to ensure MCP is used for most tasks
            ),
            agents=agents
        )

    print("\nInvoking supervisor...")
    start_time = time.time()

    # Invoke supervisor
    result = supervisor.invoke(query)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n=== Execution completed in {execution_time:.2f} seconds ===")

    # Print agent outputs
    print("\nAgent outputs:")
    for agent_name, output in result.get("agent_outputs", {}).items():
        print(f"\n{agent_name}:")
        print(json.dumps(output, indent=2))

    # Print execution plan if available
    if "execution_plan" in result:
        print("\nExecution plan:")
        print(json.dumps(result["execution_plan"], indent=2))

    # Print execution stats if available
    if "execution_stats" in result:
        print("\nExecution stats:")
        print(json.dumps(result["execution_stats"], indent=2))

    # Print final response
    print("\nFinal response:")
    print(result["messages"][-1]["content"])

    return result


def main():
    """Main function to parse arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test supervisor with real agents")
    parser.add_argument("--mode", choices=["standard", "mcp", "crew", "autogen", "langgraph", "parallel"],
                        default="mcp", help="Supervisor mode to use")
    parser.add_argument("--query", help="Query to test with")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Run the test
    test_supervisor(mode=args.mode, query=args.query)


if __name__ == "__main__":
    main()
