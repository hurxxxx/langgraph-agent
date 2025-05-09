#!/usr/bin/env python3
"""
Real-world streaming test for the Multi-Agent Supervisor System.

This script tests the supervisor's streaming functionality with real agents and API calls.
It demonstrates how streaming works in practice with the supervisor and specialized agents.

Usage:
    python test_streaming_real_world.py [--mode {standard,mcp,crew,autogen,langgraph,parallel}] [--query QUERY]

Options:
    --mode      Supervisor mode to use (default: mcp)
    --query     Query to test with (default: predefined complex query)
"""

import os
import sys
import argparse
import json
import time
import asyncio
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the supervisor and agents
from src.supervisor.supervisor import Supervisor, SupervisorConfig
from src.supervisor.parallel_supervisor import ParallelSupervisor, ParallelSupervisorConfig
from src.agents.search_agent import SearchAgent, SearchAgentConfig
from src.agents.image_generation_agent import ImageGenerationAgent, ImageGenerationAgentConfig
from src.agents.quality_agent import QualityAgent, QualityAgentConfig


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

    return {
        "search_agent": search_agent,
        "image_generation_agent": image_agent,
        "quality_agent": quality_agent
    }


async def test_streaming(mode="mcp", query=None):
    """
    Test the supervisor's streaming functionality with real agents.

    Args:
        mode: Supervisor mode to use
        query: Query to test with
    """
    # Default query if none provided
    if query is None:
        query = (
            "Research the latest advancements in quantum computing, focusing on recent breakthroughs "
            "and potential applications. Generate an image that illustrates quantum entanglement."
        )

    print(f"\n=== Testing Streaming in {mode.upper()} mode ===")
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

    print("\nInvoking supervisor with streaming...")
    start_time = time.time()

    # Create a simple streaming handler
    class StreamingHandler:
        def __init__(self):
            self.chunks = []
            self.current_agent = None
            self.agents_seen = set()

        def on_chunk(self, chunk):
            # Extract the current agent if available
            if "current_agent" in chunk:
                self.current_agent = chunk["current_agent"]
                if self.current_agent:
                    self.agents_seen.add(self.current_agent)

            # Extract the content if available
            if "messages" in chunk and chunk["messages"]:
                content = chunk["messages"][-1]["content"]
                self.chunks.append(content)

                # Print the chunk with agent information
                agent_info = f"[{self.current_agent}] " if self.current_agent else ""
                print(f"{agent_info}{content}", end="", flush=True)

    # Create a streaming handler
    handler = StreamingHandler()

    # Invoke supervisor with streaming
    async for chunk in supervisor.astream(query):
        handler.on_chunk(chunk)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n\n=== Streaming completed in {execution_time:.2f} seconds ===")
    print(f"Agents involved: {', '.join(handler.agents_seen)}")
    print(f"Total chunks received: {len(handler.chunks)}")

    return handler


def main():
    """Main function to parse arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test supervisor streaming with real agents")
    parser.add_argument("--mode", choices=["standard", "mcp", "crew", "autogen", "langgraph", "parallel"],
                        default="standard", help="Supervisor mode to use")
    parser.add_argument("--query", help="Query to test with")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Run the test
    asyncio.run(test_streaming(mode=args.mode, query=args.query))


if __name__ == "__main__":
    main()
