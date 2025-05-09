#!/usr/bin/env python3
"""
Test script for the multi-agent supervisor system.

This script tests the supervisor with different configurations and queries.
"""

import os
import sys
import time
import json
import asyncio
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import the supervisor and related components
from src.supervisor.supervisor import Supervisor, SupervisorConfig
from src.supervisor.parallel_supervisor import ParallelSupervisor, ParallelSupervisorConfig
from src.agents.search_agent import SearchAgent, SearchAgentConfig
from src.agents.image_generation_agent import ImageGenerationAgent, ImageGenerationAgentConfig


# Load environment variables
load_dotenv()


class StreamingHandler:
    """Handler for streaming responses."""

    def __init__(self):
        """Initialize the streaming handler."""
        self.chunks = []
        self.agents_seen = set()

    def on_chunk(self, chunk):
        """Process a streaming chunk."""
        self.chunks.append(chunk)
        if "current_agent" in chunk:
            self.agents_seen.add(chunk["current_agent"])

        # Print the chunk for debugging
        if "messages" in chunk and len(chunk["messages"]) > 0:
            last_message = chunk["messages"][-1]
            if "content" in last_message:
                print(f"Chunk content: {last_message['content'][:100]}...")

        if "current_agent" in chunk:
            print(f"Current agent: {chunk['current_agent']}")


async def test_standard_supervisor():
    """Test the standard supervisor with a simple query."""
    print("\n=== Testing Standard Supervisor ===")

    # Create a search agent
    search_agent = SearchAgent(
        config=SearchAgentConfig(
            provider="serper",
            llm_provider="openai",
            openai_model="gpt-4o"
        )
    )

    # Create a supervisor with the search agent
    supervisor = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=True,
            mcp_mode="standard"
        ),
        agents={
            "search_agent": search_agent
        }
    )

    # Create a streaming handler
    handler = StreamingHandler()

    # Test query
    query = "What is quantum computing and how does it work?"

    print(f"Query: {query}")
    print("Running test with streaming...")

    # Start time
    start_time = time.time()

    # Process the query with streaming
    async for chunk in supervisor.astream(query):
        handler.on_chunk(chunk)

    # End time
    end_time = time.time()

    # Print results
    print(f"\nTest completed in {end_time - start_time:.2f} seconds")
    print(f"Received {len(handler.chunks)} chunks from {len(handler.agents_seen)} agents")

    # Get the final result from the last chunk
    final_result = handler.chunks[-1] if handler.chunks else {}

    # Print the final response
    if "messages" in final_result and len(final_result["messages"]) > 0:
        final_message = final_result["messages"][-1]["content"]
        print("\nFinal response:")
        print(final_message)

    return final_result


async def test_mcp_supervisor():
    """Test the MCP supervisor with a complex query."""
    print("\n=== Testing MCP Supervisor ===")

    # Create agents
    search_agent = SearchAgent(
        config=SearchAgentConfig(
            provider="serper",
            llm_provider="openai",
            openai_model="gpt-4o"
        )
    )

    image_agent = ImageGenerationAgent(
        config=ImageGenerationAgentConfig(
            provider="dalle",
            save_images=True
        )
    )

    # Create a supervisor with MCP enabled
    supervisor = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=True,
            mcp_mode="mcp",
            complexity_threshold=0.1  # Set low to ensure MCP is used
        ),
        agents={
            "search_agent": search_agent,
            "image_generation_agent": image_agent
        }
    )

    # Create a streaming handler
    handler = StreamingHandler()

    # Test query
    query = "Research quantum computing and generate an image of a quantum computer"

    print(f"Query: {query}")
    print("Running test with streaming...")

    # Start time
    start_time = time.time()

    # Process the query with streaming
    async for chunk in supervisor.astream(query):
        handler.on_chunk(chunk)

    # End time
    end_time = time.time()

    # Print results
    print(f"\nTest completed in {end_time - start_time:.2f} seconds")
    print(f"Received {len(handler.chunks)} chunks from {len(handler.agents_seen)} agents")

    # Get the final result from the last chunk
    final_result = handler.chunks[-1] if handler.chunks else {}

    # Print the final response
    if "messages" in final_result and len(final_result["messages"]) > 0:
        final_message = final_result["messages"][-1]["content"]
        print("\nFinal response:")
        print(final_message)

    # Print image URL if available
    if "agent_outputs" in final_result and "image_generation_agent" in final_result["agent_outputs"]:
        image_output = final_result["agent_outputs"]["image_generation_agent"]
        if "image_url" in image_output:
            print(f"\nImage URL: {image_output['image_url']}")

    return final_result


async def test_parallel_supervisor():
    """Test the parallel supervisor with a complex query."""
    print("\n=== Testing Parallel Supervisor ===")

    # Create agents
    search_agent = SearchAgent(
        config=SearchAgentConfig(
            provider="serper",
            llm_provider="openai",
            openai_model="gpt-4o"
        )
    )

    image_agent = ImageGenerationAgent(
        config=ImageGenerationAgentConfig(
            provider="dalle",
            save_images=True
        )
    )

    # Create a parallel supervisor
    supervisor = ParallelSupervisor(
        config=ParallelSupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=True
        ),
        agents={
            "search_agent": search_agent,
            "image_generation_agent": image_agent
        }
    )

    # Create a streaming handler
    handler = StreamingHandler()

    # Test query
    query = "Research quantum computing and generate an image of a quantum computer"

    print(f"Query: {query}")
    print("Running test with streaming...")

    # Start time
    start_time = time.time()

    # Process the query with streaming
    async for chunk in supervisor.astream(query):
        handler.on_chunk(chunk)

    # End time
    end_time = time.time()

    # Print results
    print(f"\nTest completed in {end_time - start_time:.2f} seconds")
    print(f"Received {len(handler.chunks)} chunks from {len(handler.agents_seen)} agents")

    # Get the final result from the last chunk
    final_result = handler.chunks[-1] if handler.chunks else {}

    # Print the final response
    if "messages" in final_result and len(final_result["messages"]) > 0:
        final_message = final_result["messages"][-1]["content"]
        print("\nFinal response:")
        print(final_message)

    # Print image URL if available
    if "agent_outputs" in final_result and "image_generation_agent" in final_result["agent_outputs"]:
        image_output = final_result["agent_outputs"]["image_generation_agent"]
        if "image_url" in image_output:
            print(f"\nImage URL: {image_output['image_url']}")

    return final_result


async def main():
    """Run all tests."""
    # Test standard supervisor
    await test_standard_supervisor()

    # Skip MCP and parallel tests for now
    # await test_mcp_supervisor()
    # await test_parallel_supervisor()


if __name__ == "__main__":
    asyncio.run(main())
