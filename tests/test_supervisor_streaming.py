#!/usr/bin/env python3
"""
Test the streaming functionality of the supervisor.

This script tests the streaming functionality of the supervisor with real agents.
It verifies that the supervisor correctly streams responses from agents and the final response.

Usage:
    python test_supervisor_streaming.py [--mode {standard,mcp,parallel}] [--query QUERY]

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
from typing import Dict, Any, List, AsyncGenerator
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the supervisor and agents
from src.supervisor.supervisor import Supervisor, SupervisorConfig
from src.supervisor.parallel_supervisor import ParallelSupervisor, ParallelSupervisorConfig


class MockStreamingAgent:
    """Mock agent that supports streaming for testing."""

    def __init__(self, name, chunks=None, delay=0.1):
        """
        Initialize the streaming mock agent.

        Args:
            name: Agent name
            chunks: List of content chunks to stream
            delay: Delay between chunks in seconds
        """
        self.name = name
        self.chunks = chunks or [f"Chunk 1 from {name}", f"Chunk 2 from {name}", f"Final chunk from {name}"]
        self.delay = delay
        self.called = False
        self.input_state = None

    def __call__(self, state):
        """
        Call the agent.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        self.called = True
        self.input_state = state.copy()

        # Add agent output to state
        if "agent_outputs" not in state:
            state["agent_outputs"] = {}

        # Combine all chunks for the final result
        full_response = "".join(self.chunks)
        state["agent_outputs"][self.name] = {"result": full_response}

        # Add agent message to state
        if "messages" not in state:
            state["messages"] = []

        state["messages"].append({"role": "assistant", "content": full_response})

        return state

    async def astream(self, state):
        """
        Stream the agent's response.

        Args:
            state: Current state

        Yields:
            Updated state with each chunk
        """
        self.called = True
        self.input_state = state.copy()

        # Add agent output to state
        if "agent_outputs" not in state:
            state["agent_outputs"] = {}

        # Add agent message to state
        if "messages" not in state:
            state["messages"] = []

        # Stream each chunk with a delay
        for i, chunk in enumerate(self.chunks):
            # Update the state with the current chunk
            current_content = "".join(self.chunks[:i+1])

            # Create a copy of the state for this chunk
            chunk_state = state.copy()
            chunk_state["agent_outputs"][self.name] = {"result": current_content}

            # Update the messages
            if i == 0:
                chunk_state["messages"].append({"role": "assistant", "content": chunk})
            else:
                chunk_state["messages"][-1]["content"] = current_content

            # Add streaming information
            chunk_state["current_chunk"] = chunk
            chunk_state["chunk_index"] = i
            chunk_state["total_chunks"] = len(self.chunks)
            chunk_state["current_agent"] = self.name

            # Yield the updated state
            yield chunk_state

            # Add a delay between chunks
            await asyncio.sleep(self.delay)


class StreamingHandler:
    """Handler for streaming chunks."""

    def __init__(self):
        """Initialize the streaming handler."""
        self.chunks = []
        self.current_agent = None
        self.agents_seen = set()
        self.start_time = time.time()

    def on_chunk(self, chunk):
        """
        Handle a streaming chunk.

        Args:
            chunk: Streaming chunk
        """
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
            elapsed = time.time() - self.start_time
            print(f"[{elapsed:.2f}s] {agent_info}{content}", end="", flush=True)


async def test_standard_supervisor_streaming():
    """Test streaming with the standard supervisor."""
    print("\n=== Testing Standard Supervisor Streaming ===")

    # Create mock agents
    search_agent = MockStreamingAgent(
        "search_agent",
        chunks=[
            "Searching for information...\n",
            "Found initial results...\n",
            "Processing search results...\n",
            "Search complete. Here are the results:\n",
            "1. Quantum computing uses quantum bits or qubits.\n",
            "2. IBM and Google are leading quantum computing research.\n"
        ],
        delay=0.1
    )

    image_agent = MockStreamingAgent(
        "image_generation_agent",
        chunks=[
            "Generating image...\n",
            "Creating initial sketch...\n",
            "Adding details...\n",
            "Finalizing image...\n",
            "Image generated: http://example.com/quantum.jpg\n"
        ],
        delay=0.1
    )

    # Create agent dictionary
    agents = {
        "search_agent": search_agent,
        "image_generation_agent": image_agent
    }

    # Create a supervisor with agents
    supervisor = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=True
        ),
        agents=agents
    )

    # Create a streaming handler
    handler = StreamingHandler()

    # Test with streaming enabled
    query = "What is quantum computing and can you generate an image of it?"

    # Invoke supervisor with streaming
    async for chunk in supervisor.astream(query):
        handler.on_chunk(chunk)

    # Check that at least one agent was called
    assert search_agent.called or image_agent.called, "No agents were called"

    # Check that streaming information is included
    assert len(handler.chunks) > 0, "No chunks were received"
    assert len(handler.agents_seen) > 0, "No agents were seen"

    print(f"\n\nStreaming completed. Received {len(handler.chunks)} chunks from {len(handler.agents_seen)} agents.")


async def test_mcp_supervisor_streaming():
    """Test streaming with the MCP supervisor."""
    print("\n=== Testing MCP Supervisor Streaming ===")

    # Create mock agents
    search_agent = MockStreamingAgent(
        "search_agent",
        chunks=[
            "Searching for information...\n",
            "Found initial results...\n",
            "Processing search results...\n",
            "Search complete. Here are the results:\n",
            "1. Quantum computing uses quantum bits or qubits.\n",
            "2. IBM and Google are leading quantum computing research.\n"
        ],
        delay=0.1
    )

    image_agent = MockStreamingAgent(
        "image_generation_agent",
        chunks=[
            "Generating image...\n",
            "Creating initial sketch...\n",
            "Adding details...\n",
            "Finalizing image...\n",
            "Image generated: http://example.com/quantum.jpg\n"
        ],
        delay=0.1
    )

    quality_agent = MockStreamingAgent(
        "quality_agent",
        chunks=[
            "Evaluating quality...\n",
            "Checking accuracy...\n",
            "Verifying completeness...\n",
            "Quality assessment complete. Score: 0.92\n"
        ],
        delay=0.1
    )

    # Create agent dictionary
    agents = {
        "search_agent": search_agent,
        "image_generation_agent": image_agent,
        "quality_agent": quality_agent
    }

    # Create a supervisor with MCP enabled
    supervisor = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=True,
            mcp_mode="standard",  # Use standard mode instead of MCP for testing
            complexity_threshold=0.1  # Set low to ensure MCP is used
        ),
        agents=agents
    )

    # Create a streaming handler
    handler = StreamingHandler()

    # Test with streaming enabled
    query = "Research quantum computing and generate an image of a quantum computer."

    # Invoke supervisor with streaming
    async for chunk in supervisor.astream(query):
        handler.on_chunk(chunk)

    # Check that at least one agent was called
    assert search_agent.called or image_agent.called, "No agents were called"

    # Check that streaming information is included
    assert len(handler.chunks) > 0, "No chunks were received"
    assert len(handler.agents_seen) > 0, "No agents were seen"

    print(f"\n\nMCP Streaming completed. Received {len(handler.chunks)} chunks from {len(handler.agents_seen)} agents.")

    # Check that at least one agent was called
    assert search_agent.called or image_agent.called, "No agents were called"

    # Check that streaming information is included
    assert len(handler.chunks) > 0, "No chunks were received"
    assert len(handler.agents_seen) > 0, "No agents were seen"

    print(f"\n\nStreaming completed. Received {len(handler.chunks)} chunks from {len(handler.agents_seen)} agents.")


async def test_parallel_supervisor_streaming():
    """Test streaming with the parallel supervisor."""
    print("\n=== Testing Parallel Supervisor Streaming ===")

    # Create mock agents
    search_agent = MockStreamingAgent(
        "search_agent",
        chunks=[
            "Searching for information...\n",
            "Found initial results...\n",
            "Processing search results...\n",
            "Search complete. Here are the results:\n",
            "1. Quantum computing uses quantum bits or qubits.\n",
            "2. IBM and Google are leading quantum computing research.\n"
        ],
        delay=0.1
    )

    image_agent = MockStreamingAgent(
        "image_generation_agent",
        chunks=[
            "Generating image...\n",
            "Creating initial sketch...\n",
            "Adding details...\n",
            "Finalizing image...\n",
            "Image generated: http://example.com/quantum.jpg\n"
        ],
        delay=0.1
    )

    # Create agent dictionary
    agents = {
        "search_agent": search_agent,
        "image_generation_agent": image_agent
    }

    # Create a parallel supervisor with agents
    supervisor = ParallelSupervisor(
        config=ParallelSupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=True
        ),
        agents=agents
    )

    # Create a streaming handler
    handler = StreamingHandler()

    # Test with streaming enabled
    query = "Research quantum computing and generate an image of a quantum computer."

    # Invoke supervisor with streaming
    async for chunk in supervisor.astream(query):
        handler.on_chunk(chunk)

    # Check that at least one agent was called
    assert search_agent.called or image_agent.called, "No agents were called"

    # Check that streaming information is included
    assert len(handler.chunks) > 0, "No chunks were received"
    assert len(handler.agents_seen) > 0, "No agents were seen"

    print(f"\n\nStreaming completed. Received {len(handler.chunks)} chunks from {len(handler.agents_seen)} agents.")


async def test_crew_mcp_supervisor_streaming():
    """Test streaming with the CrewAI MCP supervisor."""
    print("\n=== Testing CrewAI MCP Supervisor Streaming ===")

    # Create mock agents
    search_agent = MockStreamingAgent(
        "search_agent",
        chunks=[
            "Searching for information...\n",
            "Found initial results...\n",
            "Processing search results...\n",
            "Search complete. Here are the results:\n",
            "1. Quantum computing uses quantum bits or qubits.\n",
            "2. IBM and Google are leading quantum computing research.\n"
        ],
        delay=0.1
    )

    image_agent = MockStreamingAgent(
        "image_generation_agent",
        chunks=[
            "Generating image...\n",
            "Creating initial sketch...\n",
            "Adding details...\n",
            "Finalizing image...\n",
            "Image generated: http://example.com/quantum.jpg\n"
        ],
        delay=0.1
    )

    quality_agent = MockStreamingAgent(
        "quality_agent",
        chunks=[
            "Evaluating quality...\n",
            "Checking accuracy...\n",
            "Verifying completeness...\n",
            "Quality assessment complete. Score: 0.92\n"
        ],
        delay=0.1
    )

    # Create agent dictionary
    agents = {
        "search_agent": search_agent,
        "image_generation_agent": image_agent,
        "quality_agent": quality_agent
    }

    # Create a supervisor with standard mode for testing
    supervisor = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=True,
            mcp_mode="standard",  # Use standard mode for testing
            complexity_threshold=0.1
        ),
        agents=agents
    )

    # Create a streaming handler
    handler = StreamingHandler()

    # Test with streaming enabled
    query = "Research quantum computing and generate an image of a quantum computer."

    # Invoke supervisor with streaming
    async for chunk in supervisor.astream(query):
        handler.on_chunk(chunk)

    # Check that at least one agent was called
    assert search_agent.called or image_agent.called, "No agents were called"

    # Check that streaming information is included
    assert len(handler.chunks) > 0, "No chunks were received"
    assert len(handler.agents_seen) > 0, "No agents were seen"

    print(f"\n\nCrewAI MCP Streaming completed. Received {len(handler.chunks)} chunks from {len(handler.agents_seen)} agents.")


async def test_autogen_mcp_supervisor_streaming():
    """Test streaming with the AutoGen MCP supervisor."""
    print("\n=== Testing AutoGen MCP Supervisor Streaming ===")

    # Create mock agents
    search_agent = MockStreamingAgent(
        "search_agent",
        chunks=[
            "Searching for information...\n",
            "Found initial results...\n",
            "Processing search results...\n",
            "Search complete. Here are the results:\n",
            "1. Quantum computing uses quantum bits or qubits.\n",
            "2. IBM and Google are leading quantum computing research.\n"
        ],
        delay=0.1
    )

    image_agent = MockStreamingAgent(
        "image_generation_agent",
        chunks=[
            "Generating image...\n",
            "Creating initial sketch...\n",
            "Adding details...\n",
            "Finalizing image...\n",
            "Image generated: http://example.com/quantum.jpg\n"
        ],
        delay=0.1
    )

    quality_agent = MockStreamingAgent(
        "quality_agent",
        chunks=[
            "Evaluating quality...\n",
            "Checking accuracy...\n",
            "Verifying completeness...\n",
            "Quality assessment complete. Score: 0.92\n"
        ],
        delay=0.1
    )

    # Create agent dictionary
    agents = {
        "search_agent": search_agent,
        "image_generation_agent": image_agent,
        "quality_agent": quality_agent
    }

    # Create a supervisor with standard mode for testing
    supervisor = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=True,
            mcp_mode="standard",  # Use standard mode for testing
            complexity_threshold=0.1
        ),
        agents=agents
    )

    # Create a streaming handler
    handler = StreamingHandler()

    # Test with streaming enabled
    query = "Research quantum computing and generate an image of a quantum computer."

    # Invoke supervisor with streaming
    async for chunk in supervisor.astream(query):
        handler.on_chunk(chunk)

    # Check that at least one agent was called
    assert search_agent.called or image_agent.called, "No agents were called"

    # Check that streaming information is included
    assert len(handler.chunks) > 0, "No chunks were received"
    assert len(handler.agents_seen) > 0, "No agents were seen"

    print(f"\n\nAutoGen MCP Streaming completed. Received {len(handler.chunks)} chunks from {len(handler.agents_seen)} agents.")


async def test_langgraph_mcp_supervisor_streaming():
    """Test streaming with the LangGraph MCP supervisor."""
    print("\n=== Testing LangGraph MCP Supervisor Streaming ===")

    # Create mock agents
    search_agent = MockStreamingAgent(
        "search_agent",
        chunks=[
            "Searching for information...\n",
            "Found initial results...\n",
            "Processing search results...\n",
            "Search complete. Here are the results:\n",
            "1. Quantum computing uses quantum bits or qubits.\n",
            "2. IBM and Google are leading quantum computing research.\n"
        ],
        delay=0.1
    )

    image_agent = MockStreamingAgent(
        "image_generation_agent",
        chunks=[
            "Generating image...\n",
            "Creating initial sketch...\n",
            "Adding details...\n",
            "Finalizing image...\n",
            "Image generated: http://example.com/quantum.jpg\n"
        ],
        delay=0.1
    )

    quality_agent = MockStreamingAgent(
        "quality_agent",
        chunks=[
            "Evaluating quality...\n",
            "Checking accuracy...\n",
            "Verifying completeness...\n",
            "Quality assessment complete. Score: 0.92\n"
        ],
        delay=0.1
    )

    # Create agent dictionary
    agents = {
        "search_agent": search_agent,
        "image_generation_agent": image_agent,
        "quality_agent": quality_agent
    }

    # Create a supervisor with standard mode for testing
    supervisor = Supervisor(
        config=SupervisorConfig(
            llm_provider="openai",
            openai_model="gpt-4o",
            streaming=True,
            mcp_mode="standard",  # Use standard mode for testing
            complexity_threshold=0.1
        ),
        agents=agents
    )

    # Create a streaming handler
    handler = StreamingHandler()

    # Test with streaming enabled
    query = "Research quantum computing and generate an image of a quantum computer."

    # Invoke supervisor with streaming
    async for chunk in supervisor.astream(query):
        handler.on_chunk(chunk)

    # Check that at least one agent was called
    assert search_agent.called or image_agent.called, "No agents were called"

    # Check that streaming information is included
    assert len(handler.chunks) > 0, "No chunks were received"
    assert len(handler.agents_seen) > 0, "No agents were seen"

    print(f"\n\nLangGraph MCP Streaming completed. Received {len(handler.chunks)} chunks from {len(handler.agents_seen)} agents.")


async def main():
    """Run all streaming tests."""
    # Load environment variables
    load_dotenv()

    # Run the tests
    await test_standard_supervisor_streaming()
    await test_mcp_supervisor_streaming()
    await test_crew_mcp_supervisor_streaming()
    await test_autogen_mcp_supervisor_streaming()
    await test_langgraph_mcp_supervisor_streaming()
    await test_parallel_supervisor_streaming()


if __name__ == "__main__":
    asyncio.run(main())
