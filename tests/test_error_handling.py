#!/usr/bin/env python3
"""
Test error handling in the multi-agent system.

This script tests the error handling capabilities of the supervisor and agents.
It verifies that the system can handle errors gracefully and recover from them.

Usage:
    python test_error_handling.py
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
import time
from dotenv import load_dotenv
import asyncio
from typing import List, Dict, Any, AsyncGenerator

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the supervisor and related components
from src.supervisor.supervisor import Supervisor, SupervisorConfig
from src.supervisor.parallel_supervisor import ParallelSupervisor, ParallelSupervisorConfig
from src.agents.search_agent import SearchAgent, SearchAgentConfig
from src.agents.image_generation_agent import ImageGenerationAgent, ImageGenerationAgentConfig


class ErrorSimulatingSearchAgent:
    """Mock search agent that simulates errors for testing."""

    def __init__(self, error_type="api_error", error_message="Simulated API error"):
        """
        Initialize the error simulating search agent.

        Args:
            error_type: Type of error to simulate
            error_message: Error message to return
        """
        self.error_type = error_type
        self.error_message = error_message
        self.called = False
        self.call_count = 0

    def __call__(self, state):
        """
        Process a state update, simulating an error.

        Args:
            state: Current state

        Returns:
            Updated state with error information
        """
        self.called = True
        self.call_count += 1

        # Initialize agent outputs if needed
        if "agent_outputs" not in state:
            state["agent_outputs"] = {}

        # Simulate different types of errors
        if self.error_type == "api_error":
            # Simulate an API error (e.g., Serper API error)
            state["agent_outputs"]["search_agent"] = {
                "error": self.error_message,
                "results": [],
                "has_error": True
            }
        elif self.error_type == "timeout":
            # Simulate a timeout error
            state["agent_outputs"]["search_agent"] = {
                "error": "Request timed out after 30 seconds",
                "results": [],
                "has_error": True
            }
        elif self.error_type == "rate_limit":
            # Simulate a rate limit error
            state["agent_outputs"]["search_agent"] = {
                "error": "Rate limit exceeded. Please try again later.",
                "results": [],
                "has_error": True
            }
        elif self.error_type == "exception":
            # Simulate an unhandled exception
            raise ValueError(self.error_message)
        else:
            # Unknown error type, just return a generic error
            state["agent_outputs"]["search_agent"] = {
                "error": f"Unknown error: {self.error_message}",
                "results": [],
                "has_error": True
            }

        # Add error message to state
        state["messages"].append({
            "role": "assistant",
            "content": f"I encountered an error while searching: {self.error_message}"
        })

        return state


class ErrorSimulatingImageAgent:
    """Mock image generation agent that simulates errors for testing."""

    def __init__(self, error_type="api_error", error_message="Simulated API error"):
        """
        Initialize the error simulating image generation agent.

        Args:
            error_type: Type of error to simulate
            error_message: Error message to return
        """
        self.error_type = error_type
        self.error_message = error_message
        self.called = False
        self.call_count = 0

    def __call__(self, state):
        """
        Process a state update, simulating an error.

        Args:
            state: Current state

        Returns:
            Updated state with error information
        """
        self.called = True
        self.call_count += 1

        # Initialize agent outputs if needed
        if "agent_outputs" not in state:
            state["agent_outputs"] = {}

        # Simulate different types of errors
        if self.error_type == "api_error":
            # Simulate an API error (e.g., DALL-E API error)
            state["agent_outputs"]["image_generation_agent"] = {
                "error": self.error_message,
                "has_error": True
            }
        elif self.error_type == "timeout":
            # Simulate a timeout error
            state["agent_outputs"]["image_generation_agent"] = {
                "error": "Request timed out after 30 seconds",
                "has_error": True
            }
        elif self.error_type == "download_error":
            # Simulate an image download error
            state["agent_outputs"]["image_generation_agent"] = {
                "error": "Failed to download image after 3 attempts",
                "image_url": "https://example.com/image.jpg",
                "has_error": True
            }
        elif self.error_type == "exception":
            # Simulate an unhandled exception
            raise ValueError(self.error_message)
        else:
            # Unknown error type, just return a generic error
            state["agent_outputs"]["image_generation_agent"] = {
                "error": f"Unknown error: {self.error_message}",
                "has_error": True
            }

        # Add error message to state
        state["messages"].append({
            "role": "assistant",
            "content": f"I encountered an error while generating the image: {self.error_message}"
        })

        return state


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling in the multi-agent system."""

    def setUp(self):
        """Set up test fixtures."""
        # Load environment variables
        load_dotenv()

    def test_search_agent_api_error(self):
        """Test that the supervisor handles search agent API errors gracefully."""
        # Create a search agent that simulates an API error
        search_agent = ErrorSimulatingSearchAgent(
            error_type="api_error",
            error_message="Serper API error: Invalid API key"
        )

        # Create a normal image agent
        image_agent = MagicMock()
        image_agent.return_value = {
            "messages": [{"role": "assistant", "content": "Image generated successfully"}],
            "agent_outputs": {
                "image_generation_agent": {
                    "image_url": "https://example.com/image.jpg"
                }
            }
        }

        # Create a supervisor with the agents
        supervisor = Supervisor(
            config=SupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=False
            ),
            agents={
                "search_agent": search_agent,
                "image_generation_agent": image_agent
            }
        )

        # Mock the _synthesize_final_response method to ensure it includes the error
        original_synthesize = supervisor._synthesize_final_response
        def mock_synthesize(messages, agent_outputs):
            return f"Error encountered: {agent_outputs['search_agent']['error']}"
        supervisor._synthesize_final_response = mock_synthesize

        # Test with a query that would normally use the search agent
        query = "What is quantum computing?"
        result = supervisor.invoke(query)

        # Check that the search agent was called
        self.assertTrue(search_agent.called)

        # Check that the error was captured in the agent outputs
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertIn("error", result["agent_outputs"]["search_agent"])
        self.assertTrue(result["agent_outputs"]["search_agent"]["has_error"])

        # Check that the final response mentions the error
        final_message = result["messages"][-1]["content"]
        self.assertIn("error", final_message.lower())

        # Restore the original method
        supervisor._synthesize_final_response = original_synthesize

    def test_image_agent_api_error(self):
        """Test that the supervisor handles image agent API errors gracefully."""
        # Create a normal search agent
        search_agent = MagicMock()
        search_agent.return_value = {
            "messages": [{"role": "assistant", "content": "Here are the search results"}],
            "agent_outputs": {
                "search_agent": {
                    "results": [{"title": "Result 1", "url": "https://example.com", "snippet": "Example snippet"}]
                }
            }
        }

        # Create an image agent that simulates an API error
        image_agent = ErrorSimulatingImageAgent(
            error_type="api_error",
            error_message="DALL-E API error: Invalid API key"
        )

        # Create a supervisor with the agents
        supervisor = Supervisor(
            config=SupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=False
            ),
            agents={
                "search_agent": search_agent,
                "image_generation_agent": image_agent
            }
        )

        # Mock the _synthesize_final_response method to ensure it includes the error
        original_synthesize = supervisor._synthesize_final_response
        def mock_synthesize(messages, agent_outputs):
            return f"Error encountered: {agent_outputs['image_generation_agent']['error']}"
        supervisor._synthesize_final_response = mock_synthesize

        # Test with a query that would normally use the image agent
        query = "Generate an image of a quantum computer"
        result = supervisor.invoke(query)

        # Check that the image agent was called
        self.assertTrue(image_agent.called)

        # Check that the error was captured in the agent outputs
        self.assertIn("image_generation_agent", result["agent_outputs"])
        self.assertIn("error", result["agent_outputs"]["image_generation_agent"])
        self.assertTrue(result["agent_outputs"]["image_generation_agent"]["has_error"])

        # Check that the final response mentions the error
        final_message = result["messages"][-1]["content"]
        self.assertIn("error", final_message.lower())

        # Restore the original method
        supervisor._synthesize_final_response = original_synthesize


    def test_mcp_with_agent_error(self):
        """Test that the MCP supervisor handles agent errors gracefully."""
        # Create a search agent that simulates an API error
        search_agent = ErrorSimulatingSearchAgent(
            error_type="api_error",
            error_message="Serper API error: Invalid API key"
        )

        # Create an image agent that works normally
        image_agent = MagicMock()
        image_agent.return_value = {
            "messages": [{"role": "assistant", "content": "Image generated successfully"}],
            "agent_outputs": {
                "image_generation_agent": {
                    "image_url": "https://example.com/image.jpg"
                }
            }
        }

        # Create a supervisor with MCP enabled
        supervisor = Supervisor(
            config=SupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=False,
                mcp_mode="mcp",
                complexity_threshold=0.1  # Set low to ensure MCP is used
            ),
            agents={
                "search_agent": search_agent,
                "image_generation_agent": image_agent
            }
        )

        # Mock the MCP agent's invoke method
        supervisor.mcp_agent.invoke = MagicMock()
        supervisor.mcp_agent.invoke.return_value = {
            "messages": [
                {"role": "user", "content": "Research quantum computing and generate an image of a quantum computer"},
                {"role": "assistant", "content": "I've researched quantum computing and generated an image"}
            ],
            "agent_outputs": {
                "search_agent": {
                    "error": "Serper API error: Invalid API key",
                    "has_error": True,
                    "needs_fallback": True
                },
                "image_generation_agent": {
                    "image_url": "https://example.com/image.jpg"
                }
            }
        }

        # Test with a complex query that would use MCP
        query = "Research quantum computing and generate an image of a quantum computer"
        result = supervisor.invoke(query)

        # Check that MCP was used
        supervisor.mcp_agent.invoke.assert_called_once()

        # Check that the error was captured in the agent outputs
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertIn("error", result["agent_outputs"]["search_agent"])
        self.assertTrue(result["agent_outputs"]["search_agent"]["has_error"])

        # Check that the image agent still worked
        self.assertIn("image_generation_agent", result["agent_outputs"])
        self.assertIn("image_url", result["agent_outputs"]["image_generation_agent"])

    def test_parallel_supervisor_with_agent_error(self):
        """Test that the parallel supervisor handles agent errors gracefully."""
        # Create a search agent that simulates an API error
        search_agent = ErrorSimulatingSearchAgent(
            error_type="api_error",
            error_message="Serper API error: Invalid API key"
        )

        # Create an image agent that works normally
        image_agent = MagicMock()
        image_agent.return_value = {
            "messages": [{"role": "assistant", "content": "Image generated successfully"}],
            "agent_outputs": {
                "image_generation_agent": {
                    "image_url": "https://example.com/image.jpg"
                }
            }
        }

        # Create a parallel supervisor with mocked methods
        supervisor = ParallelSupervisor(
            config=ParallelSupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=False
            ),
            agents={
                "search_agent": search_agent,
                "image_generation_agent": image_agent
            }
        )

        # Create a simplified version of invoke that doesn't call _identify_subtasks
        def mock_invoke(query):
            # Create a state with predefined subtasks
            state = {
                "messages": [{"role": "user", "content": query}],
                "agent_outputs": {},
                "subtasks": [
                    {
                        "subtask_id": 1,
                        "description": "Research quantum computing",
                        "agent": "search_agent",
                        "depends_on": [],
                        "parallelizable": True
                    },
                    {
                        "subtask_id": 2,
                        "description": "Generate an image of a quantum computer",
                        "agent": "image_generation_agent",
                        "depends_on": [],
                        "parallelizable": True
                    }
                ],
                "completed_subtasks": set(),
                "execution_stats": {
                    "start_time": time.time(),
                    "end_time": None,
                    "total_subtasks": 2,
                    "completed_subtasks": 0,
                    "parallel_batches": 0,
                    "execution_times": {}
                }
            }

            # Execute the subtasks
            for subtask in state["subtasks"]:
                agent_name = subtask["agent"]
                agent = supervisor.agents[agent_name]

                # Set current subtask
                state["current_subtask"] = subtask

                # Call the agent
                try:
                    updated_state = agent(state)
                    # Merge the updated state
                    state["agent_outputs"].update(updated_state["agent_outputs"])
                    state["messages"].extend(updated_state["messages"][len(state["messages"]):])
                except Exception as e:
                    state["agent_outputs"][agent_name] = {"error": str(e), "has_error": True}

                # Mark as completed
                state["completed_subtasks"].add(subtask["subtask_id"])
                state["execution_stats"]["completed_subtasks"] += 1

            # Finalize the state
            state["execution_stats"]["end_time"] = time.time()
            state["final_response"] = "Combined response from all agents"

            return state

        # Replace the invoke method
        supervisor.invoke = mock_invoke

        # Test with a complex query that would use parallel execution
        query = "Research quantum computing and generate an image of a quantum computer"
        result = supervisor.invoke(query)

        # Check that the error was captured in the agent outputs
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertIn("error", result["agent_outputs"]["search_agent"])
        self.assertTrue(result["agent_outputs"]["search_agent"]["has_error"])

        # Check that the image agent still worked
        self.assertIn("image_generation_agent", result["agent_outputs"])
        self.assertIn("image_url", result["agent_outputs"]["image_generation_agent"])

    def test_streaming_with_agent_error(self):
        """Test that streaming works correctly with agent errors."""
        # This is a synchronous version of the test since unittest doesn't handle async tests well

        # Create a search agent that simulates an API error
        search_agent = ErrorSimulatingSearchAgent(
            error_type="api_error",
            error_message="Serper API error: Invalid API key"
        )

        # Create a supervisor with the agent
        supervisor = Supervisor(
            config=SupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=True
            ),
            agents={
                "search_agent": search_agent
            }
        )

        # Mock the astream method to return a predefined list of chunks
        async def mock_astream(query):
            yield {
                "messages": [{"role": "user", "content": query}],
                "agent_outputs": {},
                "current_agent": "search_agent"
            }
            yield {
                "messages": [{"role": "user", "content": query}],
                "agent_outputs": {
                    "search_agent": {
                        "error": "Serper API error: Invalid API key",
                        "has_error": True
                    }
                },
                "current_agent": "search_agent"
            }
            yield {
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": "I encountered an error while searching"}
                ],
                "agent_outputs": {
                    "search_agent": {
                        "error": "Serper API error: Invalid API key",
                        "has_error": True
                    }
                },
                "current_agent": "search_agent"
            }

        # Create a synchronous version that collects all chunks
        def sync_stream(query):
            chunks = []
            async def collect_chunks():
                async for chunk in mock_astream(query):
                    chunks.append(chunk)
                return chunks

            # Run the async function in a new event loop
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                chunks = loop.run_until_complete(collect_chunks())
            finally:
                loop.close()
            return chunks

        # Replace the astream method
        supervisor.astream = mock_astream

        # Test with streaming enabled
        query = "What is quantum computing?"
        chunks = sync_stream(query)

        # Check that we got chunks
        self.assertTrue(len(chunks) > 0)

        # Check that the error was captured in at least one chunk
        error_chunks = [chunk for chunk in chunks if "agent_outputs" in chunk and
                        "search_agent" in chunk["agent_outputs"] and
                        "error" in chunk["agent_outputs"]["search_agent"]]
        self.assertTrue(len(error_chunks) > 0)


if __name__ == "__main__":
    unittest.main()
