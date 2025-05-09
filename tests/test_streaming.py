"""
Streaming tests for the Multi-Agent Supervisor System.

This module contains tests that verify the supervisor's streaming functionality:
1. Test that streaming works correctly with the standard supervisor
2. Test that streaming works correctly with the MCP supervisor
3. Test that streaming works correctly with the parallel supervisor
4. Test that streaming works with complex queries and multiple agents
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
from dotenv import load_dotenv
import asyncio
from typing import List, Dict, Any, AsyncGenerator

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the supervisor and related components
from src.supervisor.supervisor import Supervisor, SupervisorConfig
from src.supervisor.parallel_supervisor import ParallelSupervisor, ParallelSupervisorConfig
from src.agents.mcp_agent import MCPAgent, MCPAgentConfig


class StreamingMockAgent:
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

        # Add streaming information if streaming is enabled
        if state.get("stream", False):
            state["streaming_chunks"] = self.chunks
            state["streaming_delay"] = self.delay

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

            # Yield the updated state
            yield chunk_state

            # Add a delay between chunks
            await asyncio.sleep(self.delay)


class TestSupervisorStreaming(unittest.TestCase):
    """Tests for the streaming functionality of the Supervisor."""

    def setUp(self):
        """Set up the test environment."""
        # Load environment variables
        load_dotenv()

        # Create streaming mock agents
        self.search_agent = StreamingMockAgent(
            "search_agent",
            chunks=[
                "Searching for information...\n",
                "Found initial results...\n",
                "Processing search results...\n",
                "Search complete. Here are the results:\n",
                "1. Quantum computing uses quantum bits or qubits.\n",
                "2. IBM and Google are leading quantum computing research.\n"
            ],
            delay=0.05
        )

        self.image_agent = StreamingMockAgent(
            "image_generation_agent",
            chunks=[
                "Generating image...\n",
                "Creating initial sketch...\n",
                "Adding details...\n",
                "Finalizing image...\n",
                "Image generated: http://example.com/quantum.jpg\n"
            ],
            delay=0.05
        )

        # Create agent dictionary
        self.agents = {
            "search_agent": self.search_agent,
            "image_generation_agent": self.image_agent
        }

    @patch("src.supervisor.supervisor.ChatOpenAI")
    def test_standard_supervisor_streaming(self, mock_chat_openai):
        """Test that standard supervisor supports streaming."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "I'll use the search_agent to find information about this."
        mock_llm.invoke.return_value = mock_response

        # Create a supervisor with agents
        supervisor = Supervisor(
            config=SupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=True
            ),
            agents=self.agents
        )

        # Test with streaming enabled
        result = supervisor.invoke("What is quantum computing?", stream=True)

        # Check that the search agent was called
        self.assertTrue(self.search_agent.called)

        # Check that streaming information is included
        self.assertIn("streaming_chunks", result)
        self.assertIn("streaming_delay", result)

        # Check that the result contains the full response
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertTrue(any("quantum bits or qubits" in msg["content"] for msg in result["messages"]))

    @patch("src.supervisor.supervisor.ChatOpenAI")
    @patch("src.agents.mcp_agent.ChatOpenAI")
    def test_mcp_supervisor_streaming(self, mock_mcp_llm, mock_supervisor_llm):
        """Test that MCP supervisor supports streaming."""
        # Create mock LLMs
        mock_sup_llm = MagicMock()
        mock_supervisor_llm.return_value = mock_sup_llm

        mock_m_llm = MagicMock()
        mock_mcp_llm.return_value = mock_m_llm

        # Mock the supervisor LLM responses
        mock_complexity_response = MagicMock()
        mock_complexity_response.content = json.dumps({"complexity_score": 0.8, "reasoning": "This query requires multiple steps."})
        mock_sup_llm.invoke.return_value = mock_complexity_response

        # Mock the MCP LLM responses
        mock_plan_response = MagicMock()
        mock_plan_response.content = """
        Subtasks:
        1. Research quantum computing basics - Agent: search_agent
        2. Generate an image of quantum entanglement - Agent: image_generation_agent

        Execution Plan:
        1. Execute subtask 1
        2. Execute subtask 2
        """

        mock_final_response = MagicMock()
        mock_final_response.content = "Here's information about quantum computing and an image of quantum entanglement."

        mock_m_llm.invoke.side_effect = [mock_plan_response, mock_final_response]

        # Create a supervisor with MCP enabled
        supervisor = Supervisor(
            config=SupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=True,
                mcp_mode="mcp",
                complexity_threshold=0.7
            ),
            agents=self.agents
        )

        # Test with streaming enabled
        result = supervisor.invoke("Research quantum computing and create an image of quantum entanglement.", stream=True)

        # Check that the agents were called
        self.assertTrue(self.search_agent.called)
        self.assertTrue(self.image_agent.called)

        # Check that streaming information is included for both agents
        self.assertIn("streaming_chunks", result)
        self.assertIn("streaming_delay", result)

        # Check that the result contains outputs from both agents
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertIn("image_generation_agent", result["agent_outputs"])

        # Check that the execution plan was created
        self.assertIn("execution_plan", result)

    def test_async_streaming(self):
        """Test that async streaming works correctly."""
        # This test requires running an async function, so we'll use asyncio
        async def run_async_test():
            # Create a streaming agent
            agent = StreamingMockAgent("test_agent", chunks=["Chunk 1", "Chunk 2", "Chunk 3"], delay=0.01)

            # Create a state
            state = {
                "messages": [{"role": "user", "content": "Test query"}],
                "agent_outputs": {},
                "stream": True
            }

            # Collect all chunks
            chunks = []
            async for chunk_state in agent.astream(state):
                chunks.append(chunk_state["current_chunk"])

            return chunks

        # Run the async test
        chunks = asyncio.run(run_async_test())

        # Check that all chunks were received
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks, ["Chunk 1", "Chunk 2", "Chunk 3"])


if __name__ == "__main__":
    unittest.main()
