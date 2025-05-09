"""
Integration tests for the Multi-Agent Supervisor System.

This module contains integration tests that verify the supervisor's ability to:
1. Analyze user prompts
2. Create execution plans
3. Delegate tasks to appropriate agents
4. Monitor task execution
5. Integrate results from multiple agents
6. Deliver coherent final responses

Tests are provided for both the standard supervisor and MCP-based supervisor.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the supervisor and related components
from src.supervisor.supervisor import Supervisor, SupervisorConfig
from src.supervisor.parallel_supervisor import ParallelSupervisor, ParallelSupervisorConfig
from src.agents.mcp_agent import MCPAgent, MCPAgentConfig


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name, response=None):
        """Initialize the mock agent."""
        self.name = name
        self.response = response or f"Response from {name}"
        self.called = False
        self.input_state = None

    def __call__(self, state):
        """Call the agent."""
        self.called = True
        self.input_state = state.copy()

        # Add agent output to state
        if "agent_outputs" not in state:
            state["agent_outputs"] = {}

        state["agent_outputs"][self.name] = {"result": self.response}

        # Add agent message to state
        if "messages" not in state:
            state["messages"] = []

        state["messages"].append({"role": "assistant", "content": self.response})

        return state


class TestSupervisorIntegration(unittest.TestCase):
    """Integration tests for the Supervisor class."""

    def setUp(self):
        """Set up the test environment."""
        # Load environment variables
        load_dotenv()

        # Create mock agents
        self.search_agent = MockAgent("search_agent", "Search results for the query")
        self.image_agent = MockAgent("image_generation_agent", "Generated image URL: http://example.com/image.jpg")
        self.quality_agent = MockAgent("quality_agent", "Quality assessment: 85% accuracy")
        self.vector_agent = MockAgent("vector_storage_agent", "Information stored in vector database")

        # Create agent dictionary
        self.agents = {
            "search_agent": self.search_agent,
            "image_generation_agent": self.image_agent,
            "quality_agent": self.quality_agent,
            "vector_storage_agent": self.vector_agent
        }

    @patch("src.supervisor.supervisor.ChatOpenAI")
    def test_standard_supervisor_simple_query(self, mock_chat_openai):
        """Test that standard supervisor handles simple queries correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        # Mock the LLM responses
        mock_response1 = MagicMock()
        mock_response1.content = "I'll use the search_agent to find information about this."

        mock_response2 = MagicMock()
        mock_response2.content = "The capital of France is Paris."

        mock_llm.invoke.side_effect = [mock_response1, mock_response2]

        # Create a supervisor with agents
        supervisor = Supervisor(
            config=SupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=False
            ),
            agents=self.agents
        )

        # Test with a simple query
        result = supervisor.invoke("What is the capital of France?")

        # Check that the search agent was called
        self.assertTrue(self.search_agent.called)

        # Check that the result is correct
        self.assertEqual(result["messages"][-1]["content"], "The capital of France is Paris.")
        self.assertIn("search_agent", result["agent_outputs"])

    @patch("src.supervisor.supervisor.ChatOpenAI")
    @patch("src.agents.mcp_agent.ChatOpenAI")
    def test_mcp_supervisor_complex_query(self, mock_mcp_llm, mock_supervisor_llm):
        """Test that MCP supervisor handles complex queries correctly."""
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
        3. Evaluate the quality of the information - Agent: quality_agent

        Execution Plan:
        1. Execute subtask 1
        2. Execute subtask 2
        3. Execute subtask 3
        """

        mock_final_response = MagicMock()
        mock_final_response.content = "Here's information about quantum computing and an image of quantum entanglement."

        mock_m_llm.invoke.side_effect = [mock_plan_response, mock_final_response]

        # Create a supervisor with MCP enabled
        supervisor = Supervisor(
            config=SupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=False,
                mcp_mode="mcp",
                complexity_threshold=0.7
            ),
            agents=self.agents
        )

        # Test with a complex query
        result = supervisor.invoke("Research quantum computing and create an image of quantum entanglement, then evaluate the quality of the information.")

        # Check that the agents were called
        self.assertTrue(self.search_agent.called)
        self.assertTrue(self.image_agent.called)
        self.assertTrue(self.quality_agent.called)

        # Check that the result is correct
        self.assertEqual(result["messages"][-1]["content"], "Here's information about quantum computing and an image of quantum entanglement.")
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertIn("image_generation_agent", result["agent_outputs"])
        self.assertIn("quality_agent", result["agent_outputs"])

    @patch("src.supervisor.parallel_supervisor.ChatOpenAI")
    def test_parallel_supervisor(self, mock_chat_openai):
        """Test that parallel supervisor handles complex queries with parallelization."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        # Mock the LLM responses
        mock_plan_response = MagicMock()
        mock_plan_response.content = json.dumps([
            {"subtask_id": 1, "description": "Search for information about climate change", "agent": "search_agent", "depends_on": [], "parallelizable": True, "parallel_group": "searches"},
            {"subtask_id": 2, "description": "Generate an image of a sustainable city", "agent": "image_generation_agent", "depends_on": [], "parallelizable": True},
            {"subtask_id": 3, "description": "Evaluate the quality of the information", "agent": "quality_agent", "depends_on": [1], "parallelizable": False}
        ])

        mock_final_response = MagicMock()
        mock_final_response.content = "Here's information about climate change and an image of a sustainable city."

        mock_llm.invoke.side_effect = [mock_plan_response, mock_final_response]

        # Create a parallel supervisor with agents
        supervisor = ParallelSupervisor(
            config=ParallelSupervisorConfig(
                llm_provider="openai",
                openai_model="gpt-4o",
                streaming=False
            ),
            agents=self.agents
        )

        # Test with a complex query
        result = supervisor.invoke("Search for information about climate change, then generate an image of a sustainable city, and finally evaluate the quality of the information")

        # Check that the agents were called
        self.assertTrue(self.search_agent.called)
        self.assertTrue(self.image_agent.called)
        self.assertTrue(self.quality_agent.called)

        # Check that the result is correct
        self.assertEqual(result["messages"][-1]["content"], "Here's information about climate change and an image of a sustainable city.")
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertIn("image_generation_agent", result["agent_outputs"])
        self.assertIn("quality_agent", result["agent_outputs"])

        # Check that execution stats are recorded
        self.assertIn("execution_stats", result)
        self.assertIn("parallel_batches", result["execution_stats"])
        self.assertIn("total_execution_time", result["execution_stats"])


if __name__ == "__main__":
    unittest.main()
