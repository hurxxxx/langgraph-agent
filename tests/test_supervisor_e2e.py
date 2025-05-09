"""
End-to-End tests for the Multi-Agent Supervisor System.

This module contains end-to-end tests that verify the supervisor's ability to:
1. Analyze complex user prompts
2. Create detailed execution plans
3. Delegate tasks to appropriate agents
4. Monitor task execution and adapt as needed
5. Integrate results from multiple agents
6. Deliver coherent final responses

These tests use more realistic scenarios and verify the complete flow from user prompt to final response.
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


class DetailedMockAgent:
    """Detailed mock agent for testing that records all inputs and can provide staged responses."""

    def __init__(self, name, responses=None):
        """
        Initialize the detailed mock agent.

        Args:
            name: Agent name
            responses: List of responses to return in sequence, or a single response
        """
        self.name = name
        self.call_count = 0
        self.call_history = []

        # Handle both single response and list of responses
        if responses is None:
            self.responses = [f"Response from {name}"]
        elif isinstance(responses, list):
            self.responses = responses
        else:
            self.responses = [responses]

    def __call__(self, state):
        """
        Call the agent.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        # Record the call
        self.call_count += 1
        self.call_history.append(state.copy())

        # Get the appropriate response
        response_index = min(self.call_count - 1, len(self.responses) - 1)
        response = self.responses[response_index]

        # Add agent output to state
        if "agent_outputs" not in state:
            state["agent_outputs"] = {}

        state["agent_outputs"][self.name] = {"result": response}

        # Add agent message to state
        if "messages" not in state:
            state["messages"] = []

        state["messages"].append({"role": "assistant", "content": response})

        return state


class TestSupervisorE2E(unittest.TestCase):
    """End-to-End tests for the Supervisor class."""

    def setUp(self):
        """Set up the test environment."""
        # Load environment variables
        load_dotenv()

        # Create detailed mock agents with realistic responses
        self.search_agent = DetailedMockAgent(
            "search_agent",
            [
                "Search results for quantum computing:\n1. Quantum computing uses quantum bits or qubits.\n2. IBM and Google are leading quantum computing research.\n3. Quantum computers can solve certain problems exponentially faster than classical computers.",
                "Search results for climate change:\n1. Global temperatures have increased by about 1°C since pre-industrial times.\n2. Sea levels are rising at an accelerating rate.\n3. Extreme weather events are becoming more frequent and intense."
            ]
        )

        self.image_agent = DetailedMockAgent(
            "image_generation_agent",
            [
                "Generated image of quantum entanglement: http://example.com/quantum.jpg",
                "Generated image of sustainable city: http://example.com/sustainable_city.jpg"
            ]
        )

        self.quality_agent = DetailedMockAgent(
            "quality_agent",
            [
                "Quality assessment of quantum computing information:\n- Accuracy: 92%\n- Completeness: 85%\n- Relevance: 95%\nOverall quality score: 90.7%",
                "Quality assessment of climate change information:\n- Accuracy: 95%\n- Completeness: 88%\n- Relevance: 97%\nOverall quality score: 93.3%"
            ]
        )

        self.vector_agent = DetailedMockAgent(
            "vector_storage_agent",
            [
                "Information about quantum computing stored in vector database with ID: qc-12345",
                "Information about climate change stored in vector database with ID: cc-67890"
            ]
        )

        # Create agent dictionary
        self.agents = {
            "search_agent": self.search_agent,
            "image_generation_agent": self.image_agent,
            "quality_agent": self.quality_agent,
            "vector_storage_agent": self.vector_agent
        }

    @patch("src.supervisor.supervisor.ChatOpenAI")
    @patch("src.agents.mcp_agent.ChatOpenAI")
    def test_complex_research_task(self, mock_mcp_llm, mock_supervisor_llm):
        """Test that supervisor handles a complex research task correctly."""
        # Create mock LLMs
        mock_sup_llm = MagicMock()
        mock_supervisor_llm.return_value = mock_sup_llm

        mock_m_llm = MagicMock()
        mock_mcp_llm.return_value = mock_m_llm

        # Mock the supervisor LLM responses
        mock_complexity_response = MagicMock()
        mock_complexity_response.content = json.dumps({"complexity_score": 0.85, "reasoning": "This query requires multiple steps including research, image generation, and quality assessment."})
        mock_sup_llm.invoke.return_value = mock_complexity_response

        # Mock the MCP LLM responses for planning and synthesis
        mock_plan_response = MagicMock()
        mock_plan_response.content = """
        Subtasks:
        1. Research quantum computing basics - Agent: search_agent
           Description: Find information about quantum computing principles, current state, and applications
           Dependencies: None
           Expected output: Comprehensive information about quantum computing

        2. Generate an image of quantum entanglement - Agent: image_generation_agent
           Description: Create a visual representation of quantum entanglement
           Dependencies: None
           Expected output: URL to generated image

        3. Evaluate the quality of the research information - Agent: quality_agent
           Description: Assess the accuracy, completeness, and relevance of the quantum computing information
           Dependencies: [1]
           Expected output: Quality assessment with scores

        4. Store the research information in vector database - Agent: vector_storage_agent
           Description: Save the quantum computing information for future retrieval
           Dependencies: [1]
           Expected output: Confirmation of storage with reference ID

        Execution Plan:
        1. Execute subtasks 1 and 2 in parallel (no dependencies)
        2. After subtask 1 completes, execute subtasks 3 and 4 in parallel (both depend on subtask 1)
        3. Synthesize final response incorporating all results
        """

        mock_final_response = MagicMock()
        mock_final_response.content = """
        # Quantum Computing Research Summary

        ## Key Information
        Quantum computing uses quantum bits or qubits, which can exist in multiple states simultaneously due to superposition. Major companies like IBM and Google are leading research in this field. Quantum computers have the potential to solve certain problems exponentially faster than classical computers.

        ## Visual Representation
        An image of quantum entanglement has been generated and is available at: http://example.com/quantum.jpg

        ## Quality Assessment
        The information has been evaluated with the following scores:
        - Accuracy: 92%
        - Completeness: 85%
        - Relevance: 95%
        - Overall quality score: 90.7%

        ## Storage Information
        The research has been stored in our vector database for future reference with ID: qc-12345
        """

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

        # Test with a complex research task
        result = supervisor.invoke(
            "Research quantum computing in detail, including its principles, current state, and applications. "
            "Also generate an image of quantum entanglement, evaluate the quality of the information, "
            "and store the research in a vector database for future reference."
        )

        # Check that all agents were called
        self.assertEqual(self.search_agent.call_count, 1)
        self.assertEqual(self.image_agent.call_count, 1)
        self.assertEqual(self.quality_agent.call_count, 1)
        self.assertEqual(self.vector_agent.call_count, 1)

        # Check that the result contains outputs from all agents
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertIn("image_generation_agent", result["agent_outputs"])
        self.assertIn("quality_agent", result["agent_outputs"])
        self.assertIn("vector_storage_agent", result["agent_outputs"])

        # Check that the execution plan was created and stored
        self.assertIn("execution_plan", result)

        # Check that the final response is comprehensive
        final_response = result["messages"][-1]["content"]
        self.assertIn("Quantum Computing Research Summary", final_response)
        self.assertIn("quantum bits or qubits", final_response)
        self.assertIn("http://example.com/quantum.jpg", final_response)
        self.assertIn("Quality Assessment", final_response)
        self.assertIn("vector database", final_response)

    @patch("src.supervisor.parallel_supervisor.ChatOpenAI")
    def test_parallel_climate_change_task(self, mock_chat_openai):
        """Test that parallel supervisor handles a climate change research task with parallelization."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        # Mock the LLM responses
        mock_plan_response = MagicMock()
        mock_plan_response.content = json.dumps([
            {
                "subtask_id": 1,
                "description": "Research current climate change data and trends",
                "agent": "search_agent",
                "depends_on": [],
                "parallelizable": True,
                "parallel_group": "research"
            },
            {
                "subtask_id": 2,
                "description": "Generate an image of a sustainable city with renewable energy",
                "agent": "image_generation_agent",
                "depends_on": [],
                "parallelizable": True
            },
            {
                "subtask_id": 3,
                "description": "Evaluate the quality of the climate change information",
                "agent": "quality_agent",
                "depends_on": [1],
                "parallelizable": False
            },
            {
                "subtask_id": 4,
                "description": "Store the climate change information in vector database",
                "agent": "vector_storage_agent",
                "depends_on": [1, 3],
                "parallelizable": False
            }
        ])

        mock_final_response = MagicMock()
        mock_final_response.content = """
        # Climate Change Research Summary

        ## Key Information
        Global temperatures have increased by about 1°C since pre-industrial times. Sea levels are rising at an accelerating rate, and extreme weather events are becoming more frequent and intense.

        ## Visual Representation
        An image of a sustainable city with renewable energy has been generated and is available at: http://example.com/sustainable_city.jpg

        ## Quality Assessment
        The information has been evaluated with the following scores:
        - Accuracy: 95%
        - Completeness: 88%
        - Relevance: 97%
        - Overall quality score: 93.3%

        ## Storage Information
        The research has been stored in our vector database for future reference with ID: cc-67890
        """

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

        # Test with a complex climate change research task
        result = supervisor.invoke(
            "Research current climate change data and trends, generate an image of a sustainable city with renewable energy, "
            "evaluate the quality of the climate change information, and store the research in a vector database."
        )

        # Check that all agents were called
        self.assertEqual(self.search_agent.call_count, 1)
        self.assertEqual(self.image_agent.call_count, 1)
        self.assertEqual(self.quality_agent.call_count, 1)
        self.assertEqual(self.vector_agent.call_count, 1)

        # Check that the result contains outputs from all agents
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertIn("image_generation_agent", result["agent_outputs"])
        self.assertIn("quality_agent", result["agent_outputs"])
        self.assertIn("vector_storage_agent", result["agent_outputs"])

        # Check that execution stats are recorded
        self.assertIn("execution_stats", result)
        self.assertIn("parallel_batches", result["execution_stats"])
        self.assertGreaterEqual(result["execution_stats"]["parallel_batches"], 2)  # At least 2 batches for this task

        # Check that the final response is comprehensive
        final_response = result["messages"][-1]["content"]
        self.assertIn("Climate Change Research Summary", final_response)
        self.assertIn("Global temperatures", final_response)
        self.assertIn("http://example.com/sustainable_city.jpg", final_response)
        self.assertIn("Quality Assessment", final_response)
        self.assertIn("vector database", final_response)


if __name__ == "__main__":
    unittest.main()
