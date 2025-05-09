"""
Reactivity tests for the Multi-Agent Supervisor System.

This module contains tests that verify the supervisor's ability to:
1. React to changing requirements during task execution
2. Handle feedback and adjust plans accordingly
3. Recover from errors and adapt the execution plan
4. Manage complex dependencies between tasks
5. Optimize execution based on intermediate results

These tests focus on the reactive nature of the supervisor and its ability to adapt to changing conditions.
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
from src.agents.mcp_agent import MCPAgent, MCPAgentConfig


class ReactiveAgent:
    """
    Reactive agent that can simulate changing conditions, errors, and feedback.

    This agent can:
    1. Return different responses based on the execution context
    2. Simulate errors under certain conditions
    3. Request human feedback when needed
    4. Adapt its behavior based on the state
    """

    def __init__(self, name, behaviors=None):
        """
        Initialize the reactive agent.

        Args:
            name: Agent name
            behaviors: Dictionary mapping conditions to behaviors
        """
        self.name = name
        self.behaviors = behaviors or {}
        self.call_count = 0
        self.call_history = []
        self.default_response = f"Default response from {name}"

    def __call__(self, state):
        """
        Call the agent with reactive behavior.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        # Record the call
        self.call_count += 1
        self.call_history.append(state.copy())

        # Initialize agent outputs if needed
        if "agent_outputs" not in state:
            state["agent_outputs"] = {}

        # Check for error condition
        if "simulate_error" in state and state["simulate_error"] == self.name:
            state["agent_outputs"][self.name] = {"error": f"Simulated error in {self.name}"}
            raise ValueError(f"Simulated error in {self.name}")

        # Check for feedback request condition
        if "request_feedback" in state and state["request_feedback"] == self.name:
            state["agent_outputs"][self.name] = {"request_human_feedback": True, "message": f"{self.name} needs human feedback"}
            return state

        # Determine the appropriate behavior based on context
        response = self.default_response
        for condition, behavior in self.behaviors.items():
            if self._check_condition(condition, state):
                response = behavior["response"]

                # Apply any state modifications
                if "state_updates" in behavior:
                    for key, value in behavior["state_updates"].items():
                        state[key] = value

                break

        # Add agent output to state
        state["agent_outputs"][self.name] = {"result": response}

        # Add agent message to state
        if "messages" not in state:
            state["messages"] = []

        state["messages"].append({"role": "assistant", "content": response})

        return state

    def _check_condition(self, condition, state):
        """
        Check if a condition is met in the current state.

        Args:
            condition: Condition to check
            state: Current state

        Returns:
            bool: Whether the condition is met
        """
        if condition == "always":
            return True
        elif condition == "first_call" and self.call_count == 1:
            return True
        elif condition == "second_call" and self.call_count == 2:
            return True
        elif condition.startswith("subtask_") and "current_subtask" in state:
            subtask_desc = condition.split("subtask_")[1]
            return subtask_desc in state["current_subtask"]["description"]
        elif condition in state:
            return state[condition]

        return False


class TestSupervisorReactivity(unittest.TestCase):
    """Tests for the reactive capabilities of the Supervisor."""

    def setUp(self):
        """Set up the test environment."""
        # Load environment variables
        load_dotenv()

        # Create reactive agents
        self.search_agent = ReactiveAgent(
            "search_agent",
            {
                "subtask_climate": {
                    "response": "Climate change data: Global temperatures have increased by about 1°C since pre-industrial times."
                },
                "subtask_quantum": {
                    "response": "Quantum computing uses quantum bits or qubits, which can exist in multiple states simultaneously."
                },
                "subtask_error": {
                    "response": "Error: Could not complete search due to API rate limiting.",
                    "state_updates": {"search_error": True}
                },
                "search_error": {
                    "response": "Retried search with backup API: Quantum computing is based on quantum mechanics principles."
                }
            }
        )

        self.image_agent = ReactiveAgent(
            "image_generation_agent",
            {
                "subtask_sustainable": {
                    "response": "Generated image of sustainable city: http://example.com/sustainable_city.jpg"
                },
                "subtask_quantum": {
                    "response": "Generated image of quantum computer: http://example.com/quantum_computer.jpg"
                },
                "search_error": {
                    "response": "Generated alternative image based on limited information: http://example.com/quantum_concept.jpg"
                }
            }
        )

        self.quality_agent = ReactiveAgent(
            "quality_agent",
            {
                "search_error": {
                    "response": "Quality assessment: Limited information available. Accuracy: 70%, Completeness: 45%",
                    "state_updates": {"quality_warning": True}
                },
                "always": {
                    "response": "Quality assessment: Information is accurate and comprehensive. Accuracy: 95%, Completeness: 90%"
                }
            }
        )

        # Create agent dictionary
        self.agents = {
            "search_agent": self.search_agent,
            "image_generation_agent": self.image_agent,
            "quality_agent": self.quality_agent
        }

    @patch("src.supervisor.supervisor.ChatOpenAI")
    @patch("src.agents.mcp_agent.ChatOpenAI")
    def test_error_recovery(self, mock_mcp_llm, mock_supervisor_llm):
        """Test that supervisor can recover from errors and adapt the execution plan."""
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
           Description: subtask_quantum Find information about quantum computing principles
           Dependencies: None
           Expected output: Comprehensive information about quantum computing

        2. Generate an image of quantum computer - Agent: image_generation_agent
           Description: subtask_quantum Create a visual representation of a quantum computer
           Dependencies: [1]
           Expected output: URL to generated image

        3. Evaluate the quality of the research information - Agent: quality_agent
           Description: Assess the accuracy and completeness of the quantum computing information
           Dependencies: [1]
           Expected output: Quality assessment with scores

        Execution Plan:
        1. Execute subtask 1 (no dependencies)
        2. After subtask 1 completes, execute subtasks 2 and 3 in parallel (both depend on subtask 1)
        3. Synthesize final response incorporating all results
        """

        # First synthesis response (with error)
        mock_synthesis_response1 = MagicMock()
        mock_synthesis_response1.content = """
        # Quantum Computing Research with Error Recovery

        ## Initial Search Error
        The search for quantum computing information encountered an error due to API rate limiting.

        ## Recovery Action
        A backup API was used to retrieve information: Quantum computing is based on quantum mechanics principles.

        ## Visual Representation
        An alternative image was generated based on limited information: http://example.com/quantum_concept.jpg

        ## Quality Assessment
        Quality assessment: Limited information available. Accuracy: 70%, Completeness: 45%

        Note: Due to the search API limitations, the information provided is not as comprehensive as intended.
        """

        mock_m_llm.invoke.side_effect = [mock_plan_response, mock_synthesis_response1]

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

        # Create a state with an error simulation for the search agent
        state = {
            "messages": [{"role": "user", "content": "Research quantum computing and generate an image of a quantum computer."}],
            "agent_outputs": {},
            "subtask_error": True  # This will trigger the error behavior in the search agent
        }

        # Test with error recovery
        result = supervisor.mcp_agent.invoke(state)

        # Check that all agents were called
        self.assertEqual(self.search_agent.call_count, 1)
        self.assertEqual(self.image_agent.call_count, 1)
        self.assertEqual(self.quality_agent.call_count, 1)

        # Check that the search error was detected and handled
        self.assertTrue("search_error" in result)
        self.assertTrue(result["search_error"])

        # Check that the quality warning was set
        self.assertTrue("quality_warning" in result)
        self.assertTrue(result["quality_warning"])

        # Check that the final response acknowledges the error and recovery
        final_response = result["messages"][-1]["content"]
        self.assertIn("Error Recovery", final_response)
        self.assertIn("Initial Search Error", final_response)
        self.assertIn("Recovery Action", final_response)
        self.assertIn("backup API", final_response)
        self.assertIn("alternative image", final_response)
        self.assertIn("Limited information available", final_response)

    @patch("src.supervisor.supervisor.ChatOpenAI")
    @patch("src.agents.mcp_agent.ChatOpenAI")
    def test_adaptive_execution(self, mock_mcp_llm, mock_supervisor_llm):
        """Test that supervisor can adapt execution based on intermediate results."""
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
        1. Research climate change data - Agent: search_agent
           Description: subtask_climate Find information about climate change trends
           Dependencies: None
           Expected output: Comprehensive information about climate change

        2. Generate an image of sustainable city - Agent: image_generation_agent
           Description: subtask_sustainable Create a visual representation of a sustainable city
           Dependencies: [1]
           Expected output: URL to generated image

        3. Evaluate the quality of the research information - Agent: quality_agent
           Description: Assess the accuracy and completeness of the climate change information
           Dependencies: [1]
           Expected output: Quality assessment with scores

        Execution Plan:
        1. Execute subtask 1 (no dependencies)
        2. After subtask 1 completes, execute subtasks 2 and 3 in parallel (both depend on subtask 1)
        3. Synthesize final response incorporating all results
        """

        mock_synthesis_response = MagicMock()
        mock_synthesis_response.content = """
        # Climate Change Research Summary

        ## Key Information
        Climate change data: Global temperatures have increased by about 1°C since pre-industrial times.

        ## Visual Representation
        Generated image of sustainable city: http://example.com/sustainable_city.jpg

        ## Quality Assessment
        Quality assessment: Information is accurate and comprehensive. Accuracy: 95%, Completeness: 90%
        """

        mock_m_llm.invoke.side_effect = [mock_plan_response, mock_synthesis_response]

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

        # Test with adaptive execution
        result = supervisor.invoke("Research climate change data and generate an image of a sustainable city.")

        # Check that all agents were called
        self.assertEqual(self.search_agent.call_count, 1)
        self.assertEqual(self.image_agent.call_count, 1)
        self.assertEqual(self.quality_agent.call_count, 1)

        # Check that the final response includes all components
        final_response = result["messages"][-1]["content"]
        self.assertIn("Climate Change Research Summary", final_response)
        self.assertIn("Global temperatures", final_response)
        self.assertIn("http://example.com/sustainable_city.jpg", final_response)
        self.assertIn("Quality assessment", final_response)


if __name__ == "__main__":
    unittest.main()
