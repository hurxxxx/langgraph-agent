"""
Unit tests for the MCP (Master Control Program) Agent.

This module contains tests for the MCP agent, which is responsible for
breaking down complex tasks into subtasks and delegating them to specialized agents.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the MCP agent and related components
from src.agents.mcp_agent import MCPAgent, MCPAgentConfig


class TestMCPAgentConfig(unittest.TestCase):
    """Tests for the MCPAgentConfig class."""

    def test_default_config(self):
        """Test that default configuration is created correctly."""
        config = MCPAgentConfig()
        
        # Check default values
        self.assertEqual(config.llm_provider, "openai")
        self.assertEqual(config.openai_model, "gpt-4o")
        self.assertEqual(config.anthropic_model, "claude-3-opus-20240229")
        self.assertEqual(config.temperature, 0)
        self.assertEqual(config.streaming, True)
        self.assertEqual(config.max_subtasks, 5)
        
        # Check that system message and planning template are initialized
        self.assertIsNotNone(config.system_message)
        self.assertIsNotNone(config.planning_template)

    def test_custom_config(self):
        """Test that custom configuration is created correctly."""
        config = MCPAgentConfig(
            llm_provider="anthropic",
            openai_model="gpt-4-turbo",
            anthropic_model="claude-3-sonnet",
            temperature=0.5,
            streaming=False,
            max_subtasks=3
        )
        
        # Check custom values
        self.assertEqual(config.llm_provider, "anthropic")
        self.assertEqual(config.openai_model, "gpt-4-turbo")
        self.assertEqual(config.anthropic_model, "claude-3-sonnet")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.streaming, False)
        self.assertEqual(config.max_subtasks, 3)


class TestMCPAgent(unittest.TestCase):
    """Tests for the MCPAgent class."""

    @patch("src.agents.mcp_agent.ChatOpenAI")
    def test_initialization(self, mock_chat_openai):
        """Test that MCPAgent is initialized correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create an MCP agent with default config
        mcp_agent = MCPAgent()
        
        # Check that LLM is initialized
        mock_chat_openai.assert_called_once()
        self.assertEqual(mcp_agent.llm, mock_llm)
        
        # Check that agents dictionary is initialized
        self.assertEqual(mcp_agent.agents, {})

    @patch("src.agents.mcp_agent.ChatOpenAI")
    def test_create_execution_plan(self, mock_chat_openai):
        """Test that _create_execution_plan works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = """
        Subtasks:
        1. Research quantum computing basics - Agent: search_agent
        2. Generate an image of quantum entanglement - Agent: image_generation_agent
        3. Evaluate the quality of the information - Agent: quality_agent
        
        Execution Plan:
        1. Execute subtask 1
        2. Execute subtask 2
        3. Execute subtask 3
        """
        mock_llm.invoke.return_value = mock_response
        
        # Create an MCP agent with agents
        mcp_agent = MCPAgent()
        mcp_agent.agents = {
            "search_agent": MagicMock(),
            "image_generation_agent": MagicMock(),
            "quality_agent": MagicMock()
        }
        
        # Test with a complex task
        plan = mcp_agent._create_execution_plan("Research quantum computing and create an image.")
        
        # Check that LLM is called
        mock_llm.invoke.assert_called_once()
        
        # Check that the plan is created correctly
        self.assertEqual(plan["task"], "Research quantum computing and create an image.")
        self.assertEqual(plan["raw_plan"], mock_response.content)

    @patch("src.agents.mcp_agent.ChatOpenAI")
    def test_parse_execution_plan(self, mock_chat_openai):
        """Test that _parse_execution_plan works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create an MCP agent with agents
        mcp_agent = MCPAgent()
        mcp_agent.agents = {
            "search_agent": MagicMock(),
            "image_generation_agent": MagicMock(),
            "quality_agent": MagicMock()
        }
        
        # Test with a raw plan
        raw_plan = """
        Subtasks:
        1. Research quantum computing basics - Agent: search_agent
        2. Generate an image of quantum entanglement - Agent: image_generation_agent
        3. Evaluate the quality of the information - Agent: quality_agent
        
        Execution Plan:
        1. Execute subtask 1
        2. Execute subtask 2
        3. Execute subtask 3
        """
        
        parsed_plan = mcp_agent._parse_execution_plan(raw_plan)
        
        # Check that the plan is parsed correctly
        self.assertIn("subtasks", parsed_plan)
        self.assertIn("execution_order", parsed_plan)
        self.assertEqual(len(parsed_plan["subtasks"]), 3)
        self.assertEqual(len(parsed_plan["execution_order"]), 3)
        
        # Check that the subtasks are parsed correctly
        self.assertEqual(parsed_plan["subtasks"][0]["agent"], "search_agent")
        self.assertEqual(parsed_plan["subtasks"][1]["agent"], "image_generation_agent")
        self.assertEqual(parsed_plan["subtasks"][2]["agent"], "quality_agent")

    @patch("src.agents.mcp_agent.ChatOpenAI")
    def test_execute_subtask(self, mock_chat_openai):
        """Test that _execute_subtask works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a mock agent
        mock_agent = MagicMock()
        mock_agent.return_value = {
            "messages": [
                {"role": "user", "content": "Research quantum computing."},
                {"role": "assistant", "content": "Quantum computing uses quantum bits or qubits."}
            ],
            "agent_outputs": {
                "search_agent": {"results": ["Quantum computing uses quantum bits or qubits."]}
            }
        }
        
        # Create an MCP agent with agents
        mcp_agent = MCPAgent()
        mcp_agent.agents = {
            "search_agent": mock_agent
        }
        
        # Test with a subtask
        subtask = {
            "description": "Research quantum computing basics",
            "agent": "search_agent",
            "dependencies": [],
            "expected_output": "Information about quantum computing"
        }
        
        state = {
            "messages": [{"role": "user", "content": "Research quantum computing."}],
            "agent_outputs": {}
        }
        
        updated_state = mcp_agent._execute_subtask(subtask, state)
        
        # Check that the agent is called
        mock_agent.assert_called_once()
        
        # Check that the state is updated correctly
        self.assertEqual(updated_state["current_subtask"], "Research quantum computing basics")
        self.assertEqual(updated_state["expected_output"], "Information about quantum computing")
        self.assertIn("subtask_results", updated_state)
        self.assertIn("Research quantum computing basics", updated_state["subtask_results"])
        self.assertEqual(updated_state["subtask_results"]["Research quantum computing basics"]["agent"], "search_agent")

    @patch("src.agents.mcp_agent.ChatOpenAI")
    def test_invoke(self, mock_chat_openai):
        """Test that invoke works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM responses
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
        
        mock_llm.invoke.side_effect = [mock_plan_response, mock_final_response]
        
        # Create mock agents
        mock_search_agent = MagicMock()
        mock_search_agent.return_value = {
            "messages": [
                {"role": "user", "content": "Research quantum computing."},
                {"role": "assistant", "content": "Quantum computing uses quantum bits or qubits."}
            ],
            "agent_outputs": {
                "search_agent": {"results": ["Quantum computing uses quantum bits or qubits."]}
            },
            "current_subtask": "Research quantum computing basics",
            "expected_output": "Information about quantum computing"
        }
        
        mock_image_agent = MagicMock()
        mock_image_agent.return_value = {
            "messages": [
                {"role": "user", "content": "Generate an image of quantum entanglement."},
                {"role": "assistant", "content": "Here's an image of quantum entanglement."}
            ],
            "agent_outputs": {
                "image_generation_agent": {"image_url": "http://example.com/image.jpg"}
            },
            "current_subtask": "Generate an image of quantum entanglement",
            "expected_output": "Image of quantum entanglement",
            "subtask_results": {
                "Research quantum computing basics": {
                    "agent": "search_agent",
                    "result": {"results": ["Quantum computing uses quantum bits or qubits."]}
                }
            }
        }
        
        # Create an MCP agent with agents
        mcp_agent = MCPAgent()
        mcp_agent.agents = {
            "search_agent": mock_search_agent,
            "image_generation_agent": mock_image_agent
        }
        
        # Patch the _parse_execution_plan method to return a structured plan
        with patch.object(mcp_agent, '_parse_execution_plan') as mock_parse:
            mock_parse.return_value = {
                "subtasks": [
                    {
                        "description": "Research quantum computing basics",
                        "agent": "search_agent",
                        "dependencies": [],
                        "expected_output": "Information about quantum computing"
                    },
                    {
                        "description": "Generate an image of quantum entanglement",
                        "agent": "image_generation_agent",
                        "dependencies": [],
                        "expected_output": "Image of quantum entanglement"
                    }
                ],
                "execution_order": [
                    "Execute subtask 1",
                    "Execute subtask 2"
                ]
            }
            
            # Test with a complex task
            state = {
                "messages": [{"role": "user", "content": "Research quantum computing and create an image."}],
                "agent_outputs": {}
            }
            
            result = mcp_agent.invoke(state)
            
            # Check that the agents are called
            mock_search_agent.assert_called_once()
            mock_image_agent.assert_called_once()
            
            # Check that the result is correct
            self.assertEqual(result["messages"][-1]["content"], "Here's information about quantum computing and an image of quantum entanglement.")


if __name__ == "__main__":
    unittest.main()
