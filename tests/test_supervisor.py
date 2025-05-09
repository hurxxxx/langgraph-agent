"""
Unit tests for the Supervisor class.

This module contains tests for the Supervisor class, which is responsible for
orchestrating multiple specialized agents.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the Supervisor class and related components
from src.supervisor.supervisor import Supervisor, SupervisorConfig


class TestSupervisorConfig(unittest.TestCase):
    """Tests for the SupervisorConfig class."""

    def test_default_config(self):
        """Test that default configuration is created correctly."""
        config = SupervisorConfig()
        
        # Check default values
        self.assertEqual(config.llm_provider, "openai")
        self.assertEqual(config.openai_model, "gpt-4o")
        self.assertEqual(config.anthropic_model, "claude-3-opus-20240229")
        self.assertEqual(config.temperature, 0)
        self.assertEqual(config.streaming, True)
        self.assertEqual(config.mcp_mode, "standard")
        self.assertEqual(config.complexity_threshold, 0.7)
        
        # Check that MCP configs are initialized
        self.assertIsNotNone(config.mcp_config)
        self.assertIsNotNone(config.crew_mcp_config)
        self.assertIsNotNone(config.autogen_mcp_config)
        self.assertIsNotNone(config.langgraph_mcp_config)
        
        # Check backward compatibility
        self.assertFalse(config.use_mcp)

    def test_custom_config(self):
        """Test that custom configuration is created correctly."""
        config = SupervisorConfig(
            llm_provider="anthropic",
            openai_model="gpt-4-turbo",
            anthropic_model="claude-3-sonnet",
            temperature=0.5,
            streaming=False,
            mcp_mode="crew",
            complexity_threshold=0.8
        )
        
        # Check custom values
        self.assertEqual(config.llm_provider, "anthropic")
        self.assertEqual(config.openai_model, "gpt-4-turbo")
        self.assertEqual(config.anthropic_model, "claude-3-sonnet")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.streaming, False)
        self.assertEqual(config.mcp_mode, "crew")
        self.assertEqual(config.complexity_threshold, 0.8)
        
        # Check backward compatibility
        self.assertTrue(config.use_mcp)


class TestSupervisor(unittest.TestCase):
    """Tests for the Supervisor class."""

    @patch("src.supervisor.supervisor.ChatOpenAI")
    def test_initialization(self, mock_chat_openai):
        """Test that Supervisor is initialized correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a supervisor with default config
        supervisor = Supervisor()
        
        # Check that LLM is initialized
        mock_chat_openai.assert_called_once()
        self.assertEqual(supervisor.llm, mock_llm)
        
        # Check that agents dictionary is initialized
        self.assertEqual(supervisor.agents, {})
        
        # Check that MCP agents are not initialized
        self.assertIsNone(supervisor.mcp_agent)
        self.assertIsNone(supervisor.crew_mcp_agent)
        self.assertIsNone(supervisor.autogen_mcp_agent)
        self.assertIsNone(supervisor.langgraph_mcp_agent)

    @patch("src.supervisor.supervisor.ChatOpenAI")
    @patch("src.supervisor.supervisor.MCPAgent")
    def test_initialization_with_mcp(self, mock_mcp_agent, mock_chat_openai):
        """Test that Supervisor is initialized correctly with MCP."""
        # Create a mock LLM and MCP agent
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        mock_mcp = MagicMock()
        mock_mcp_agent.return_value = mock_mcp
        
        # Create a supervisor with MCP config
        supervisor = Supervisor(
            config=SupervisorConfig(mcp_mode="mcp")
        )
        
        # Check that LLM is initialized
        mock_chat_openai.assert_called_once()
        self.assertEqual(supervisor.llm, mock_llm)
        
        # Check that MCP agent is initialized
        mock_mcp_agent.assert_called_once()
        self.assertEqual(supervisor.mcp_agent, mock_mcp)

    @patch("src.supervisor.supervisor.ChatOpenAI")
    def test_determine_next_agent(self, mock_chat_openai):
        """Test that _determine_next_agent works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a supervisor with agents
        supervisor = Supervisor()
        supervisor.agents = {
            "search_agent": MagicMock(),
            "image_generation_agent": MagicMock()
        }
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "I'll use the search_agent to find information about this."
        mock_llm.invoke.return_value = mock_response
        
        # Test with a search query
        agent_name = supervisor._determine_next_agent("What is the capital of France?")
        
        # Check that LLM is called
        mock_llm.invoke.assert_called_once()
        
        # Check that the correct agent is returned
        self.assertEqual(agent_name, "search_agent")

    @patch("src.supervisor.supervisor.ChatOpenAI")
    def test_invoke_standard_mode(self, mock_chat_openai):
        """Test that invoke works correctly in standard mode."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a mock agent
        mock_agent = MagicMock()
        mock_agent.return_value = {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ],
            "agent_outputs": {
                "search_agent": {"results": ["Paris is the capital of France."]}
            }
        }
        
        # Create a supervisor with agents
        supervisor = Supervisor()
        supervisor.agents = {
            "search_agent": mock_agent
        }
        
        # Mock the LLM responses
        mock_response1 = MagicMock()
        mock_response1.content = "I'll use the search_agent to find information about this."
        
        mock_response2 = MagicMock()
        mock_response2.content = "The capital of France is Paris."
        
        mock_llm.invoke.side_effect = [mock_response1, mock_response2]
        
        # Test with a search query
        result = supervisor.invoke("What is the capital of France?")
        
        # Check that the agent is called
        mock_agent.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result["messages"][-1]["content"], "The capital of France is Paris.")
        self.assertIn("search_agent", result["agent_outputs"])

    @patch("src.supervisor.supervisor.ChatOpenAI")
    @patch("src.supervisor.supervisor.MCPAgent")
    def test_invoke_mcp_mode(self, mock_mcp_agent, mock_chat_openai):
        """Test that invoke works correctly in MCP mode."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a mock MCP agent
        mock_mcp = MagicMock()
        mock_mcp.invoke.return_value = {
            "messages": [
                {"role": "user", "content": "Research quantum computing and create an image."},
                {"role": "assistant", "content": "Here's information about quantum computing and an image."}
            ],
            "agent_outputs": {
                "search_agent": {"results": ["Quantum computing information."]},
                "image_generation_agent": {"image_url": "http://example.com/image.jpg"}
            }
        }
        mock_mcp_agent.return_value = mock_mcp
        
        # Create a supervisor with MCP config
        supervisor = Supervisor(
            config=SupervisorConfig(mcp_mode="mcp")
        )
        supervisor.mcp_agent = mock_mcp
        
        # Test with a complex query
        result = supervisor.invoke("Research quantum computing and create an image.")
        
        # Check that MCP agent is called
        mock_mcp.invoke.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result["messages"][-1]["content"], "Here's information about quantum computing and an image.")
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertIn("image_generation_agent", result["agent_outputs"])


if __name__ == "__main__":
    unittest.main()
