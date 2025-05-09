"""
Unit tests for the Search Agent.

This module contains tests for the Search Agent, which is responsible for
searching the web for information.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the Search Agent and related components
from src.agents.search_agent import SearchAgent, SearchAgentConfig


class TestSearchAgentConfig(unittest.TestCase):
    """Tests for the SearchAgentConfig class."""

    def test_default_config(self):
        """Test that default configuration is created correctly."""
        config = SearchAgentConfig()
        
        # Check default values
        self.assertEqual(config.provider, "serper")
        self.assertEqual(config.max_results, 5)
        self.assertEqual(config.include_domains, [])
        self.assertEqual(config.exclude_domains, [])
        self.assertEqual(config.search_type, "search")

    def test_custom_config(self):
        """Test that custom configuration is created correctly."""
        config = SearchAgentConfig(
            provider="google",
            max_results=3,
            include_domains=["example.com"],
            exclude_domains=["exclude.com"],
            search_type="news"
        )
        
        # Check custom values
        self.assertEqual(config.provider, "google")
        self.assertEqual(config.max_results, 3)
        self.assertEqual(config.include_domains, ["example.com"])
        self.assertEqual(config.exclude_domains, ["exclude.com"])
        self.assertEqual(config.search_type, "news")


class TestSearchAgent(unittest.TestCase):
    """Tests for the SearchAgent class."""

    @patch("src.agents.search_agent.ChatOpenAI")
    def test_initialization(self, mock_chat_openai):
        """Test that SearchAgent is initialized correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a SearchAgent with default config
        search_agent = SearchAgent()
        
        # Check that LLM is initialized
        mock_chat_openai.assert_called_once()
        self.assertEqual(search_agent.llm, mock_llm)
        
        # Check that config is initialized
        self.assertEqual(search_agent.config.provider, "serper")
        self.assertEqual(search_agent.config.max_results, 5)

    @patch("src.agents.search_agent.ChatOpenAI")
    @patch("src.agents.search_agent.SerperAPIWrapper")
    def test_initialize_search_tool_serper(self, mock_serper, mock_chat_openai):
        """Test that _initialize_search_tool works correctly with Serper."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a mock Serper wrapper
        mock_serper_instance = MagicMock()
        mock_serper.return_value = mock_serper_instance
        
        # Create a SearchAgent with Serper config
        search_agent = SearchAgent(
            config=SearchAgentConfig(
                provider="serper",
                max_results=3
            )
        )
        
        # Check that Serper wrapper is initialized
        mock_serper.assert_called_once()
        self.assertEqual(search_agent.search_tool, mock_serper_instance)

    @patch("src.agents.search_agent.ChatOpenAI")
    @patch("src.agents.search_agent.GoogleSearchAPIWrapper")
    def test_initialize_search_tool_google(self, mock_google, mock_chat_openai):
        """Test that _initialize_search_tool works correctly with Google."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a mock Google wrapper
        mock_google_instance = MagicMock()
        mock_google.return_value = mock_google_instance
        
        # Create a SearchAgent with Google config
        search_agent = SearchAgent(
            config=SearchAgentConfig(
                provider="google",
                max_results=3
            )
        )
        
        # Check that Google wrapper is initialized
        mock_google.assert_called_once()
        self.assertEqual(search_agent.search_tool, mock_google_instance)

    @patch("src.agents.search_agent.ChatOpenAI")
    @patch("src.agents.search_agent.DuckDuckGoSearchRun")
    def test_initialize_search_tool_duckduckgo(self, mock_ddg, mock_chat_openai):
        """Test that _initialize_search_tool works correctly with DuckDuckGo."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a mock DuckDuckGo wrapper
        mock_ddg_instance = MagicMock()
        mock_ddg.return_value = mock_ddg_instance
        
        # Create a SearchAgent with DuckDuckGo config
        search_agent = SearchAgent(
            config=SearchAgentConfig(
                provider="duckduckgo",
                max_results=3
            )
        )
        
        # Check that DuckDuckGo wrapper is initialized
        mock_ddg.assert_called_once()
        self.assertEqual(search_agent.search_tool, mock_ddg_instance)

    @patch("src.agents.search_agent.ChatOpenAI")
    def test_invoke(self, mock_chat_openai):
        """Test that invoke works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "Paris is the capital of France."
        mock_llm.invoke.return_value = mock_response
        
        # Create a SearchAgent with a mock search tool
        search_agent = SearchAgent()
        search_agent.search_tool = MagicMock()
        search_agent.search_tool.run.return_value = "Paris is the capital of France. It is known for the Eiffel Tower."
        
        # Test with a search query
        state = {
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "agent_outputs": {}
        }
        
        result = search_agent(state)
        
        # Check that search tool is called
        search_agent.search_tool.run.assert_called_once_with("What is the capital of France?")
        
        # Check that LLM is called
        mock_llm.invoke.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result["messages"][-1]["content"], "Paris is the capital of France.")
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertEqual(result["agent_outputs"]["search_agent"]["results"], "Paris is the capital of France. It is known for the Eiffel Tower.")

    @patch("src.agents.search_agent.ChatOpenAI")
    def test_invoke_with_context(self, mock_chat_openai):
        """Test that invoke works correctly with context."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = "Paris is the capital of France and is known for the Eiffel Tower."
        mock_llm.invoke.return_value = mock_response
        
        # Create a SearchAgent with a mock search tool
        search_agent = SearchAgent()
        search_agent.search_tool = MagicMock()
        search_agent.search_tool.run.return_value = "Paris is the capital of France. It is known for the Eiffel Tower."
        
        # Test with a search query and context
        state = {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "I'll search for information about the capital of France."}
            ],
            "agent_outputs": {},
            "context": {"previous_searches": ["France tourism"]}
        }
        
        result = search_agent(state)
        
        # Check that search tool is called
        search_agent.search_tool.run.assert_called_once_with("What is the capital of France?")
        
        # Check that LLM is called
        mock_llm.invoke.assert_called_once()
        
        # Check that the result is correct
        self.assertEqual(result["messages"][-1]["content"], "Paris is the capital of France and is known for the Eiffel Tower.")
        self.assertIn("search_agent", result["agent_outputs"])
        self.assertEqual(result["agent_outputs"]["search_agent"]["results"], "Paris is the capital of France. It is known for the Eiffel Tower.")


if __name__ == "__main__":
    unittest.main()
