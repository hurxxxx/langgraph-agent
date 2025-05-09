"""
Unit tests for specialized MCP agents.

This module contains tests for the specialized MCP agents:
1. CrewAI-style MCP: Role-based agent teams
2. AutoGen-style MCP: Conversational multi-agent systems
3. LangGraph-style MCP: Graph-based workflows
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the specialized MCP agents and related components
from src.agents.crew_mcp_agent import CrewMCPAgent, CrewMCPAgentConfig
from src.agents.autogen_mcp_agent import AutoGenMCPAgent, AutoGenMCPAgentConfig
from src.agents.langgraph_mcp_agent import LangGraphMCPAgent, LangGraphMCPAgentConfig


class TestCrewMCPAgent(unittest.TestCase):
    """Tests for the CrewAI-style MCP agent."""

    @patch("src.agents.crew_mcp_agent.ChatOpenAI")
    def test_initialization(self, mock_chat_openai):
        """Test that CrewMCPAgent is initialized correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a CrewMCPAgent with default config
        crew_mcp_agent = CrewMCPAgent()
        
        # Check that LLM is initialized
        mock_chat_openai.assert_called_once()
        self.assertEqual(crew_mcp_agent.llm, mock_llm)
        
        # Check that agents dictionary is initialized
        self.assertEqual(crew_mcp_agent.agents, {})

    @patch("src.agents.crew_mcp_agent.ChatOpenAI")
    def test_create_crew_plan(self, mock_chat_openai):
        """Test that _create_crew_plan works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.content = """
        Roles:
        1. Role: Researcher
           Agent: search_agent
           Description: Research quantum computing
           Goals: Find information about quantum computing
           
        2. Role: Illustrator
           Agent: image_generation_agent
           Description: Create visual representations
           Goals: Generate an image of quantum entanglement
           
        Workflow:
        1. Researcher finds information
        2. Illustrator creates visuals based on research
        """
        mock_llm.invoke.return_value = mock_response
        
        # Create a CrewMCPAgent with agents
        crew_mcp_agent = CrewMCPAgent()
        crew_mcp_agent.agents = {
            "search_agent": MagicMock(),
            "image_generation_agent": MagicMock()
        }
        
        # Test with a complex task
        plan = crew_mcp_agent._create_crew_plan("Research quantum computing and create an image.")
        
        # Check that LLM is called
        mock_llm.invoke.assert_called_once()
        
        # Check that the plan is created correctly
        self.assertEqual(plan["task"], "Research quantum computing and create an image.")
        self.assertEqual(plan["raw_plan"], mock_response.content)

    @patch("src.agents.crew_mcp_agent.ChatOpenAI")
    def test_invoke(self, mock_chat_openai):
        """Test that invoke works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM responses
        mock_plan_response = MagicMock()
        mock_plan_response.content = """
        Roles:
        1. Role: Researcher
           Agent: search_agent
           Description: Research quantum computing
           Goals: Find information about quantum computing
           
        2. Role: Illustrator
           Agent: image_generation_agent
           Description: Create visual representations
           Goals: Generate an image of quantum entanglement
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
            }
        }
        
        mock_image_agent = MagicMock()
        mock_image_agent.return_value = {
            "messages": [
                {"role": "user", "content": "Generate an image of quantum entanglement."},
                {"role": "assistant", "content": "Here's an image of quantum entanglement."}
            ],
            "agent_outputs": {
                "image_generation_agent": {"image_url": "http://example.com/image.jpg"}
            }
        }
        
        # Create a CrewMCPAgent with agents
        crew_mcp_agent = CrewMCPAgent()
        crew_mcp_agent.agents = {
            "search_agent": mock_search_agent,
            "image_generation_agent": mock_image_agent
        }
        
        # Patch the _parse_crew_plan method to return a structured plan
        with patch.object(crew_mcp_agent, '_parse_crew_plan') as mock_parse:
            mock_parse.return_value = {
                "roles": [
                    {
                        "name": "Researcher",
                        "agent": "search_agent",
                        "description": "Research quantum computing",
                        "goals": ["Find information about quantum computing"]
                    },
                    {
                        "name": "Illustrator",
                        "agent": "image_generation_agent",
                        "description": "Create visual representations",
                        "goals": ["Generate an image of quantum entanglement"]
                    }
                ],
                "workflow": [
                    "Researcher finds information",
                    "Illustrator creates visuals based on research"
                ]
            }
            
            # Test with a complex task
            state = {
                "messages": [{"role": "user", "content": "Research quantum computing and create an image."}],
                "agent_outputs": {}
            }
            
            result = crew_mcp_agent.invoke(state)
            
            # Check that the agents are called
            mock_search_agent.assert_called_once()
            mock_image_agent.assert_called_once()
            
            # Check that the result is correct
            self.assertEqual(result["messages"][-1]["content"], "Here's information about quantum computing and an image of quantum entanglement.")


class TestAutoGenMCPAgent(unittest.TestCase):
    """Tests for the AutoGen-style MCP agent."""

    @patch("src.agents.autogen_mcp_agent.ChatOpenAI")
    def test_initialization(self, mock_chat_openai):
        """Test that AutoGenMCPAgent is initialized correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create an AutoGenMCPAgent with default config
        autogen_mcp_agent = AutoGenMCPAgent()
        
        # Check that LLM is initialized
        mock_chat_openai.assert_called_once()
        self.assertEqual(autogen_mcp_agent.llm, mock_llm)
        
        # Check that agents dictionary is initialized
        self.assertEqual(autogen_mcp_agent.agents, {})

    @patch("src.agents.autogen_mcp_agent.ChatOpenAI")
    def test_invoke(self, mock_chat_openai):
        """Test that invoke works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM responses
        mock_plan_response = MagicMock()
        mock_plan_response.content = """
        Agent Configurations:
        1. Agent: Researcher
           Base Agent: search_agent
           System Message: You are a researcher who finds information.
           Can Call: Illustrator
           
        2. Agent: Illustrator
           Base Agent: image_generation_agent
           System Message: You are an illustrator who creates images.
           Can Call: Researcher
        """
        
        mock_agent_responses = [
            MagicMock(content="I found information about quantum computing."),
            MagicMock(content="I need help from Illustrator: Can you create an image of quantum entanglement?"),
            MagicMock(content="Here's an image of quantum entanglement."),
            MagicMock(content="TASK COMPLETE: Here's information about quantum computing and an image.")
        ]
        
        mock_final_response = MagicMock()
        mock_final_response.content = "Here's information about quantum computing and an image of quantum entanglement."
        
        mock_llm.invoke.side_effect = [mock_plan_response] + mock_agent_responses + [mock_final_response]
        
        # Create an AutoGenMCPAgent with agents
        autogen_mcp_agent = AutoGenMCPAgent()
        autogen_mcp_agent.agents = {
            "search_agent": MagicMock(),
            "image_generation_agent": MagicMock()
        }
        
        # Patch the _parse_conversation_plan method to return a structured plan
        with patch.object(autogen_mcp_agent, '_parse_conversation_plan') as mock_parse:
            mock_parse.return_value = {
                "agent_configs": [
                    {
                        "name": "Researcher",
                        "agent_name": "search_agent",
                        "system_message": "You are a researcher who finds information.",
                        "can_call_agents": ["Illustrator"]
                    },
                    {
                        "name": "Illustrator",
                        "agent_name": "image_generation_agent",
                        "system_message": "You are an illustrator who creates images.",
                        "can_call_agents": ["Researcher"]
                    }
                ],
                "flow": {
                    "start": "Researcher",
                    "completion": "When all information is gathered and visualized."
                }
            }
            
            # Patch the _simulate_conversation method to return a simulated conversation
            with patch.object(autogen_mcp_agent, '_simulate_conversation') as mock_simulate:
                mock_simulate.return_value = {
                    "messages": [{"role": "user", "content": "Research quantum computing and create an image."}],
                    "agent_outputs": {},
                    "conversation": [
                        {"role": "user", "content": "Research quantum computing and create an image."},
                        {"role": "assistant", "name": "Researcher", "content": "I found information about quantum computing."},
                        {"role": "assistant", "name": "Researcher", "content": "I need help from Illustrator: Can you create an image of quantum entanglement?"},
                        {"role": "assistant", "name": "Illustrator", "content": "Here's an image of quantum entanglement."},
                        {"role": "assistant", "name": "Researcher", "content": "TASK COMPLETE: Here's information about quantum computing and an image."}
                    ]
                }
                
                # Test with a complex task
                state = {
                    "messages": [{"role": "user", "content": "Research quantum computing and create an image."}],
                    "agent_outputs": {}
                }
                
                result = autogen_mcp_agent.invoke(state)
                
                # Check that the conversation is simulated
                mock_simulate.assert_called_once()
                
                # Check that the result is correct
                self.assertEqual(result["messages"][-1]["content"], "Here's information about quantum computing and an image of quantum entanglement.")


class TestLangGraphMCPAgent(unittest.TestCase):
    """Tests for the LangGraph-style MCP agent."""

    @patch("src.agents.langgraph_mcp_agent.ChatOpenAI")
    def test_initialization(self, mock_chat_openai):
        """Test that LangGraphMCPAgent is initialized correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Create a LangGraphMCPAgent with default config
        langgraph_mcp_agent = LangGraphMCPAgent()
        
        # Check that LLM is initialized
        mock_chat_openai.assert_called_once()
        self.assertEqual(langgraph_mcp_agent.llm, mock_llm)
        
        # Check that agents dictionary is initialized
        self.assertEqual(langgraph_mcp_agent.agents, {})

    @patch("src.agents.langgraph_mcp_agent.ChatOpenAI")
    def test_invoke(self, mock_chat_openai):
        """Test that invoke works correctly."""
        # Create a mock LLM
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock the LLM responses
        mock_plan_response = MagicMock()
        mock_plan_response.content = """
        Nodes:
        1. Node: Researcher
           Agent: search_agent
           Description: Research quantum computing
           
        2. Node: Illustrator
           Agent: image_generation_agent
           Description: Create visual representations
           
        Edges:
        Researcher -> Illustrator
        
        Execution Plan:
        Entry Point: Researcher
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
            }
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
            "graph_execution": [
                {
                    "node": "Researcher",
                    "agent": "search_agent",
                    "result": {"results": ["Quantum computing uses quantum bits or qubits."]}
                }
            ]
        }
        
        # Create a LangGraphMCPAgent with agents
        langgraph_mcp_agent = LangGraphMCPAgent()
        langgraph_mcp_agent.agents = {
            "search_agent": mock_search_agent,
            "image_generation_agent": mock_image_agent
        }
        
        # Patch the _parse_graph_plan method to return a structured plan
        with patch.object(langgraph_mcp_agent, '_parse_graph_plan') as mock_parse:
            mock_parse.return_value = {
                "nodes": [
                    {
                        "name": "Researcher",
                        "agent_name": "search_agent",
                        "description": "Research quantum computing"
                    },
                    {
                        "name": "Illustrator",
                        "agent_name": "image_generation_agent",
                        "description": "Create visual representations"
                    }
                ],
                "edges": [
                    {
                        "source": "Researcher",
                        "target": "Illustrator",
                        "condition": ""
                    }
                ],
                "execution_plan": {
                    "entry_point": "Researcher"
                }
            }
            
            # Patch the _execute_graph method to return a simulated execution
            with patch.object(langgraph_mcp_agent, '_execute_graph') as mock_execute:
                mock_execute.return_value = {
                    "messages": [{"role": "user", "content": "Research quantum computing and create an image."}],
                    "agent_outputs": {},
                    "graph_execution": [
                        {
                            "node": "Researcher",
                            "agent": "search_agent",
                            "result": {"results": ["Quantum computing uses quantum bits or qubits."]}
                        },
                        {
                            "node": "Illustrator",
                            "agent": "image_generation_agent",
                            "result": {"image_url": "http://example.com/image.jpg"}
                        }
                    ]
                }
                
                # Test with a complex task
                state = {
                    "messages": [{"role": "user", "content": "Research quantum computing and create an image."}],
                    "agent_outputs": {}
                }
                
                result = langgraph_mcp_agent.invoke(state)
                
                # Check that the graph is executed
                mock_execute.assert_called_once()
                
                # Check that the result is correct
                self.assertEqual(result["messages"][-1]["content"], "Here's information about quantum computing and an image of quantum entanglement.")


if __name__ == "__main__":
    unittest.main()
