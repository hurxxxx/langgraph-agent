"""
Supervisor Agent for Multi-Agent System

This module implements a supervisor agent that orchestrates multiple specialized agents.
The supervisor is responsible for understanding user queries, delegating tasks to appropriate
specialized agents, coordinating communication between agents, and synthesizing final responses.

The supervisor can operate in two modes:
1. Standard mode: Simple delegation to a single agent
2. MCP mode: Complex task breakdown and delegation to multiple agents using the Master Control Program
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union

# Import LangSmith utilities
from utils.langsmith_utils import tracer

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Import MCP agent
from agents.mcp_agent import MCPAgent, MCPAgentConfig


class SupervisorConfig:
    """Configuration for the supervisor agent."""
    def __init__(
        self,
        llm_provider="openai",
        openai_model="gpt-4o",
        anthropic_model="claude-3-opus-20240229",
        temperature=0,
        streaming=True,
        system_message=None,
        use_mcp=False,
        complexity_threshold=0.7,
        mcp_config=None
    ):
        self.llm_provider = llm_provider
        self.openai_model = openai_model
        self.anthropic_model = anthropic_model
        self.temperature = temperature
        self.streaming = streaming
        self.use_mcp = use_mcp
        self.complexity_threshold = complexity_threshold
        self.mcp_config = mcp_config or MCPAgentConfig(
            llm_provider=llm_provider,
            openai_model=openai_model,
            anthropic_model=anthropic_model,
            temperature=temperature,
            streaming=streaming
        )
        self.system_message = system_message or """
        You are a supervisor agent that coordinates multiple specialized agents to solve complex tasks.
        Your job is to:
        1. Understand the user's request
        2. Determine which specialized agent(s) should handle the request
        3. Coordinate the flow of information between agents
        4. Synthesize a final response for the user

        Always think carefully about which agent(s) would be most appropriate for the task.
        You can use multiple agents in sequence or in parallel if needed.
        """


class Supervisor:
    """
    Supervisor agent that orchestrates multiple specialized agents.

    The supervisor can operate in two modes:
    1. Standard mode: Simple delegation to a single agent
    2. MCP mode: Complex task breakdown and delegation to multiple agents
    """

    def __init__(
        self,
        config=None,
        agents=None,
        use_task_descriptions=False  # Ignored in simplified version
    ):
        """
        Initialize the supervisor agent.

        Args:
            config: Configuration for the supervisor
            agents: Dictionary of agent functions keyed by agent name
            use_task_descriptions: Whether to use task descriptions for handoffs (ignored in simplified version)
        """
        self.config = config or SupervisorConfig()
        self.agents = agents or {}

        # Initialize LLM based on provider
        if self.config.llm_provider == "openai":
            self.llm = ChatOpenAI(
                model=self.config.openai_model,
                temperature=self.config.temperature,
                streaming=self.config.streaming
            )
        elif self.config.llm_provider == "anthropic":
            self.llm = ChatAnthropic(
                model=self.config.anthropic_model,
                temperature=self.config.temperature,
                streaming=self.config.streaming
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

        # Initialize MCP agent if enabled
        self.mcp_agent = None
        if self.config.use_mcp:
            self.mcp_agent = MCPAgent(
                config=self.config.mcp_config,
                agents=self.agents
            )

    def _determine_next_agent(self, query):
        """
        Determine which agent to use based on the query.

        Args:
            query: User query

        Returns:
            str: Name of the agent to use, or None if no agent is appropriate
        """
        # Create a prompt for the LLM to determine the next agent
        prompt = f"""
        System: {self.config.system_message}

        Available agents: {', '.join(self.agents.keys())}

        User query: {query}

        Which agent would be most appropriate to handle this query? Respond with just the agent name, or "none" if no agent is appropriate.
        """

        # Get response from LLM
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        # Extract agent name from response
        if isinstance(response, dict):
            agent_name = response.get("content", "").strip().lower()
        elif hasattr(response, "content"):
            agent_name = response.content.strip().lower()
        else:
            agent_name = str(response).strip().lower()



        # Check if the agent exists based on LLM response
        for name in self.agents.keys():
            if name.lower() in agent_name:
                return name

        return None

    def _get_human_feedback(self, state):
        """
        Get feedback from a human user.

        Args:
            state: Current state of the system

        Returns:
            dict: Updated state with human feedback
        """
        # In a real implementation, this would wait for user input
        # For now, we'll just simulate it

        # Display the current state to the user
        print("\n=== Requesting Human Feedback ===")
        print(f"Current messages: {state['messages'][-1]}")

        # Get feedback (in a real implementation, this would be from user input)
        feedback = input("Please provide feedback: ")

        # Update state with human feedback
        state["human_input"] = {"feedback": feedback}
        state["messages"].append({"role": "user", "content": feedback})

        return state

    def _synthesize_final_response(self, messages, agent_outputs):
        """
        Synthesize a final response based on agent outputs.

        Args:
            messages: List of messages in the conversation
            agent_outputs: Outputs from each agent

        Returns:
            str: Synthesized response
        """
        # Create a prompt for the LLM to synthesize a response
        prompt = f"""
        System: {self.config.system_message}

        You need to synthesize a final response based on the outputs from various agents.

        Conversation history:
        {messages}

        Agent outputs:
        {agent_outputs}

        Please synthesize a helpful response that incorporates the information from the agents.
        """

        # Get response from LLM
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        # Handle different response formats
        if isinstance(response, dict):
            return response.get("content", "No content available")
        elif hasattr(response, "content"):
            return response.content
        else:
            return str(response)

    def _assess_task_complexity(self, query):
        """
        Assess the complexity of a task to determine if it should be handled by the MCP agent.

        Args:
            query: User query

        Returns:
            float: Complexity score between 0 and 1
            bool: Whether to use MCP agent
        """
        # If MCP is disabled, always return False
        if not self.config.use_mcp or self.mcp_agent is None:
            return 0.0, False

        # Create a prompt for the LLM to assess task complexity
        prompt = f"""
        System: You are an AI assistant that evaluates the complexity of tasks.

        User query: {query}

        On a scale of 0 to 1, how complex is this task? Consider the following factors:
        - Does it require multiple steps or subtasks?
        - Does it involve different types of operations (search, analysis, generation, etc.)?
        - Would it benefit from being broken down into smaller tasks?
        - Does it require coordination between different specialized agents?

        Respond with just a number between 0 and 1, where:
        - 0: Simple task that can be handled by a single agent
        - 1: Complex task that should be broken down and delegated to multiple agents
        """

        # Get response from LLM
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        # Extract complexity score from response
        complexity_score = 0.0
        try:
            if isinstance(response, dict):
                content = response.get("content", "")
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Extract the first number from the response
            import re
            numbers = re.findall(r"0\.\d+|\d+\.\d+|\d+", content)
            if numbers:
                complexity_score = float(numbers[0])
                # Ensure the score is between 0 and 1
                complexity_score = max(0.0, min(1.0, complexity_score))
        except Exception as e:
            print(f"Error extracting complexity score: {str(e)}")
            complexity_score = 0.0

        # Determine whether to use MCP based on complexity threshold
        use_mcp = complexity_score >= self.config.complexity_threshold

        return complexity_score, use_mcp

    @tracer.trace_supervisor("StandardSupervisor")
    def invoke(self, query, stream=False):
        """
        Process a user query using the multi-agent system.

        Args:
            query: User query
            stream: Whether to stream the response

        Returns:
            dict: Final state after processing
        """
        # Initialize state
        state = {
            "messages": [{"role": "user", "content": query}],
            "agent_outputs": {},
            "human_input": None,
            "stream": stream
        }

        # Assess task complexity and determine whether to use MCP
        complexity_score, use_mcp = self._assess_task_complexity(query)
        state["complexity_score"] = complexity_score

        # If task is complex and MCP is enabled, use MCP agent
        if use_mcp and self.mcp_agent is not None:
            print(f"Using MCP agent for complex task (complexity score: {complexity_score:.2f})")
            return self.mcp_agent.invoke(state)

        # Otherwise, use standard supervisor logic
        print(f"Using standard supervisor (complexity score: {complexity_score:.2f})")

        # Determine which agent to use
        agent_name = self._determine_next_agent(query)
        state["next_agent"] = agent_name

        # If no agent is appropriate, generate a response directly
        if agent_name is None:
            response = self.llm.invoke([
                {"role": "system", "content": self.config.system_message},
                {"role": "user", "content": query}
            ])
            state["messages"].append({"role": "assistant", "content": response.content})
            return state

        # Call the agent
        agent = self.agents[agent_name]
        updated_state = agent(state)

        # Check if we need human feedback
        if "request_human_feedback" in updated_state.get("agent_outputs", {}).get(agent_name, {}):
            updated_state = self._get_human_feedback(updated_state)

        # Synthesize final response if needed
        if stream:
            # For streaming, just return the updated state
            return updated_state
        else:
            # For non-streaming, synthesize a final response
            final_response = self._synthesize_final_response(
                updated_state["messages"],
                updated_state["agent_outputs"]
            )
            updated_state["messages"].append({"role": "assistant", "content": final_response})
            return updated_state



