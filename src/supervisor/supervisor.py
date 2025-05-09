"""
Simplified Supervisor Agent for Multi-Agent System

This module implements a simplified supervisor agent that orchestrates multiple specialized agents.
The supervisor is responsible for:
1. Understanding user queries
2. Delegating tasks to appropriate specialized agents
3. Coordinating communication between agents
4. Synthesizing final responses

The implementation supports both streaming and non-streaming responses and includes
LangSmith tracing for monitoring and debugging.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Callable

# Import LangSmith utilities
from utils.langsmith_utils import tracer

# Import LangChain components
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
except ImportError:
    print("Warning: LangChain components not available. Using mock implementations.")
    # Mock implementations for testing
    class ChatOpenAI:
        def __init__(self, model=None, temperature=0, streaming=False):
            self.model = model
            self.temperature = temperature
            self.streaming = streaming

        def invoke(self, messages):
            return {"content": f"Response from {self.model} about {messages[-1]['content']}"}

    class ChatAnthropic:
        def __init__(self, model=None, temperature=0, streaming=False):
            self.model = model
            self.temperature = temperature
            self.streaming = streaming

        def invoke(self, messages):
            return {"content": f"Response from {self.model} about {messages[-1]['content']}"}


class SupervisorConfig:
    """Configuration for the supervisor agent."""
    def __init__(
        self,
        llm_provider="openai",
        openai_model="gpt-4o",
        anthropic_model="claude-3-opus-20240229",
        temperature=0,
        streaming=True,
        system_message=None
    ):
        self.llm_provider = llm_provider
        self.openai_model = openai_model
        self.anthropic_model = anthropic_model
        self.temperature = temperature
        self.streaming = streaming
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
    Simplified supervisor agent that orchestrates multiple specialized agents.
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
        try:
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
        except Exception as e:
            print(f"Warning: Could not initialize LLM: {str(e)}")
            # Use a mock implementation
            class MockLLM:
                def __init__(self):
                    self.content = "This is a mock response from the supervisor."

                def invoke(self, messages):
                    # Simple logic to determine which agent to use based on the query
                    query = ""
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            query = msg.get("content", "")
                        elif hasattr(msg, "content"):
                            if getattr(msg, "role", "") == "user":
                                query = msg.content

                    if "search" in query.lower() or "find" in query.lower() or "what is" in query.lower():
                        return {"content": "I'll use the search_agent to find information about this."}
                    elif "image" in query.lower() or "picture" in query.lower() or "generate" in query.lower():
                        return {"content": "I'll use the image_generation_agent to create an image."}
                    elif "store" in query.lower() or "save" in query.lower() or "database" in query.lower():
                        return {"content": "I'll use the vector_storage_agent to store this information."}
                    elif "quality" in query.lower() or "evaluate" in query.lower() or "assess" in query.lower():
                        return {"content": "I'll use the quality_agent to evaluate this."}
                    else:
                        return {"content": "I'll handle this query directly. " + query}

            self.llm = MockLLM()

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

        # Simple keyword matching for mock mode
        if "search" in query.lower() or "find" in query.lower() or "what is" in query.lower():
            for name in self.agents.keys():
                if "search" in name.lower():
                    return name
        elif "image" in query.lower() or "picture" in query.lower() or "generate" in query.lower():
            for name in self.agents.keys():
                if "image" in name.lower():
                    return name
        elif "store" in query.lower() or "save" in query.lower() or "database" in query.lower():
            for name in self.agents.keys():
                if "vector" in name.lower() or "storage" in name.lower():
                    return name
        elif "quality" in query.lower() or "evaluate" in query.lower() or "assess" in query.lower():
            for name in self.agents.keys():
                if "quality" in name.lower():
                    return name

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


# Example usage
if __name__ == "__main__":
    # This is just a placeholder for testing
    def search_agent(state):
        # Simulate search agent
        query = state["messages"][-1]["content"]
        state["agent_outputs"]["search"] = {"results": [f"Search result for {query}"]}
        state["messages"].append({"role": "assistant", "content": f"Found information about {query}"})
        return state

    # Create supervisor with mock agents
    supervisor = Supervisor(
        config=SupervisorConfig(),
        agents={"search_agent": search_agent}
    )

    # Test with a query
    result = supervisor.invoke("Tell me about the history of AI", stream=False)
    print(result["messages"][-1]["content"])
