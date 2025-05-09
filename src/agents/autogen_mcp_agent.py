"""
AutoGen Style MCP Agent

This module implements an AutoGen-style Master Control Program (MCP) agent, which:
1. Creates a conversational multi-agent system
2. Enables agents to communicate with each other in a chat-like format
3. Allows agents to request assistance from other agents
4. Supports dynamic agent selection based on conversation context
5. Provides a coherent final response by integrating the multi-agent conversation

The AutoGen style MCP is inspired by the Microsoft AutoGen framework, which focuses on
conversational multi-agent systems with dynamic agent interactions.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

# Import LangSmith utilities
from src.utils.langsmith_utils import tracer

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate


@dataclass
class AgentConfig:
    """Configuration for an agent in the AutoGen-style system."""
    name: str
    agent_name: str
    system_message: str
    can_call_agents: List[str] = field(default_factory=list)


@dataclass
class AutoGenMCPAgentConfig:
    """Configuration for the AutoGen-style MCP agent."""
    llm_provider: str = "openai"
    openai_model: str = "gpt-4o"
    openai_reasoning_model: str = "o3"
    openai_efficient_model: str = "o3-mini"
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    anthropic_reasoning_model: str = "claude-3-7-haiku-20250201"
    temperature: float = 0
    streaming: bool = True
    max_turns: int = 15
    system_message: str = """
    You are the Conversation Manager, responsible for coordinating a conversational multi-agent system.
    Your job is to:
    1. Analyze the task to understand its requirements
    2. Set up a group chat with specialized agents
    3. Facilitate communication between agents
    4. Allow agents to request assistance from other agents
    5. Determine when the conversation has reached a conclusion
    6. Integrate the conversation into a coherent final response

    Always think carefully about which agents should participate in the conversation.
    Consider the expertise and capabilities of each agent when setting up the group chat.
    """
    planning_template: str = """
    Task: {task}

    Please analyze this task and create a conversational multi-agent system to handle it.

    Available agents:
    {available_agents}

    For each agent in the conversation, specify:
    1. Agent name
    2. System message (instructions for the agent)
    3. Which other agents this agent can call for assistance

    Then provide a conversation plan that specifies:
    1. Which agent should start the conversation
    2. The expected flow of the conversation
    3. How to determine when the conversation is complete
    """


class AutoGenMCPAgent:
    """
    AutoGen-style Master Control Program (MCP) agent for conversational multi-agent systems.
    """

    def __init__(
        self,
        config: Optional[AutoGenMCPAgentConfig] = None,
        agents: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the AutoGen-style MCP agent.

        Args:
            config: Configuration for the MCP agent
            agents: Dictionary of agent functions keyed by agent name
        """
        self.config = config or AutoGenMCPAgentConfig()
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

        # Initialize planning prompt template
        self.planning_prompt = PromptTemplate(
            template=self.config.planning_template,
            input_variables=["task", "available_agents"]
        )

    def _create_conversation_plan(self, task: str) -> Dict[str, Any]:
        """
        Create a conversation plan for a task.

        Args:
            task: The task to plan for

        Returns:
            Dict: Conversation plan with agent configurations and flow
        """
        # Format available agents for the prompt
        available_agents_str = "\n".join([f"- {name}: {self.agents[name].__doc__ or 'No description'}" for name in self.agents.keys()])

        # Format the planning prompt
        planning_input = self.planning_prompt.format(
            task=task,
            available_agents=available_agents_str
        )

        # Get planning response from LLM
        messages = [
            SystemMessage(content=self.config.system_message),
            HumanMessage(content=planning_input)
        ]

        response = self.llm.invoke(messages)

        # Return the raw plan for now
        return {
            "raw_plan": response.content,
            "task": task
        }

    def _parse_conversation_plan(self, raw_plan: str) -> Dict[str, Any]:
        """
        Parse the raw conversation plan into a structured format.

        Args:
            raw_plan: Raw conversation plan from LLM

        Returns:
            Dict: Structured conversation plan
        """
        # This is a simplified implementation - in a real system, you would want to
        # parse the plan more carefully to extract structured information

        # For now, we'll use a simple heuristic to extract agent configurations and flow
        lines = raw_plan.strip().split("\n")
        agent_configs = []
        current_agent = None

        # Extract agent configurations
        for line in lines:
            line = line.strip()

            # Look for agent headers
            if "agent:" in line.lower() or "agent name:" in line.lower():
                # If we were processing an agent, add it to the list
                if current_agent:
                    agent_configs.append(current_agent)

                # Start a new agent
                current_agent = {
                    "name": line.split(":", 1)[1].strip(),
                    "agent_name": None,
                    "system_message": "",
                    "can_call_agents": []
                }
            elif current_agent:
                # Look for agent assignment
                if "agent type:" in line.lower() or "base agent:" in line.lower():
                    agent_part = line.lower().split(":", 1)[1].strip()
                    # Extract agent name
                    for agent_name in self.agents.keys():
                        if agent_name.lower() in agent_part:
                            current_agent["agent_name"] = agent_name
                            break

                # Look for system message
                elif "system message:" in line.lower() or "instructions:" in line.lower():
                    current_agent["system_message"] = line.split(":", 1)[1].strip()

                # Look for can call agents
                elif "can call:" in line.lower() or "can request:" in line.lower():
                    calls_part = line.split(":", 1)[1].strip()
                    current_agent["can_call_agents"] = [c.strip() for c in calls_part.split(",")]

        # Add the last agent if there is one
        if current_agent:
            agent_configs.append(current_agent)

        # Extract conversation flow
        flow = {}
        flow_section = False

        for line in lines:
            if "flow:" in line.lower() or "conversation flow:" in line.lower():
                flow_section = True
                continue

            if flow_section and line.strip():
                if "start:" in line.lower() or "initiator:" in line.lower():
                    flow["start"] = line.split(":", 1)[1].strip()
                elif "completion:" in line.lower() or "termination:" in line.lower():
                    flow["completion"] = line.split(":", 1)[1].strip()
                else:
                    if "steps" not in flow:
                        flow["steps"] = []
                    flow["steps"].append(line.strip())

        return {
            "agent_configs": agent_configs,
            "flow": flow
        }

    def _simulate_conversation(self, conversation_plan: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a conversation between agents based on the conversation plan.

        Args:
            conversation_plan: Conversation plan with agent configurations and flow
            state: Current state

        Returns:
            Dict: Updated state with conversation results
        """
        # Extract query from state
        query = state["messages"][-1]["content"] if state.get("messages") else "No query"

        # Initialize conversation
        conversation = [
            {"role": "user", "content": query}
        ]

        # Get the starting agent
        start_agent = conversation_plan["flow"].get("start", conversation_plan["agent_configs"][0]["name"])

        # Find the starting agent configuration
        start_agent_config = None
        for agent_config in conversation_plan["agent_configs"]:
            if agent_config["name"].lower() == start_agent.lower():
                start_agent_config = agent_config
                break

        if not start_agent_config:
            start_agent_config = conversation_plan["agent_configs"][0]

        # Initialize current agent
        current_agent = start_agent_config

        # Simulate conversation for a maximum number of turns
        for turn in range(self.config.max_turns):
            # Create a prompt for the current agent
            agent_prompt = f"""
            System: {current_agent['system_message']}

            You are {current_agent['name']}, participating in a multi-agent conversation to solve a task.

            Task: {query}

            Conversation so far:
            {json.dumps(conversation, indent=2)}

            Your response should either:
            1. Provide information or take action to help solve the task
            2. Request assistance from another agent by starting with "I need help from [agent name]:"
            3. Indicate the task is complete by starting with "TASK COMPLETE:"

            You can call these agents for assistance: {', '.join(current_agent['can_call_agents'])}
            """

            # Get response from LLM
            response = self.llm.invoke([{"role": "user", "content": agent_prompt}])

            # Extract content from response
            if isinstance(response, dict):
                content = response.get("content", "No content available")
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Add response to conversation
            conversation.append({
                "role": "assistant",
                "name": current_agent["name"],
                "content": content
            })

            # Check if task is complete
            if "TASK COMPLETE:" in content:
                break

            # Check if agent is requesting help
            if "I need help from" in content:
                # Extract the requested agent name
                requested_agent = None
                for agent_config in conversation_plan["agent_configs"]:
                    if agent_config["name"].lower() in content.lower():
                        requested_agent = agent_config
                        break

                # If requested agent is found and is in the can_call_agents list, switch to that agent
                if requested_agent and requested_agent["name"] in current_agent["can_call_agents"]:
                    current_agent = requested_agent

            # If we've reached the maximum number of turns, break
            if turn == self.config.max_turns - 1:
                # Add a final message indicating the conversation was truncated
                conversation.append({
                    "role": "system",
                    "content": "Conversation truncated due to maximum number of turns reached."
                })

        # Store the conversation in the state
        state["conversation"] = conversation

        return state

    @tracer.trace_agent("AutoGenMCPAgent")
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task using the AutoGen-style MCP agent.

        Args:
            state: Current state

        Returns:
            Dict: Updated state with results
        """
        # Extract query from state
        query = state["messages"][-1]["content"] if state.get("messages") else "No query"

        # Create conversation plan
        plan_data = self._create_conversation_plan(query)
        parsed_plan = self._parse_conversation_plan(plan_data["raw_plan"])

        # Store the plan in the state
        state["conversation_plan"] = parsed_plan

        # Simulate conversation
        state = self._simulate_conversation(parsed_plan, state)

        # Synthesize final response
        final_response = self._synthesize_final_response(state)

        # Update state with final response
        state["messages"].append({"role": "assistant", "content": final_response})

        return state

    def _synthesize_final_response(self, state: Dict[str, Any]) -> str:
        """
        Synthesize a final response based on the conversation.

        Args:
            state: Current state

        Returns:
            str: Synthesized response
        """
        # Create a prompt for the LLM to synthesize a response
        conversation = state.get("conversation", [])
        conversation_str = json.dumps(conversation, indent=2)

        prompt = f"""
        System: {self.config.system_message}

        You need to synthesize a final response based on a multi-agent conversation.

        Original task: {state["messages"][0]["content"]}

        Conversation:
        {conversation_str}

        Please synthesize a helpful response that integrates the information from the conversation.
        The response should be coherent, comprehensive, and directly address the original task.
        """

        # Get response from LLM
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        # Extract content from response
        if isinstance(response, dict):
            return response.get("content", "No content available")
        elif hasattr(response, "content"):
            return response.content
        else:
            return str(response)
