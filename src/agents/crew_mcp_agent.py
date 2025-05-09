"""
CrewAI Style MCP Agent

This module implements a CrewAI-style Master Control Program (MCP) agent, which:
1. Assigns specific roles to specialized agents
2. Creates a hierarchical team structure with clear responsibilities
3. Manages task delegation based on agent roles and expertise
4. Coordinates sequential and parallel execution of tasks
5. Provides a coherent final response by integrating results from the crew

The CrewAI style MCP is inspired by the CrewAI framework, which focuses on
role-based agent teams with clear responsibilities and hierarchical structures.
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
class AgentRole:
    """Definition of an agent role in the crew."""
    name: str
    agent_name: str
    description: str
    goals: List[str]
    backstory: str = ""


@dataclass
class CrewMCPAgentConfig:
    """Configuration for the CrewAI-style MCP agent."""
    llm_provider: str = "openai"
    openai_model: str = "gpt-4o"
    openai_reasoning_model: str = "o3"
    openai_efficient_model: str = "o3-mini"
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    anthropic_reasoning_model: str = "claude-3-7-haiku-20250201"
    temperature: float = 0
    streaming: bool = True
    max_agents: int = 5
    system_message: str = """
    You are the Crew Manager, responsible for coordinating a team of specialized AI agents.
    Your job is to:
    1. Analyze the task to understand its requirements
    2. Assign roles to specialized agents based on their expertise
    3. Create a hierarchical team structure with clear responsibilities
    4. Delegate subtasks to appropriate agents based on their roles
    5. Coordinate sequential and parallel execution of tasks
    6. Integrate results from the crew into a coherent final response

    Always think carefully about which agent would be most appropriate for each role.
    Consider the expertise and capabilities of each agent when assigning roles.
    """
    planning_template: str = """
    Task: {task}

    Please analyze this task and create a crew of specialized agents to handle it.

    Available agents:
    {available_agents}

    For each agent in the crew, specify:
    1. Role name
    2. Agent to assign to this role
    3. Role description
    4. Goals for this role
    5. Backstory (optional)

    Then provide a workflow plan that specifies:
    1. The hierarchical structure of the crew
    2. The sequence of operations
    3. How the results will be integrated
    """


class CrewMCPAgent:
    """
    CrewAI-style Master Control Program (MCP) agent for role-based team coordination.
    """

    def __init__(
        self,
        config: Optional[CrewMCPAgentConfig] = None,
        agents: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the CrewAI-style MCP agent.

        Args:
            config: Configuration for the MCP agent
            agents: Dictionary of agent functions keyed by agent name
        """
        self.config = config or CrewMCPAgentConfig()
        self.agents = agents or {}

        # Format system message with config values
        self.config.system_message = self.config.system_message.format(
            max_agents=self.config.max_agents
        )

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

    def _create_crew_plan(self, task: str) -> Dict[str, Any]:
        """
        Create a crew plan for a task.

        Args:
            task: The task to plan for

        Returns:
            Dict: Crew plan with roles, hierarchy, and workflow
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

    def _parse_crew_plan(self, raw_plan: str) -> Dict[str, Any]:
        """
        Parse the raw crew plan into a structured format.

        Args:
            raw_plan: Raw crew plan from LLM

        Returns:
            Dict: Structured crew plan
        """
        # This is a simplified implementation - in a real system, you would want to
        # parse the plan more carefully to extract structured information

        # For now, we'll use a simple heuristic to extract roles and workflow
        lines = raw_plan.strip().split("\n")
        roles = []
        current_role = None

        # Extract roles
        for line in lines:
            line = line.strip()

            # Look for role headers
            if "role:" in line.lower() or "role name:" in line.lower():
                # If we were processing a role, add it to the list
                if current_role:
                    roles.append(current_role)

                # Start a new role
                current_role = {
                    "name": line.split(":", 1)[1].strip(),
                    "agent": None,
                    "description": "",
                    "goals": [],
                    "backstory": ""
                }
            elif current_role:
                # Look for agent assignment
                if "agent:" in line.lower():
                    agent_part = line.lower().split("agent:")[1].strip()
                    # Extract agent name
                    for agent_name in self.agents.keys():
                        if agent_name.lower() in agent_part:
                            current_role["agent"] = agent_name
                            break

                # Look for role description
                elif "description:" in line.lower():
                    current_role["description"] = line.split(":", 1)[1].strip()

                # Look for goals
                elif "goal:" in line.lower() or "goals:" in line.lower():
                    goals_part = line.split(":", 1)[1].strip()
                    current_role["goals"] = [g.strip() for g in goals_part.split(",")]

                # Look for backstory
                elif "backstory:" in line.lower():
                    current_role["backstory"] = line.split(":", 1)[1].strip()

        # Add the last role if there is one
        if current_role:
            roles.append(current_role)

        # Extract workflow
        workflow = []
        workflow_section = False

        for line in lines:
            if "workflow" in line.lower() or "sequence" in line.lower() or "plan" in line.lower():
                workflow_section = True
                continue

            if workflow_section and line.strip():
                workflow.append(line.strip())

        return {
            "roles": roles,
            "workflow": workflow
        }

    def _execute_role(self, role: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a role using the appropriate agent.

        Args:
            role: Role to execute
            state: Current state

        Returns:
            Dict: Updated state with role results
        """
        agent_name = role["agent"]

        if agent_name not in self.agents:
            raise ValueError(f"Agent not found: {agent_name}")

        agent = self.agents[agent_name]

        # Update state with role information
        state["current_role"] = role["name"]
        state["role_description"] = role["description"]
        state["role_goals"] = role["goals"]

        # Execute the agent
        updated_state = agent(state)

        # Store the result
        if "role_results" not in updated_state:
            updated_state["role_results"] = {}

        updated_state["role_results"][role["name"]] = {
            "agent": agent_name,
            "result": updated_state.get("agent_outputs", {}).get(agent_name, "No output")
        }

        return updated_state

    @tracer.trace_agent("CrewMCPAgent")
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task using the CrewAI-style MCP agent.

        Args:
            state: Current state

        Returns:
            Dict: Updated state with results
        """
        # Extract query from state
        query = state["messages"][-1]["content"] if state.get("messages") else "No query"

        # Create crew plan
        plan_data = self._create_crew_plan(query)
        parsed_plan = self._parse_crew_plan(plan_data["raw_plan"])

        # Store the plan in the state
        state["crew_plan"] = parsed_plan

        # Execute roles according to the workflow
        for role in parsed_plan["roles"]:
            # Execute the role
            state = self._execute_role(role, state)

        # Synthesize final response
        final_response = self._synthesize_final_response(state)

        # Update state with final response
        state["messages"].append({"role": "assistant", "content": final_response})

        return state

    def _synthesize_final_response(self, state: Dict[str, Any]) -> str:
        """
        Synthesize a final response based on role results.

        Args:
            state: Current state

        Returns:
            str: Synthesized response
        """
        # Create a prompt for the LLM to synthesize a response
        role_results = state.get("role_results", {})
        results_str = json.dumps(role_results, indent=2)

        prompt = f"""
        System: {self.config.system_message}

        You need to synthesize a final response based on the results from various roles in the crew.

        Original task: {state["messages"][0]["content"]}

        Crew plan: {state["crew_plan"]}

        Role results:
        {results_str}

        Please synthesize a helpful response that integrates the information from all roles.
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
