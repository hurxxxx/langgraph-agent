"""
LangGraph Style MCP Agent

This module implements a LangGraph-style Master Control Program (MCP) agent, which:
1. Creates a graph-based workflow with nodes and edges
2. Defines conditional routing between agents based on state
3. Supports both sequential and parallel execution paths
4. Manages state transitions between agents
5. Provides a coherent final response by following the graph execution

The LangGraph style MCP is inspired by the LangGraph framework, which focuses on
graph-based workflows with clear control flow and state management.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field

# Import LangSmith utilities
from src.utils.langsmith_utils import tracer

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate


@dataclass
class Node:
    """Definition of a node in the graph."""
    name: str
    agent_name: str
    description: str


@dataclass
class Edge:
    """Definition of an edge in the graph."""
    source: str
    target: str
    condition: str = ""


@dataclass
class LangGraphMCPAgentConfig:
    """Configuration for the LangGraph-style MCP agent."""
    llm_provider: str = "openai"
    openai_model: str = "gpt-4o"
    openai_reasoning_model: str = "o3"
    openai_efficient_model: str = "o3-mini"
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    anthropic_reasoning_model: str = "claude-3-7-haiku-20250201"
    temperature: float = 0
    streaming: bool = True
    max_nodes: int = 10
    system_message: str = """
    You are the Graph Manager, responsible for creating and executing graph-based workflows.
    Your job is to:
    1. Analyze the task to understand its requirements
    2. Create a graph with nodes (agents) and edges (transitions)
    3. Define conditional routing between agents
    4. Execute the graph by following the defined paths
    5. Manage state transitions between agents
    6. Provide a coherent final response based on the graph execution

    Always think carefully about the structure of the graph.
    Consider both sequential and parallel execution paths when appropriate.
    """
    planning_template: str = """
    Task: {task}

    Please analyze this task and create a graph-based workflow to handle it.

    Available agents:
    {available_agents}

    For each node in the graph, specify:
    1. Node name
    2. Agent to assign to this node
    3. Node description

    For each edge in the graph, specify:
    1. Source node
    2. Target node
    3. Condition for taking this edge (optional)

    Then provide a graph execution plan that specifies:
    1. The entry point node
    2. The expected flow through the graph
    3. How to determine when execution is complete
    """


class LangGraphMCPAgent:
    """
    LangGraph-style Master Control Program (MCP) agent for graph-based workflows.
    """

    def __init__(
        self,
        config: Optional[LangGraphMCPAgentConfig] = None,
        agents: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the LangGraph-style MCP agent.

        Args:
            config: Configuration for the MCP agent
            agents: Dictionary of agent functions keyed by agent name
        """
        self.config = config or LangGraphMCPAgentConfig()
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

    def _create_graph_plan(self, task: str) -> Dict[str, Any]:
        """
        Create a graph plan for a task.

        Args:
            task: The task to plan for

        Returns:
            Dict: Graph plan with nodes, edges, and execution plan
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

    def _parse_graph_plan(self, raw_plan: str) -> Dict[str, Any]:
        """
        Parse the raw graph plan into a structured format.

        Args:
            raw_plan: Raw graph plan from LLM

        Returns:
            Dict: Structured graph plan
        """
        # This is a simplified implementation - in a real system, you would want to
        # parse the plan more carefully to extract structured information

        # For now, we'll use a simple heuristic to extract nodes and edges
        lines = raw_plan.strip().split("\n")
        nodes = []
        current_node = None

        # Extract nodes
        for line in lines:
            line = line.strip()

            # Look for node headers
            if "node:" in line.lower() or "node name:" in line.lower():
                # If we were processing a node, add it to the list
                if current_node:
                    nodes.append(current_node)

                # Start a new node
                current_node = {
                    "name": line.split(":", 1)[1].strip(),
                    "agent_name": None,
                    "description": ""
                }
            elif current_node:
                # Look for agent assignment
                if "agent:" in line.lower():
                    agent_part = line.lower().split("agent:")[1].strip()
                    # Extract agent name
                    for agent_name in self.agents.keys():
                        if agent_name.lower() in agent_part:
                            current_node["agent_name"] = agent_name
                            break

                # Look for node description
                elif "description:" in line.lower():
                    current_node["description"] = line.split(":", 1)[1].strip()

        # Add the last node if there is one
        if current_node:
            nodes.append(current_node)

        # Extract edges
        edges = []
        edge_section = False
        current_edge = None

        for line in lines:
            line = line.strip()

            # Look for edge section
            if "edges:" in line.lower() or "connections:" in line.lower():
                edge_section = True
                continue

            if edge_section:
                # Look for edge definitions
                if "->" in line or "to" in line.lower():
                    # If we were processing an edge, add it to the list
                    if current_edge:
                        edges.append(current_edge)

                    # Parse the edge
                    if "->" in line:
                        parts = line.split("->")
                        source = parts[0].strip()
                        target = parts[1].strip()
                    else:
                        parts = line.lower().split("to")
                        source = parts[0].replace("from", "").strip()
                        target = parts[1].strip()

                    # Start a new edge
                    current_edge = {
                        "source": source,
                        "target": target,
                        "condition": ""
                    }
                elif current_edge and "condition:" in line.lower():
                    current_edge["condition"] = line.split(":", 1)[1].strip()

        # Add the last edge if there is one
        if current_edge:
            edges.append(current_edge)

        # Extract execution plan
        execution_plan = {}
        execution_section = False

        for line in lines:
            if "execution plan:" in line.lower() or "execution:" in line.lower():
                execution_section = True
                continue

            if execution_section and line.strip():
                if "entry point:" in line.lower() or "start:" in line.lower():
                    execution_plan["entry_point"] = line.split(":", 1)[1].strip()
                elif "completion:" in line.lower() or "end:" in line.lower():
                    execution_plan["completion"] = line.split(":", 1)[1].strip()
                else:
                    if "steps" not in execution_plan:
                        execution_plan["steps"] = []
                    execution_plan["steps"].append(line.strip())

        return {
            "nodes": nodes,
            "edges": edges,
            "execution_plan": execution_plan
        }

    def _execute_graph(self, graph_plan: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a graph based on the graph plan.

        Args:
            graph_plan: Graph plan with nodes, edges, and execution plan
            state: Current state

        Returns:
            Dict: Updated state with graph execution results
        """
        # Get the entry point node
        entry_point = graph_plan["execution_plan"].get("entry_point")

        # Find the entry point node
        current_node = None
        for node in graph_plan["nodes"]:
            if node["name"].lower() == entry_point.lower():
                current_node = node
                break

        # If no entry point is specified, use the first node
        if not current_node and graph_plan["nodes"]:
            current_node = graph_plan["nodes"][0]

        # Initialize graph execution history
        if "graph_execution" not in state:
            state["graph_execution"] = []

        # Execute the graph
        visited_nodes = set()
        max_steps = 20  # Prevent infinite loops

        for step in range(max_steps):
            # If we've reached a terminal state or visited all nodes, break
            if not current_node or len(visited_nodes) >= len(graph_plan["nodes"]):
                break

            # Get the agent for the current node
            agent_name = current_node["agent_name"]

            if agent_name not in self.agents:
                raise ValueError(f"Agent not found: {agent_name}")

            agent = self.agents[agent_name]

            # Update state with node information
            state["current_node"] = current_node["name"]
            state["node_description"] = current_node["description"]

            # Execute the agent
            updated_state = agent(state)

            # Record the execution
            updated_state["graph_execution"].append({
                "node": current_node["name"],
                "agent": agent_name,
                "result": updated_state.get("agent_outputs", {}).get(agent_name, "No output")
            })

            # Mark the node as visited
            visited_nodes.add(current_node["name"])

            # Find the next node based on edges
            next_node = None
            for edge in graph_plan["edges"]:
                if edge["source"].lower() == current_node["name"].lower():
                    # If there's a condition, evaluate it
                    if edge["condition"]:
                        # This is a simplified condition evaluation - in a real system,
                        # you would want to evaluate the condition more carefully
                        condition_met = self._evaluate_condition(edge["condition"], updated_state)
                        if condition_met:
                            # Find the target node
                            for node in graph_plan["nodes"]:
                                if node["name"].lower() == edge["target"].lower():
                                    next_node = node
                                    break
                            if next_node:
                                break
                    else:
                        # If there's no condition, just follow the edge
                        for node in graph_plan["nodes"]:
                            if node["name"].lower() == edge["target"].lower():
                                next_node = node
                                break
                        if next_node:
                            break

            # Update the current node
            current_node = next_node

            # Update the state
            state = updated_state

        return state

    def _evaluate_condition(self, condition: str, state: Dict[str, Any]) -> bool:
        """
        Evaluate a condition based on the current state.

        Args:
            condition: Condition to evaluate
            state: Current state

        Returns:
            bool: Whether the condition is met
        """
        # This is a simplified condition evaluation - in a real system,
        # you would want to evaluate the condition more carefully

        # For now, we'll just check if certain keywords are present in the agent outputs
        agent_outputs = state.get("agent_outputs", {})

        # Check for positive conditions
        if "success" in condition.lower() or "complete" in condition.lower():
            # Check if any agent output contains success or complete
            for output in agent_outputs.values():
                if isinstance(output, dict):
                    output_str = json.dumps(output)
                else:
                    output_str = str(output)

                if "success" in output_str.lower() or "complete" in output_str.lower():
                    return True

        # Check for negative conditions
        if "failure" in condition.lower() or "error" in condition.lower():
            # Check if any agent output contains failure or error
            for output in agent_outputs.values():
                if isinstance(output, dict):
                    output_str = json.dumps(output)
                else:
                    output_str = str(output)

                if "failure" in output_str.lower() or "error" in output_str.lower():
                    return True

        # Default to True for empty or unrecognized conditions
        return True

    @tracer.trace_agent("LangGraphMCPAgent")
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task using the LangGraph-style MCP agent.

        Args:
            state: Current state

        Returns:
            Dict: Updated state with results
        """
        # Extract query from state
        query = state["messages"][-1]["content"] if state.get("messages") else "No query"

        # Create graph plan
        plan_data = self._create_graph_plan(query)
        parsed_plan = self._parse_graph_plan(plan_data["raw_plan"])

        # Store the plan in the state
        state["graph_plan"] = parsed_plan

        # Execute the graph
        state = self._execute_graph(parsed_plan, state)

        # Synthesize final response
        final_response = self._synthesize_final_response(state)

        # Update state with final response
        state["messages"].append({"role": "assistant", "content": final_response})

        return state

    def _synthesize_final_response(self, state: Dict[str, Any]) -> str:
        """
        Synthesize a final response based on the graph execution.

        Args:
            state: Current state

        Returns:
            str: Synthesized response
        """
        # Create a prompt for the LLM to synthesize a response
        graph_execution = state.get("graph_execution", [])
        execution_str = json.dumps(graph_execution, indent=2)

        prompt = f"""
        System: {self.config.system_message}

        You need to synthesize a final response based on the graph execution.

        Original task: {state["messages"][0]["content"]}

        Graph plan: {state["graph_plan"]}

        Graph execution:
        {execution_str}

        Please synthesize a helpful response that integrates the information from the graph execution.
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
