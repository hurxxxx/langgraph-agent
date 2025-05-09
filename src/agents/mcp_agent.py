"""
Master Control Program (MCP) Agent

This module implements the Master Control Program (MCP) agent, which is responsible for:
1. Breaking down complex tasks into subtasks
2. Planning the execution of subtasks
3. Delegating subtasks to appropriate specialized agents
4. Coordinating the execution of subtasks
5. Integrating results from multiple agents
6. Providing a coherent final response

The MCP agent is designed to handle more complex scenarios than the standard supervisor,
with advanced planning and coordination capabilities.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field

# Import LangSmith utilities
from utils.langsmith_utils import tracer

# Import LangChain components
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate


@dataclass
class MCPAgentConfig:
    """Configuration for the MCP agent."""
    llm_provider: str = "openai"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-opus-20240229"
    temperature: float = 0
    streaming: bool = True
    max_subtasks: int = 5
    system_message: str = """
    You are the Master Control Program (MCP), responsible for high-level planning and orchestration.
    Your job is to break down complex tasks into subtasks and delegate them to specialized agents.
    
    When given a complex task:
    1. Analyze the task to understand its requirements and constraints
    2. Break it down into logical subtasks (maximum {max_subtasks} subtasks)
    3. Determine which specialized agent should handle each subtask
    4. Plan the execution order (sequential or parallel)
    5. Coordinate the execution of subtasks
    6. Integrate results from multiple agents
    7. Provide a coherent final response
    
    Always think carefully about which agent(s) would be most appropriate for each subtask.
    Consider dependencies between subtasks when planning execution order.
    """
    planning_template: str = """
    Task: {task}
    
    Please break down this task into subtasks and create an execution plan.
    
    Available agents:
    {available_agents}
    
    For each subtask, specify:
    1. Subtask description
    2. Agent to handle the subtask
    3. Dependencies (which subtasks must be completed before this one)
    4. Expected output
    
    Then provide an execution plan that specifies the order in which subtasks should be executed,
    taking into account dependencies and opportunities for parallel execution.
    """


class MCPAgent:
    """
    Master Control Program (MCP) agent for high-level planning and orchestration.
    """
    
    def __init__(
        self,
        config: Optional[MCPAgentConfig] = None,
        agents: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the MCP agent.
        
        Args:
            config: Configuration for the MCP agent
            agents: Dictionary of agent functions keyed by agent name
        """
        self.config = config or MCPAgentConfig()
        self.agents = agents or {}
        
        # Format system message with config values
        self.config.system_message = self.config.system_message.format(
            max_subtasks=self.config.max_subtasks
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
    
    def _create_execution_plan(self, task: str) -> Dict[str, Any]:
        """
        Create an execution plan for a complex task.
        
        Args:
            task: The task to plan for
            
        Returns:
            Dict: Execution plan with subtasks, dependencies, and execution order
        """
        # Format available agents for the prompt
        available_agents_str = "\n".join([f"- {name}" for name in self.agents.keys()])
        
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
        
        # Parse the response to extract the execution plan
        # This is a simplified implementation - in a real system, you would want to
        # parse the response more carefully to extract structured information
        
        # For now, we'll just return the raw response and parse it later
        return {
            "raw_plan": response.content,
            "task": task
        }
    
    def _parse_execution_plan(self, raw_plan: str) -> Dict[str, Any]:
        """
        Parse the raw execution plan into a structured format.
        
        Args:
            raw_plan: Raw execution plan from LLM
            
        Returns:
            Dict: Structured execution plan
        """
        # This is a simplified implementation - in a real system, you would want to
        # parse the plan more carefully to extract structured information
        
        # For now, we'll use a simple heuristic to extract subtasks
        lines = raw_plan.strip().split("\n")
        subtasks = []
        current_subtask = None
        
        for line in lines:
            line = line.strip()
            
            # Look for subtask headers (numbered items)
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "0.")):
                # If we were processing a subtask, add it to the list
                if current_subtask:
                    subtasks.append(current_subtask)
                
                # Start a new subtask
                current_subtask = {
                    "description": line,
                    "agent": None,
                    "dependencies": [],
                    "expected_output": ""
                }
            elif current_subtask:
                # Look for agent assignment
                if "agent:" in line.lower():
                    agent_part = line.lower().split("agent:")[1].strip()
                    # Extract agent name
                    for agent_name in self.agents.keys():
                        if agent_name.lower() in agent_part:
                            current_subtask["agent"] = agent_name
                            break
                
                # Look for dependencies
                elif "dependencies:" in line.lower() or "depends on:" in line.lower():
                    deps_part = line.split(":")[1].strip()
                    current_subtask["dependencies"] = [d.strip() for d in deps_part.split(",")]
                
                # Look for expected output
                elif "expected output:" in line.lower() or "output:" in line.lower():
                    output_part = line.split(":")[1].strip()
                    current_subtask["expected_output"] = output_part
                
                # Otherwise, append to the description
                else:
                    current_subtask["description"] += " " + line
        
        # Add the last subtask if there is one
        if current_subtask:
            subtasks.append(current_subtask)
        
        # Extract execution order
        execution_order = []
        execution_section = False
        
        for line in lines:
            if "execution plan" in line.lower() or "execution order" in line.lower():
                execution_section = True
                continue
            
            if execution_section and line.strip():
                execution_order.append(line.strip())
        
        return {
            "subtasks": subtasks,
            "execution_order": execution_order
        }
    
    def _execute_subtask(self, subtask: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single subtask using the appropriate agent.
        
        Args:
            subtask: Subtask to execute
            state: Current state
            
        Returns:
            Dict: Updated state with subtask results
        """
        agent_name = subtask["agent"]
        
        if agent_name not in self.agents:
            raise ValueError(f"Agent not found: {agent_name}")
        
        agent = self.agents[agent_name]
        
        # Update state with subtask information
        state["current_subtask"] = subtask["description"]
        state["expected_output"] = subtask["expected_output"]
        
        # Execute the agent
        updated_state = agent(state)
        
        # Store the result
        if "subtask_results" not in updated_state:
            updated_state["subtask_results"] = {}
        
        updated_state["subtask_results"][subtask["description"]] = {
            "agent": agent_name,
            "result": updated_state.get("agent_outputs", {}).get(agent_name, "No output")
        }
        
        return updated_state
    
    @tracer.trace_agent("MCPAgent")
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complex task using the MCP agent.
        
        Args:
            state: Current state
            
        Returns:
            Dict: Updated state with results
        """
        # Extract query from state
        query = state["messages"][-1]["content"] if state.get("messages") else "No query"
        
        # Create execution plan
        plan_data = self._create_execution_plan(query)
        parsed_plan = self._parse_execution_plan(plan_data["raw_plan"])
        
        # Store the plan in the state
        state["execution_plan"] = parsed_plan
        
        # Execute subtasks according to the execution order
        for subtask in parsed_plan["subtasks"]:
            # Check if all dependencies are satisfied
            # (simplified implementation - in a real system, you would want to
            # check dependencies more carefully)
            
            # Execute the subtask
            state = self._execute_subtask(subtask, state)
        
        # Synthesize final response
        final_response = self._synthesize_final_response(state)
        
        # Update state with final response
        state["messages"].append({"role": "assistant", "content": final_response})
        
        return state
    
    def _synthesize_final_response(self, state: Dict[str, Any]) -> str:
        """
        Synthesize a final response based on subtask results.
        
        Args:
            state: Current state
            
        Returns:
            str: Synthesized response
        """
        # Create a prompt for the LLM to synthesize a response
        subtask_results = state.get("subtask_results", {})
        results_str = json.dumps(subtask_results, indent=2)
        
        prompt = f"""
        System: {self.config.system_message}
        
        You need to synthesize a final response based on the results from various subtasks.
        
        Original task: {state["messages"][0]["content"]}
        
        Execution plan: {state["execution_plan"]}
        
        Subtask results:
        {results_str}
        
        Please synthesize a helpful response that integrates the information from all subtasks.
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
