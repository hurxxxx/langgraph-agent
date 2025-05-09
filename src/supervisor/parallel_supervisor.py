"""
Parallel Supervisor for Multi-Agent System

This module implements a supervisor agent that can orchestrate multiple specialized agents
in parallel using a simpler approach without complex graph structures.
"""

import os
import json
import concurrent.futures
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage


class ParallelSupervisorConfig(BaseModel):
    """Configuration for the parallel supervisor agent."""
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-opus-20240229"
    temperature: float = 0
    streaming: bool = True
    system_message: str = """
    You are a supervisor agent that coordinates multiple specialized agents to solve complex tasks.
    Your job is to:
    1. Understand the user's request
    2. Break down the request into subtasks
    3. Determine which specialized agent(s) should handle each subtask
    4. Coordinate the flow of information between agents
    5. Synthesize a final response for the user

    You can run agents in parallel when appropriate to save time.
    """


class ParallelSupervisor:
    """
    Supervisor agent that orchestrates multiple specialized agents in parallel.
    """

    def __init__(
        self,
        config=None,
        agents=None
    ):
        """
        Initialize the parallel supervisor agent.

        Args:
            config: Configuration for the supervisor
            agents: Dictionary of agent functions keyed by agent name
        """
        self.config = config or ParallelSupervisorConfig()
        self.agents = agents or {}

        # Initialize LLM
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
                def invoke(self, messages):
                    return {"content": "This is a mock response from the supervisor."}
            self.llm = MockLLM()

    def _plan_tasks(self, query: str) -> List[Dict[str, Any]]:
        """
        Plan tasks based on the user query.

        Args:
            query: User query

        Returns:
            List[Dict[str, Any]]: List of planned subtasks
        """
        # Create a prompt for the LLM to break down the task
        prompt = f"""
        System: {self.config.system_message}

        Available agents: {', '.join(self.agents.keys())}

        User query: {query}

        Break down this query into subtasks. For each subtask, specify which agent should handle it.
        Format your response as a JSON array of objects, each with 'subtask_id', 'description', and 'agent' fields.
        Example:
        [
            {{"subtask_id": 1, "description": "Search for information about X", "agent": "search_agent"}},
            {{"subtask_id": 2, "description": "Generate an image of Y", "agent": "image_generation_agent"}}
        ]
        """

        # Get response from LLM
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        # Extract content from response
        if isinstance(response, dict):
            content = response.get("content", "")
        elif hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        # Parse the JSON response
        try:
            # Find JSON array in the response
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                subtasks = json.loads(json_str)
            else:
                # Fallback: create subtasks based on keywords
                subtasks = self._create_fallback_subtasks(query)
        except Exception as e:
            print(f"Error parsing subtasks: {str(e)}")
            # Fallback: create subtasks based on keywords
            subtasks = self._create_fallback_subtasks(query)

        return subtasks

    def _create_fallback_subtasks(self, query: str) -> List[Dict[str, Any]]:
        """
        Create fallback subtasks based on keywords in the query.

        Args:
            query: User query

        Returns:
            List[Dict[str, Any]]: List of subtasks
        """
        subtasks = []
        subtask_id = 1

        # Check for search-related keywords
        if any(keyword in query.lower() for keyword in ["search", "find", "information", "what is", "tell me about"]):
            for agent_name in self.agents.keys():
                if "search" in agent_name.lower():
                    subtasks.append({
                        "subtask_id": subtask_id,
                        "description": f"Search for information about {query}",
                        "agent": agent_name
                    })
                    subtask_id += 1
                    break

        # Check for image-related keywords
        if any(keyword in query.lower() for keyword in ["image", "picture", "generate", "create", "draw"]):
            for agent_name in self.agents.keys():
                if "image" in agent_name.lower():
                    subtasks.append({
                        "subtask_id": subtask_id,
                        "description": f"Generate an image related to {query}",
                        "agent": agent_name
                    })
                    subtask_id += 1
                    break

        # Check for quality-related keywords
        if any(keyword in query.lower() for keyword in ["quality", "evaluate", "assess", "review"]):
            for agent_name in self.agents.keys():
                if "quality" in agent_name.lower():
                    subtasks.append({
                        "subtask_id": subtask_id,
                        "description": f"Evaluate the quality of information about {query}",
                        "agent": agent_name
                    })
                    subtask_id += 1
                    break

        # Check for storage-related keywords
        if any(keyword in query.lower() for keyword in ["store", "save", "database", "vector"]):
            for agent_name in self.agents.keys():
                if "vector" in agent_name.lower() or "storage" in agent_name.lower():
                    subtasks.append({
                        "subtask_id": subtask_id,
                        "description": f"Store information about {query}",
                        "agent": agent_name
                    })
                    subtask_id += 1
                    break

        # If no subtasks were created, use the first available agent
        if not subtasks and self.agents:
            first_agent = next(iter(self.agents.keys()))
            subtasks.append({
                "subtask_id": 1,
                "description": f"Process the query: {query}",
                "agent": first_agent
            })

        return subtasks

    def _execute_subtask(self, subtask: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single subtask using the appropriate agent.

        Args:
            subtask: Subtask to execute
            state: Current state

        Returns:
            Dict[str, Any]: Updated state after executing the subtask
        """
        agent_name = subtask["agent"]

        # Check if the agent exists
        if agent_name not in self.agents:
            print(f"Warning: Agent '{agent_name}' not found. Skipping subtask.")
            return state

        # Create a copy of the state for this agent
        agent_state = {
            "messages": state["messages"].copy(),
            "agent_outputs": {},
            "current_subtask": subtask,
            "subtasks": []
        }

        # Execute the agent
        try:
            agent = self.agents[agent_name]
            updated_state = agent(agent_state)

            # Extract the agent's output
            if "agent_outputs" in updated_state:
                state["agent_outputs"][agent_name] = updated_state["agent_outputs"].get(agent_name, {})

            # Add any messages from the agent
            if "messages" in updated_state and len(updated_state["messages"]) > len(state["messages"]):
                for i in range(len(state["messages"]), len(updated_state["messages"])):
                    state["messages"].append(updated_state["messages"][i])
        except Exception as e:
            print(f"Error executing agent '{agent_name}': {str(e)}")
            state["agent_outputs"][agent_name] = {"error": str(e)}

        return state

    def _synthesize_response(self, query: str, state: Dict[str, Any]) -> str:
        """
        Synthesize a final response based on agent outputs.

        Args:
            query: Original user query
            state: Current state with agent outputs

        Returns:
            str: Synthesized response
        """
        # Create a prompt for the LLM to synthesize a response
        prompt = f"""
        System: {self.config.system_message}

        You need to synthesize a final response based on the outputs from various agents.

        Original query: {query}

        Agent outputs:
        {json.dumps(state["agent_outputs"], indent=2)}

        Please synthesize a helpful response that incorporates the information from all agents.
        """

        # Get response from LLM
        response = self.llm.invoke([{"role": "user", "content": prompt}])

        # Extract content from response
        if isinstance(response, dict):
            content = response.get("content", "No content available")
        elif hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        return content

    def invoke(self, query: str, stream: bool = False) -> Dict[str, Any]:
        """
        Process a user query using the multi-agent system.

        Args:
            query: User query
            stream: Whether to stream the response (ignored in this implementation)

        Returns:
            Dict[str, Any]: Final state after processing
        """
        # Initialize state
        state = {
            "messages": [{"role": "user", "content": query}],
            "agent_outputs": {},
            "subtasks": [],
            "final_response": None
        }

        # Plan tasks
        subtasks = self._plan_tasks(query)
        state["subtasks"] = subtasks

        # Execute subtasks in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a future for each subtask
            future_to_subtask = {
                executor.submit(self._execute_subtask, subtask, state): subtask
                for subtask in subtasks
            }

            # Process completed futures as they complete
            for future in concurrent.futures.as_completed(future_to_subtask):
                subtask = future_to_subtask[future]
                try:
                    # Get the result (this will re-raise any exception that occurred)
                    updated_state = future.result()

                    # Merge the updated state with the main state
                    for agent_name, output in updated_state["agent_outputs"].items():
                        state["agent_outputs"][agent_name] = output
                except Exception as e:
                    print(f"Error executing subtask {subtask['subtask_id']}: {str(e)}")
                    agent_name = subtask["agent"]
                    state["agent_outputs"][agent_name] = {"error": str(e)}

        # Synthesize final response
        final_response = self._synthesize_response(query, state)
        state["final_response"] = final_response
        state["messages"].append({"role": "assistant", "content": final_response})

        return state


# Example usage
if __name__ == "__main__":
    # This is just a placeholder for testing
    def search_agent(state):
        # Simulate search agent
        subtask = state["current_subtask"]
        state["agent_outputs"]["search_agent"] = {"results": [f"Search result for: {subtask['description']}"]}
        return state

    def image_generation_agent(state):
        # Simulate image generation agent
        subtask = state["current_subtask"]
        state["agent_outputs"]["image_generation_agent"] = {"url": "https://example.com/image.jpg"}
        return state

    def quality_agent(state):
        # Simulate quality agent
        subtask = state["current_subtask"]
        state["agent_outputs"]["quality_agent"] = {"quality_score": 0.85}
        return state

    def vector_storage_agent(state):
        # Simulate vector storage agent
        subtask = state["current_subtask"]
        state["agent_outputs"]["vector_storage_agent"] = {"stored": True}
        return state

    # Create supervisor with mock agents
    supervisor = ParallelSupervisor(
        config=ParallelSupervisorConfig(),
        agents={
            "search_agent": search_agent,
            "image_generation_agent": image_generation_agent,
            "quality_agent": quality_agent,
            "vector_storage_agent": vector_storage_agent
        }
    )

    # Test with a complex query
    result = supervisor.invoke("Search for information about climate change, then generate an image of a sustainable city, and finally evaluate the quality of the information")
    print(result["final_response"])
