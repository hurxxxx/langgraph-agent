"""
Parallel Supervisor for Multi-Agent System

This module implements a supervisor agent that can orchestrate multiple specialized agents
in parallel using a simpler approach without complex graph structures.

Includes LangSmith tracing for monitoring and debugging.
"""

import os
import json
import time
import concurrent.futures
from typing import Dict, List, Any, Optional, Literal, AsyncGenerator
from pydantic import BaseModel

# Import LangSmith utilities
from src.utils.langsmith_utils import tracer

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage


class ParallelSupervisorConfig(BaseModel):
    """Configuration for the parallel supervisor agent."""
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    anthropic_reasoning_model: str = "claude-3-7-haiku-20250201"
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
        Plan tasks based on the user query, identifying which tasks can be run in parallel.

        Args:
            query: User query

        Returns:
            List[Dict[str, Any]]: List of planned subtasks with parallelization information
        """
        # Create a prompt for the LLM to break down the task and identify parallelizable tasks
        prompt = f"""
        System: {self.config.system_message}

        Available agents: {', '.join(self.agents.keys())}

        User query: {query}

        Break down this query into subtasks. For each subtask, specify which agent should handle it.
        Also, analyze which tasks can be executed in parallel and which must be sequential.

        Guidelines for parallelization:
        1. Web search and vector search can be run in parallel as they are independent operations
        2. Multiple search topics within a single query can be processed in parallel
        3. Tasks that depend on the output of previous tasks must be sequential
        4. Image generation can typically run in parallel with search operations
        5. Quality evaluation usually needs to happen after information is gathered

        Format your response as a JSON array of objects, each with:
        - 'subtask_id': A unique identifier for the subtask
        - 'description': A description of what the subtask should do
        - 'agent': Which agent should handle this subtask
        - 'depends_on': Array of subtask_ids this task depends on (empty array if no dependencies)
        - 'parallelizable': Boolean indicating if this task can be run in parallel with other tasks
        - 'parallel_group': Optional group identifier for tasks that can be run in parallel together

        Example:
        [
            {{"subtask_id": 1, "description": "Search for information about climate change", "agent": "search_agent", "depends_on": [], "parallelizable": true, "parallel_group": "searches"}},
            {{"subtask_id": 2, "description": "Search for information about renewable energy", "agent": "search_agent", "depends_on": [], "parallelizable": true, "parallel_group": "searches"}},
            {{"subtask_id": 3, "description": "Generate an image of a sustainable city", "agent": "image_generation_agent", "depends_on": [], "parallelizable": true}},
            {{"subtask_id": 4, "description": "Evaluate the quality of the search results", "agent": "quality_agent", "depends_on": [1, 2], "parallelizable": false}}
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
        Create fallback subtasks based on keywords in the query, with parallelization information.

        Args:
            query: User query

        Returns:
            List[Dict[str, Any]]: List of subtasks with parallelization information
        """
        subtasks = []
        subtask_id = 1
        search_subtasks = []

        # Parse the query to identify multiple search topics
        search_topics = self._identify_search_topics(query)

        # If we have multiple search topics, create a subtask for each
        if search_topics and len(search_topics) > 1:
            for topic in search_topics:
                for agent_name in self.agents.keys():
                    if "search" in agent_name.lower():
                        search_subtasks.append({
                            "subtask_id": subtask_id,
                            "description": f"Search for information about {topic}",
                            "agent": agent_name,
                            "depends_on": [],
                            "parallelizable": True,
                            "parallel_group": "searches"
                        })
                        subtask_id += 1
                        break

            # Add all search subtasks to the main subtasks list
            subtasks.extend(search_subtasks)
        else:
            # Check for search-related keywords
            if any(keyword in query.lower() for keyword in ["search", "find", "information", "what is", "tell me about"]):
                for agent_name in self.agents.keys():
                    if "search" in agent_name.lower():
                        subtasks.append({
                            "subtask_id": subtask_id,
                            "description": f"Search for information about {query}",
                            "agent": agent_name,
                            "depends_on": [],
                            "parallelizable": True,
                            "parallel_group": "searches"
                        })
                        subtask_id += 1
                        break

        # Check for vector search keywords
        if any(keyword in query.lower() for keyword in ["vector", "semantic", "similar"]):
            for agent_name in self.agents.keys():
                if "vector" in agent_name.lower() and "retrieval" in agent_name.lower():
                    subtasks.append({
                        "subtask_id": subtask_id,
                        "description": f"Perform vector search for {query}",
                        "agent": agent_name,
                        "depends_on": [],
                        "parallelizable": True,
                        "parallel_group": "searches"
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
                        "agent": agent_name,
                        "depends_on": [],
                        "parallelizable": True
                    })
                    subtask_id += 1
                    break

        # Check for quality-related keywords
        if any(keyword in query.lower() for keyword in ["quality", "evaluate", "assess", "review"]):
            # Quality evaluation depends on search results
            search_ids = [task["subtask_id"] for task in subtasks if "search" in task["agent"].lower()]

            for agent_name in self.agents.keys():
                if "quality" in agent_name.lower():
                    subtasks.append({
                        "subtask_id": subtask_id,
                        "description": f"Evaluate the quality of information about {query}",
                        "agent": agent_name,
                        "depends_on": search_ids,
                        "parallelizable": False
                    })
                    subtask_id += 1
                    break

        # Check for storage-related keywords
        if any(keyword in query.lower() for keyword in ["store", "save", "database"]):
            # Storage depends on search results
            search_ids = [task["subtask_id"] for task in subtasks if "search" in task["agent"].lower()]

            for agent_name in self.agents.keys():
                if "vector" in agent_name.lower() and "storage" in agent_name.lower():
                    subtasks.append({
                        "subtask_id": subtask_id,
                        "description": f"Store information about {query}",
                        "agent": agent_name,
                        "depends_on": search_ids,
                        "parallelizable": False
                    })
                    subtask_id += 1
                    break

        # If no subtasks were created, use the first available agent
        if not subtasks and self.agents:
            first_agent = next(iter(self.agents.keys()))
            subtasks.append({
                "subtask_id": 1,
                "description": f"Process the query: {query}",
                "agent": first_agent,
                "depends_on": [],
                "parallelizable": False
            })

        return subtasks

    def _identify_search_topics(self, query: str) -> List[str]:
        """
        Identify multiple search topics in a query.

        Args:
            query: User query

        Returns:
            List[str]: List of identified search topics
        """
        # Use the LLM to identify multiple search topics
        prompt = f"""
        Analyze the following query and identify distinct search topics that could be researched in parallel.
        If there are multiple distinct topics, list them one per line. If there's only one main topic, just return that.

        Query: {query}

        Format your response as a simple list, one topic per line, with no numbering or bullets.
        """

        try:
            # Get response from LLM
            response = self.llm.invoke([{"role": "user", "content": prompt}])

            # Extract content from response
            if isinstance(response, dict):
                content = response.get("content", "")
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Split into lines and clean up
            topics = [topic.strip() for topic in content.strip().split('\n') if topic.strip()]

            # If we have only one topic that's very similar to the original query,
            # it might not be a multi-topic query
            if len(topics) == 1 and self._text_similarity(topics[0], query) > 0.8:
                return []

            return topics if len(topics) > 1 else []

        except Exception as e:
            print(f"Error identifying search topics: {str(e)}")
            return []

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Similarity score between 0 and 1
        """
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0

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

        # Execute the agent with LangSmith tracing
        try:
            # Get the agent function
            agent = self.agents[agent_name]

            # Wrap the agent with tracing if it's not already wrapped
            if not hasattr(agent, "__wrapped__"):
                agent = tracer.trace_agent(agent_name)(agent)

            # Execute the agent
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
        Synthesize a final response based on agent outputs, including parallelization information.

        Args:
            query: Original user query
            state: Current state with agent outputs and execution stats

        Returns:
            str: Synthesized response
        """
        # Extract execution statistics if available
        execution_stats = state.get("execution_stats", {})
        parallel_batches = execution_stats.get("parallel_batches", 0)
        total_execution_time = execution_stats.get("total_execution_time", 0)

        # Create a prompt for the LLM to synthesize a response
        prompt = f"""
        System: {self.config.system_message}

        You need to synthesize a final response based on the outputs from various agents.

        Original query: {query}

        Agent outputs:
        {json.dumps(state["agent_outputs"], indent=2)}

        Execution information:
        - Tasks were executed in {parallel_batches} parallel batches
        - Total execution time: {total_execution_time:.2f} seconds
        - Subtasks: {json.dumps(state["subtasks"], indent=2)}

        Please synthesize a helpful response that incorporates the information from all agents.
        Do NOT mention the execution details or parallel processing in your response unless specifically asked about system performance.
        Focus on providing a coherent answer to the user's query.
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

    @tracer.trace_supervisor("ParallelSupervisor")
    def invoke(self, query: str, stream: bool = False) -> Dict[str, Any]:
        """
        Process a user query using the multi-agent system with intelligent parallelization.

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
            "completed_subtasks": set(),
            "final_response": None
        }

        # Plan tasks
        subtasks = self._plan_tasks(query)
        state["subtasks"] = subtasks

        # Create a dependency graph
        dependency_graph = self._create_dependency_graph(subtasks)

        # Group subtasks by parallel groups
        parallel_groups = self._group_subtasks_by_parallel_group(subtasks)

        # Track execution metrics
        execution_start_time = time.time()
        execution_stats = {
            "total_subtasks": len(subtasks),
            "completed_subtasks": 0,
            "parallel_batches": 0,
            "execution_times": {}
        }

        # Execute subtasks in waves based on dependencies
        while len(state["completed_subtasks"]) < len(subtasks):
            # Find all subtasks that can be executed now (all dependencies satisfied)
            executable_subtasks = self._get_executable_subtasks(subtasks, state["completed_subtasks"], dependency_graph)

            if not executable_subtasks:
                # If no subtasks can be executed but we haven't completed all tasks,
                # there might be a circular dependency or other issue
                print("Warning: No executable subtasks found but not all subtasks completed.")
                break

            execution_stats["parallel_batches"] += 1
            batch_start_time = time.time()

            # Execute this batch of subtasks in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create a future for each executable subtask
                future_to_subtask = {
                    executor.submit(self._execute_subtask, subtask, state): subtask
                    for subtask in executable_subtasks
                }

                # Process completed futures as they complete
                for future in concurrent.futures.as_completed(future_to_subtask):
                    subtask = future_to_subtask[future]
                    subtask_id = subtask["subtask_id"]
                    subtask_start_time = time.time()

                    try:
                        # Get the result (this will re-raise any exception that occurred)
                        updated_state = future.result()

                        # Merge the updated state with the main state
                        for agent_name, output in updated_state["agent_outputs"].items():
                            state["agent_outputs"][agent_name] = output

                        # Mark this subtask as completed
                        state["completed_subtasks"].add(subtask_id)
                        execution_stats["completed_subtasks"] += 1

                    except Exception as e:
                        print(f"Error executing subtask {subtask_id}: {str(e)}")
                        agent_name = subtask["agent"]
                        state["agent_outputs"][agent_name] = {"error": str(e)}

                        # Even if it failed, mark it as completed to avoid deadlock
                        state["completed_subtasks"].add(subtask_id)
                        execution_stats["completed_subtasks"] += 1

                    # Record execution time for this subtask
                    execution_stats["execution_times"][subtask_id] = time.time() - subtask_start_time

            # Record batch execution time
            batch_execution_time = time.time() - batch_start_time
            print(f"Batch {execution_stats['parallel_batches']} executed {len(executable_subtasks)} subtasks in {batch_execution_time:.2f} seconds")

        # Record total execution time
        execution_stats["total_execution_time"] = time.time() - execution_start_time
        print(f"Total execution time: {execution_stats['total_execution_time']:.2f} seconds")
        print(f"Executed {execution_stats['completed_subtasks']} subtasks in {execution_stats['parallel_batches']} parallel batches")

        # Add execution stats to state
        state["execution_stats"] = execution_stats

        # Synthesize final response
        final_response = self._synthesize_response(query, state)
        state["final_response"] = final_response
        state["messages"].append({"role": "assistant", "content": final_response})

        return state

    async def astream(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a user query using the multi-agent system with intelligent parallelization and streaming.

        Args:
            query: User query

        Yields:
            Dict[str, Any]: State updates during processing
        """
        import asyncio

        # Initialize state
        state = {
            "messages": [{"role": "user", "content": query}],
            "agent_outputs": {},
            "subtasks": [],
            "completed_subtasks": set(),
            "final_response": None,
            "stream": True
        }

        # Plan tasks
        subtasks = self._plan_tasks(query)
        state["subtasks"] = subtasks

        # Yield initial state with planned subtasks
        yield state.copy()

        # Create a dependency graph
        dependency_graph = self._create_dependency_graph(subtasks)

        # Group subtasks by parallel groups
        parallel_groups = self._group_subtasks_by_parallel_group(subtasks)

        # Track execution metrics
        execution_start_time = time.time()
        execution_stats = {
            "total_subtasks": len(subtasks),
            "completed_subtasks": 0,
            "parallel_batches": 0,
            "execution_times": {}
        }

        # Update state with execution stats
        state["execution_stats"] = execution_stats
        yield state.copy()

        # Execute subtasks in waves based on dependencies
        while len(state["completed_subtasks"]) < len(subtasks):
            # Find all subtasks that can be executed now (all dependencies satisfied)
            executable_subtasks = self._get_executable_subtasks(subtasks, state["completed_subtasks"], dependency_graph)

            if not executable_subtasks:
                # If no subtasks can be executed but we haven't completed all tasks,
                # there might be a circular dependency or other issue
                print("Warning: No executable subtasks found but not all subtasks completed.")
                state["warning"] = "No executable subtasks found but not all subtasks completed."
                yield state.copy()
                break

            execution_stats["parallel_batches"] += 1
            batch_start_time = time.time()

            # Update state with batch information
            state["current_batch"] = execution_stats["parallel_batches"]
            state["current_subtasks"] = [subtask["description"] for subtask in executable_subtasks]
            yield state.copy()

            # Execute this batch of subtasks
            # Note: We can't use ThreadPoolExecutor with async/await directly,
            # so we'll execute subtasks sequentially for streaming
            for subtask in executable_subtasks:
                subtask_id = subtask["subtask_id"]
                agent_name = subtask["agent"]

                # Update state with current subtask
                state["current_subtask"] = subtask
                state["current_agent"] = agent_name
                yield state.copy()

                subtask_start_time = time.time()

                try:
                    # Execute the subtask
                    agent = self.agents[agent_name]

                    # Check if agent supports streaming
                    if hasattr(agent, 'astream') and callable(getattr(agent, 'astream')):
                        # Create a copy of the state for this agent
                        agent_state = {
                            "messages": state["messages"].copy(),
                            "agent_outputs": {},
                            "current_subtask": subtask,
                            "subtasks": [],
                            "stream": True
                        }

                        # Stream the agent's response
                        async for chunk in agent.astream(agent_state):
                            # Update the main state with the agent's output
                            if "agent_outputs" in chunk and agent_name in chunk["agent_outputs"]:
                                state["agent_outputs"][agent_name] = chunk["agent_outputs"][agent_name]

                            # Update the current chunk information
                            chunk["current_subtask"] = subtask
                            chunk["current_agent"] = agent_name

                            # Yield the updated chunk
                            yield chunk
                    else:
                        # Fall back to non-streaming invoke
                        updated_state = self._execute_subtask(subtask, state)

                        # Merge the updated state with the main state
                        for name, output in updated_state["agent_outputs"].items():
                            state["agent_outputs"][name] = output

                        # Yield the updated state
                        yield state.copy()

                    # Mark this subtask as completed
                    state["completed_subtasks"].add(subtask_id)
                    execution_stats["completed_subtasks"] += 1

                except Exception as e:
                    print(f"Error executing subtask {subtask_id}: {str(e)}")
                    state["agent_outputs"][agent_name] = {"error": str(e)}
                    state["error"] = f"Error executing subtask {subtask_id}: {str(e)}"
                    yield state.copy()

                    # Even if it failed, mark it as completed to avoid deadlock
                    state["completed_subtasks"].add(subtask_id)
                    execution_stats["completed_subtasks"] += 1

                # Record execution time for this subtask
                execution_stats["execution_times"][subtask_id] = time.time() - subtask_start_time

                # Update execution stats in state
                state["execution_stats"] = execution_stats
                yield state.copy()

                # Small delay to allow for better streaming visualization
                await asyncio.sleep(0.1)

            # Record batch execution time
            batch_execution_time = time.time() - batch_start_time
            print(f"Batch {execution_stats['parallel_batches']} executed {len(executable_subtasks)} subtasks in {batch_execution_time:.2f} seconds")

            # Update state with batch completion
            state["batch_complete"] = True
            state["batch_execution_time"] = batch_execution_time
            yield state.copy()

        # Record total execution time
        execution_stats["total_execution_time"] = time.time() - execution_start_time
        print(f"Total execution time: {execution_stats['total_execution_time']:.2f} seconds")
        print(f"Executed {execution_stats['completed_subtasks']} subtasks in {execution_stats['parallel_batches']} parallel batches")

        # Update execution stats in state
        state["execution_stats"] = execution_stats
        state["execution_complete"] = True
        yield state.copy()

        # Synthesize final response
        final_response = self._synthesize_response(query, state)
        state["final_response"] = final_response
        state["messages"].append({"role": "assistant", "content": final_response})

        # Yield final state
        yield state

    def _create_dependency_graph(self, subtasks: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """
        Create a dependency graph from the subtasks.

        Args:
            subtasks: List of subtasks

        Returns:
            Dict[int, List[int]]: Dictionary mapping subtask IDs to lists of dependent subtask IDs
        """
        # Create a graph where keys are subtask IDs and values are lists of subtasks that depend on this subtask
        graph = {subtask["subtask_id"]: [] for subtask in subtasks}

        # Populate the graph
        for subtask in subtasks:
            for dependency_id in subtask.get("depends_on", []):
                if dependency_id in graph:
                    graph[dependency_id].append(subtask["subtask_id"])

        return graph

    def _get_executable_subtasks(self, subtasks: List[Dict[str, Any]], completed_subtasks: set, dependency_graph: Dict[int, List[int]]) -> List[Dict[str, Any]]:
        """
        Get subtasks that can be executed now (all dependencies satisfied).

        Args:
            subtasks: List of all subtasks
            completed_subtasks: Set of completed subtask IDs
            dependency_graph: Dependency graph

        Returns:
            List[Dict[str, Any]]: List of executable subtasks
        """
        executable_subtasks = []

        for subtask in subtasks:
            subtask_id = subtask["subtask_id"]

            # Skip if already completed
            if subtask_id in completed_subtasks:
                continue

            # Check if all dependencies are satisfied
            dependencies = subtask.get("depends_on", [])
            if all(dep_id in completed_subtasks for dep_id in dependencies):
                executable_subtasks.append(subtask)

        return executable_subtasks

    def _group_subtasks_by_parallel_group(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group subtasks by their parallel_group attribute.

        Args:
            subtasks: List of subtasks

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary mapping group names to lists of subtasks
        """
        groups = {}

        for subtask in subtasks:
            group = subtask.get("parallel_group", None)
            if group:
                if group not in groups:
                    groups[group] = []
                groups[group].append(subtask)

        return groups


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
