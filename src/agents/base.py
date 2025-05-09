"""
Base Agent Interface

This module defines the base interface for all agents in the LangGraph Agent system.
Agents are LangGraph constructs that use tools to solve domain-specific problems.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph


class BaseAgentConfig(BaseModel):
    """Base configuration for all agents."""
    
    # LLM configuration
    llm_provider: str = Field(
        default="openai",
        description="LLM provider to use (openai or anthropic)"
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use"
    )
    anthropic_model: str = Field(
        default="claude-3-7-sonnet-20250219",
        description="Anthropic model to use"
    )
    temperature: float = Field(
        default=0,
        description="Temperature for the LLM"
    )
    streaming: bool = Field(
        default=True,
        description="Whether to stream the response"
    )
    
    # Agent configuration
    system_message: str = Field(
        default="You are a helpful assistant.",
        description="System message for the agent"
    )


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, config: Optional[BaseAgentConfig] = None):
        """
        Initialize the agent.
        
        Args:
            config: Configuration for the agent
        """
        self.config = config or self._get_default_config()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Create ReAct agent
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=SystemMessage(content=self._get_system_message())
        )
        
        # Create agent graph
        self.graph = StateGraph(self._get_graph_state_schema())
        self.graph.add_node("agent", self.agent)
        self.graph.set_entry_point("agent")
        self.graph.add_edge("agent", END)
        self.compiled_graph = self.graph.compile()
    
    @abstractmethod
    def _get_default_config(self) -> BaseAgentConfig:
        """
        Get the default configuration for the agent.
        
        Returns:
            BaseAgentConfig: Default configuration
        """
        pass
    
    def _initialize_llm(self) -> Any:
        """
        Initialize the LLM based on the provider.
        
        Returns:
            Any: Initialized LLM
        """
        if self.config.llm_provider == "openai":
            return ChatOpenAI(
                model=self.config.openai_model,
                temperature=self.config.temperature,
                streaming=self.config.streaming
            )
        elif self.config.llm_provider == "anthropic":
            return ChatAnthropic(
                model=self.config.anthropic_model,
                temperature=self.config.temperature,
                streaming=self.config.streaming
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
    @abstractmethod
    def _initialize_tools(self) -> List[Any]:
        """
        Initialize the tools for the agent.
        
        Returns:
            List[Any]: List of initialized tools
        """
        pass
    
    def _get_system_message(self) -> str:
        """
        Get the system message for the agent.
        
        Returns:
            str: System message
        """
        return self.config.system_message
    
    def _get_graph_state_schema(self) -> Any:
        """
        Get the state schema for the agent graph.
        
        Returns:
            Any: State schema class
        """
        from typing import Annotated, Optional
        from typing_extensions import TypedDict
        from langgraph.graph.message import add_messages
        
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            agent_outcome: Optional[Dict[str, Any]]
        
        return AgentState
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state update.
        
        Args:
            state: Current state
            
        Returns:
            Dict[str, Any]: Updated state
        """
        try:
            # Extract the query from the last message
            if "messages" in state and isinstance(state["messages"], list) and state["messages"]:
                if isinstance(state["messages"][-1], dict) and "content" in state["messages"][-1]:
                    query = state["messages"][-1]["content"]
                else:
                    query = str(state["messages"][-1])
            else:
                query = state.get("query", "")
            
            # Create input for the agent
            agent_input = {"messages": [{"role": "user", "content": query}]}
            
            # Run the agent
            result = self.compiled_graph.invoke(agent_input)
            
            # Update the state with the agent's response
            if "messages" in result and result["messages"]:
                # Convert LangChain message objects to dictionaries if needed
                processed_messages = []
                for msg in result["messages"]:
                    if hasattr(msg, "content") and hasattr(msg, "type"):
                        processed_messages.append({
                            "role": "assistant" if msg.type == "ai" else msg.type,
                            "content": msg.content
                        })
                    else:
                        processed_messages.append(msg)
                
                state["messages"] = state.get("messages", [])[:-1] + processed_messages
            
            # Store agent outcome in the state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"][self._get_agent_name()] = {
                "result": result,
                "query": query
            }
            
            return state
        
        except Exception as e:
            # Handle errors gracefully
            error_message = f"Agent encountered an error: {str(e)}"
            
            # Update state with error information
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"][self._get_agent_name()] = {
                "error": str(e),
                "has_error": True
            }
            
            # Add error response to messages
            if "messages" in state:
                state["messages"].append({
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error: {str(e)}. Please try again."
                })
            
            return state
    
    def stream(self, state: Dict[str, Any], stream_mode: str = "values"):
        """
        Stream the agent's response.
        
        Args:
            state: Current state
            stream_mode: Streaming mode ("values" or "steps")
            
        Yields:
            Dict[str, Any]: Streamed response
        """
        try:
            # Extract the query from the last message
            if "messages" in state and isinstance(state["messages"], list) and state["messages"]:
                if isinstance(state["messages"][-1], dict) and "content" in state["messages"][-1]:
                    query = state["messages"][-1]["content"]
                else:
                    query = str(state["messages"][-1])
            else:
                query = state.get("query", "")
            
            # Create input for the agent
            agent_input = {"messages": [{"role": "user", "content": query}]}
            
            # Stream the agent's response
            for chunk in self.compiled_graph.stream(
                agent_input,
                stream_mode=stream_mode
            ):
                # Process the chunk to handle LangChain message objects
                if "messages" in chunk and chunk["messages"]:
                    processed_messages = []
                    for msg in chunk["messages"]:
                        if hasattr(msg, "content") and hasattr(msg, "type"):
                            processed_messages.append({
                                "role": "assistant" if msg.type == "ai" else msg.type,
                                "content": msg.content
                            })
                        else:
                            processed_messages.append(msg)
                    
                    chunk["messages"] = processed_messages
                
                yield chunk
        
        except Exception as e:
            # Handle errors gracefully
            yield {
                "messages": [
                    {"role": "user", "content": state.get("query", "")},
                    {"role": "assistant", "content": f"I apologize, but I encountered an error: {str(e)}. Please try again."}
                ]
            }
    
    async def astream(self, state: Dict[str, Any], stream_mode: str = "values"):
        """
        Stream the agent's response asynchronously.
        
        Args:
            state: Current state
            stream_mode: Streaming mode ("values" or "steps")
            
        Yields:
            Dict[str, Any]: Streamed response
        """
        try:
            # Extract the query from the last message
            if "messages" in state and isinstance(state["messages"], list) and state["messages"]:
                if isinstance(state["messages"][-1], dict) and "content" in state["messages"][-1]:
                    query = state["messages"][-1]["content"]
                else:
                    query = str(state["messages"][-1])
            else:
                query = state.get("query", "")
            
            # Create input for the agent
            agent_input = {"messages": [{"role": "user", "content": query}]}
            
            # Stream the agent's response
            async for chunk in self.compiled_graph.astream(
                agent_input,
                stream_mode=stream_mode
            ):
                # Process the chunk to handle LangChain message objects
                if "messages" in chunk and chunk["messages"]:
                    processed_messages = []
                    for msg in chunk["messages"]:
                        if hasattr(msg, "content") and hasattr(msg, "type"):
                            processed_messages.append({
                                "role": "assistant" if msg.type == "ai" else msg.type,
                                "content": msg.content
                            })
                        else:
                            processed_messages.append(msg)
                    
                    chunk["messages"] = processed_messages
                
                yield chunk
        
        except Exception as e:
            # Handle errors gracefully
            yield {
                "messages": [
                    {"role": "user", "content": state.get("query", "")},
                    {"role": "assistant", "content": f"I apologize, but I encountered an error: {str(e)}. Please try again."}
                ]
            }
    
    def _get_agent_name(self) -> str:
        """
        Get the name of the agent.
        
        Returns:
            str: Agent name
        """
        return self.__class__.__name__.lower()
