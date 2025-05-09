"""
Base Supervisor Interface

This module defines the base interface for supervisors in the LangGraph Agent system.
Supervisors are LangGraph constructs that coordinate agents to solve complex tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage

from langgraph.graph import END, StateGraph


class BaseSupervisorConfig(BaseModel):
    """Base configuration for all supervisors."""
    
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
    
    # Supervisor configuration
    system_message: str = Field(
        default="""
        You are a supervisor agent that coordinates multiple specialized agents to solve complex tasks.
        Your job is to:
        1. Understand the user's request
        2. Determine which specialized agent(s) should handle the request
        3. Coordinate the flow of information between agents
        4. Synthesize a final response for the user

        Always think carefully about which agent(s) would be most appropriate for the task.
        You can use multiple agents in sequence or in parallel if needed.
        """,
        description="System message for the supervisor"
    )


class BaseSupervisor(ABC):
    """Base class for all supervisors."""
    
    def __init__(self, config: Optional[BaseSupervisorConfig] = None, agents: Optional[Dict[str, Any]] = None):
        """
        Initialize the supervisor.
        
        Args:
            config: Configuration for the supervisor
            agents: Dictionary of agent functions keyed by agent name
        """
        self.config = config or self._get_default_config()
        self.agents = agents or {}
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Create supervisor graph
        self.graph = self._create_graph()
    
    @abstractmethod
    def _get_default_config(self) -> BaseSupervisorConfig:
        """
        Get the default configuration for the supervisor.
        
        Returns:
            BaseSupervisorConfig: Default configuration
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
    def _create_graph(self) -> Any:
        """
        Create the supervisor graph.
        
        Returns:
            Any: Compiled graph
        """
        pass
    
    @abstractmethod
    def _get_graph_state_schema(self) -> Any:
        """
        Get the state schema for the supervisor graph.
        
        Returns:
            Any: State schema class
        """
        pass
    
    @abstractmethod
    def invoke(self, query: str, stream: bool = False) -> Dict[str, Any]:
        """
        Process a user query.
        
        Args:
            query: User query
            stream: Whether to stream the response
            
        Returns:
            Dict[str, Any]: Final state after processing
        """
        pass
    
    @abstractmethod
    def stream(self, query: str) -> Any:
        """
        Stream the processing of a user query.
        
        Args:
            query: User query
            
        Yields:
            Dict[str, Any]: Streamed response
        """
        pass
    
    @abstractmethod
    async def astream(self, query: str) -> Any:
        """
        Stream the processing of a user query asynchronously.
        
        Args:
            query: User query
            
        Yields:
            Dict[str, Any]: Streamed response
        """
        pass
