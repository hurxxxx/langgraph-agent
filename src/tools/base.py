"""
Base Tool Interface

This module defines the base interface for all tools in the LangGraph Agent system.
Tools are specific capabilities that perform discrete tasks and are used by agents
to accomplish their goals.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class BaseToolConfig(BaseModel):
    """Base configuration for all tools."""
    
    name: str = Field(
        ...,
        description="The name of the tool"
    )
    description: str = Field(
        ...,
        description="A description of what the tool does"
    )


class BaseTool(ABC):
    """Base class for all tools."""
    
    def __init__(self, config: Optional[BaseToolConfig] = None):
        """
        Initialize the tool.
        
        Args:
            config: Configuration for the tool
        """
        self.config = config or self._get_default_config()
    
    @abstractmethod
    def _get_default_config(self) -> BaseToolConfig:
        """
        Get the default configuration for the tool.
        
        Returns:
            BaseToolConfig: Default configuration
        """
        pass
    
    @abstractmethod
    def run(self, input_text: str) -> str:
        """
        Run the tool with the given input.
        
        Args:
            input_text: Input text for the tool
            
        Returns:
            str: Output from the tool
        """
        pass
    
    def to_langchain_tool(self) -> Any:
        """
        Convert the tool to a LangChain Tool.
        
        Returns:
            Tool: LangChain Tool
        """
        from langchain.tools import Tool
        
        return Tool(
            name=self.config.name,
            description=self.config.description,
            func=self.run
        )


class StructuredTool(BaseTool):
    """Base class for tools with structured input and output."""
    
    @abstractmethod
    def run_structured(self, **kwargs) -> Any:
        """
        Run the tool with structured input.
        
        Args:
            **kwargs: Keyword arguments for the tool
            
        Returns:
            Any: Output from the tool
        """
        pass
    
    def run(self, input_text: str) -> str:
        """
        Run the tool with the given input text.
        
        This method parses the input text into structured arguments and calls run_structured.
        
        Args:
            input_text: Input text for the tool
            
        Returns:
            str: Output from the tool
        """
        import json
        
        try:
            # Try to parse the input as JSON
            args = json.loads(input_text)
            result = self.run_structured(**args)
            
            # Convert the result to a string
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            # If the input is not valid JSON, try to parse it as a simple string
            return self.run_structured(input=input_text)
        except Exception as e:
            # Handle any other errors
            return f"Error running tool: {str(e)}"
    
    def to_langchain_tool(self) -> Any:
        """
        Convert the tool to a LangChain Tool.
        
        Returns:
            Tool: LangChain Tool
        """
        from langchain.tools import StructuredTool as LangChainStructuredTool
        
        # Get the signature of the run_structured method
        import inspect
        signature = inspect.signature(self.run_structured)
        
        # Create the LangChain StructuredTool
        return LangChainStructuredTool.from_function(
            func=self.run_structured,
            name=self.config.name,
            description=self.config.description
        )


class AsyncTool(BaseTool):
    """Base class for asynchronous tools."""
    
    @abstractmethod
    async def arun(self, input_text: str) -> str:
        """
        Run the tool asynchronously with the given input.
        
        Args:
            input_text: Input text for the tool
            
        Returns:
            str: Output from the tool
        """
        pass
    
    def run(self, input_text: str) -> str:
        """
        Run the tool synchronously with the given input.
        
        This method creates an event loop and runs the arun method.
        
        Args:
            input_text: Input text for the tool
            
        Returns:
            str: Output from the tool
        """
        import asyncio
        
        try:
            # Create an event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there is no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the arun method
        return loop.run_until_complete(self.arun(input_text))
    
    def to_langchain_tool(self) -> Any:
        """
        Convert the tool to a LangChain Tool.
        
        Returns:
            Tool: LangChain Tool
        """
        from langchain.tools import Tool
        
        return Tool(
            name=self.config.name,
            description=self.config.description,
            func=self.run,
            coroutine=self.arun
        )


class StreamingTool(BaseTool):
    """Base class for streaming tools."""
    
    @abstractmethod
    def stream(self, input_text: str) -> Any:
        """
        Stream the tool's output with the given input.
        
        Args:
            input_text: Input text for the tool
            
        Yields:
            Any: Output chunks from the tool
        """
        pass
    
    def run(self, input_text: str) -> str:
        """
        Run the tool synchronously with the given input.
        
        This method collects all chunks from the stream method and joins them.
        
        Args:
            input_text: Input text for the tool
            
        Returns:
            str: Output from the tool
        """
        chunks = []
        for chunk in self.stream(input_text):
            chunks.append(chunk)
        
        return "".join(chunks) if all(isinstance(c, str) for c in chunks) else chunks
    
    def to_langchain_tool(self) -> Any:
        """
        Convert the tool to a LangChain Tool.
        
        Returns:
            Tool: LangChain Tool
        """
        from langchain.tools import Tool
        
        return Tool(
            name=self.config.name,
            description=self.config.description,
            func=self.run
        )
