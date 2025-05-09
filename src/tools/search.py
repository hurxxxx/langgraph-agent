"""
Search Tools

This module implements search tools for the LangGraph Agent system.
These tools provide the ability to search the web for information.
"""

import os
import json
from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field

from src.tools.base import BaseToolConfig, BaseTool, StructuredTool


class SerperSearchToolConfig(BaseToolConfig):
    """Configuration for the Serper search tool."""
    
    name: str = "serper_search"
    description: str = "Search Google using Serper API for recent results."
    serper_api_key: Optional[str] = None
    max_results: int = 5
    time_period: Optional[str] = None  # Time period for search (e.g., "1d", "1w", "1m")
    news_only: bool = False  # Whether to search only for news
    region: Optional[str] = None  # Region for search results (e.g., "kr" for Korea)


class SerperSearchTool(StructuredTool):
    """
    Search tool using the Serper API.
    
    This tool searches Google using the Serper API and returns the results.
    """
    
    def __init__(self, config: Optional[SerperSearchToolConfig] = None):
        """
        Initialize the Serper search tool.
        
        Args:
            config: Configuration for the Serper search tool
        """
        super().__init__(config)
        
        # Initialize the Serper API wrapper
        self.serper_wrapper = self._initialize_serper_wrapper()
    
    def _get_default_config(self) -> SerperSearchToolConfig:
        """
        Get the default configuration for the Serper search tool.
        
        Returns:
            SerperSearchToolConfig: Default configuration
        """
        return SerperSearchToolConfig(
            serper_api_key=os.getenv("SERPER_API_KEY")
        )
    
    def _initialize_serper_wrapper(self) -> Any:
        """
        Initialize the Serper API wrapper.
        
        Returns:
            GoogleSerperAPIWrapper: Initialized Serper API wrapper
        """
        from langchain_community.utilities import GoogleSerperAPIWrapper
        
        # Get the API key
        api_key = self.config.serper_api_key or os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError("Serper API key is required")
        
        # Create the wrapper parameters
        params = {
            "serper_api_key": api_key,
            "k": self.config.max_results
        }
        
        # Add time period if specified
        if self.config.time_period:
            if self.config.time_period == "1d":
                params["tbs"] = "qdr:d"  # Past 24 hours
            elif self.config.time_period == "1w":
                params["tbs"] = "qdr:w"  # Past week
            elif self.config.time_period == "1m":
                params["tbs"] = "qdr:m"  # Past month
        
        # Set to news only if specified
        if self.config.news_only:
            params["gl"] = self.config.region or "us"
            params["search_type"] = "news"
        elif self.config.region:
            params["gl"] = self.config.region
        
        # Create the wrapper
        return GoogleSerperAPIWrapper(**params)
    
    def run_structured(self, query: str) -> str:
        """
        Search Google using the Serper API.
        
        Args:
            query: Search query
            
        Returns:
            str: Search results
        """
        try:
            # Run the search
            results = self.serper_wrapper.results(query)
            
            # Format the results
            formatted_results = self._format_results(results)
            
            return formatted_results
        except Exception as e:
            return f"Error searching with Serper: {str(e)}"
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """
        Format the search results.
        
        Args:
            results: Search results from the Serper API
            
        Returns:
            str: Formatted search results
        """
        formatted = []
        
        # Add the search information
        if "searchParameters" in results:
            params = results["searchParameters"]
            formatted.append(f"Search: {params.get('q', 'Unknown query')}")
            formatted.append(f"Results: {params.get('num', 'Unknown')} results")
            formatted.append("")
        
        # Process organic results
        if "organic" in results:
            formatted.append("Organic Results:")
            for i, result in enumerate(results["organic"], 1):
                formatted.append(f"{i}. {result.get('title', 'No title')}")
                formatted.append(f"   URL: {result.get('link', 'No link')}")
                if "snippet" in result:
                    formatted.append(f"   Snippet: {result['snippet']}")
                if "date" in result:
                    formatted.append(f"   Date: {result['date']}")
                formatted.append("")
        
        # Process news results
        if "news" in results:
            formatted.append("News Results:")
            for i, result in enumerate(results["news"], 1):
                formatted.append(f"{i}. {result.get('title', 'No title')}")
                formatted.append(f"   URL: {result.get('link', 'No link')}")
                if "snippet" in result:
                    formatted.append(f"   Snippet: {result['snippet']}")
                if "date" in result:
                    formatted.append(f"   Date: {result['date']}")
                if "source" in result:
                    formatted.append(f"   Source: {result['source']}")
                formatted.append("")
        
        # Process knowledge graph
        if "knowledgeGraph" in results:
            kg = results["knowledgeGraph"]
            formatted.append("Knowledge Graph:")
            formatted.append(f"Title: {kg.get('title', 'No title')}")
            if "description" in kg:
                formatted.append(f"Description: {kg['description']}")
            if "attributes" in kg:
                formatted.append("Attributes:")
                for key, value in kg["attributes"].items():
                    formatted.append(f"  {key}: {value}")
            formatted.append("")
        
        # Join the formatted results
        return "\n".join(formatted)


class TavilySearchToolConfig(BaseToolConfig):
    """Configuration for the Tavily search tool."""
    
    name: str = "tavily_search"
    description: str = "Search the web using Tavily API for comprehensive results."
    tavily_api_key: Optional[str] = None
    max_results: int = 5
    search_depth: Literal["basic", "advanced"] = "basic"
    topic: Literal["general", "news"] = "general"


class TavilySearchTool(StructuredTool):
    """
    Search tool using the Tavily API.
    
    This tool searches the web using the Tavily API and returns the results.
    """
    
    def __init__(self, config: Optional[TavilySearchToolConfig] = None):
        """
        Initialize the Tavily search tool.
        
        Args:
            config: Configuration for the Tavily search tool
        """
        super().__init__(config)
        
        # Initialize the Tavily API wrapper
        self.tavily_search = self._initialize_tavily_search()
    
    def _get_default_config(self) -> TavilySearchToolConfig:
        """
        Get the default configuration for the Tavily search tool.
        
        Returns:
            TavilySearchToolConfig: Default configuration
        """
        return TavilySearchToolConfig(
            tavily_api_key=os.getenv("TAVILY_API_KEY")
        )
    
    def _initialize_tavily_search(self) -> Any:
        """
        Initialize the Tavily API wrapper.
        
        Returns:
            TavilySearch: Initialized Tavily API wrapper
        """
        from langchain_tavily import TavilySearch
        
        # Get the API key
        api_key = self.config.tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("Tavily API key is required")
        
        # Create the wrapper parameters
        params = {
            "api_key": api_key,
            "max_results": self.config.max_results,
            "search_depth": self.config.search_depth,
            "topic": self.config.topic
        }
        
        # Create the wrapper
        return TavilySearch(**params)
    
    def run_structured(self, query: str) -> str:
        """
        Search the web using the Tavily API.
        
        Args:
            query: Search query
            
        Returns:
            str: Search results
        """
        try:
            # Run the search
            results = self.tavily_search.invoke(query)
            
            # Format the results
            formatted_results = self._format_results(results)
            
            return formatted_results
        except Exception as e:
            return f"Error searching with Tavily: {str(e)}"
    
    def _format_results(self, results: Dict[str, Any]) -> str:
        """
        Format the search results.
        
        Args:
            results: Search results from the Tavily API
            
        Returns:
            str: Formatted search results
        """
        formatted = []
        
        # Add the search information
        if "query" in results:
            formatted.append(f"Search: {results['query']}")
            formatted.append("")
        
        # Process results
        if "results" in results:
            formatted.append("Search Results:")
            for i, result in enumerate(results["results"], 1):
                formatted.append(f"{i}. {result.get('title', 'No title')}")
                formatted.append(f"   URL: {result.get('url', 'No URL')}")
                if "content" in result:
                    formatted.append(f"   Content: {result['content']}")
                if "score" in result:
                    formatted.append(f"   Relevance Score: {result['score']}")
                if "published_date" in result:
                    formatted.append(f"   Published Date: {result['published_date']}")
                formatted.append("")
        
        # Join the formatted results
        return "\n".join(formatted)


# Example usage
if __name__ == "__main__":
    # Create Serper search tool
    serper_tool = SerperSearchTool(
        config=SerperSearchToolConfig(
            max_results=5,
            time_period="1d",
            news_only=True
        )
    )
    
    # Search with Serper
    serper_results = serper_tool.run("Latest news about AI")
    print(serper_results)
    print("\n" + "-" * 50 + "\n")
    
    # Create Tavily search tool
    tavily_tool = TavilySearchTool(
        config=TavilySearchToolConfig(
            max_results=5,
            search_depth="advanced",
            topic="news"
        )
    )
    
    # Search with Tavily
    tavily_results = tavily_tool.run("Latest news about AI")
    print(tavily_results)
