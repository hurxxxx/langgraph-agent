"""
Search Agent for Multi-Agent System

This module implements a search agent that can retrieve information from various search providers:
- Tavily
- Serper (Google Search API)
- Google Search
- DuckDuckGo
- Other search APIs

The agent can be configured to use different search providers and supports streaming responses.
"""

import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal, Union
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Search providers
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import GoogleSearchAPIWrapper, GoogleSerperAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool


class SearchResult(BaseModel):
    """Model for a search result."""
    title: str
    url: str
    snippet: str


class SearchAgentConfig(BaseModel):
    """Configuration for the search agent."""
    provider: Literal["tavily", "serper", "google", "duckduckgo"] = "tavily"
    llm_model: str = "gpt-4o"
    temperature: float = 0
    streaming: bool = True
    max_results: int = 5
    system_message: str = """
    You are a search agent that retrieves information from the web.
    Your job is to:
    1. Understand the search query
    2. Retrieve relevant information from the web
    3. Summarize the results in a clear and concise way

    Always cite your sources with URLs.
    """


class SearchAgent:
    """
    Search agent that retrieves information from various search providers.
    """

    def __init__(self, config: SearchAgentConfig = SearchAgentConfig()):
        """
        Initialize the search agent.

        Args:
            config: Configuration for the search agent
        """
        self.config = config

        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=config.llm_model,
                temperature=config.temperature,
                streaming=config.streaming
            )
        except Exception as e:
            print(f"Warning: Could not initialize ChatOpenAI: {str(e)}")
            # Use a mock implementation
            class MockLLM:
                def invoke(self, messages):
                    # Handle different message formats
                    if isinstance(messages, list):
                        if messages and isinstance(messages[-1], dict):
                            query = messages[-1].get("content", "unknown query")
                        elif messages and hasattr(messages[-1], "content"):
                            query = messages[-1].content
                        else:
                            query = str(messages[-1])
                    else:
                        query = str(messages)

                    return {"content": f"Here are the search results for: {query}"}
            self.llm = MockLLM()

        # Initialize search tool based on provider
        self.search_tool = self._initialize_search_tool()

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=config.system_message),
            MessagesPlaceholder(variable_name="messages"),
            SystemMessage(content="Search results: {search_results}")
        ])

    def _initialize_search_tool(self) -> Any:
        """
        Initialize the appropriate search tool based on the configured provider.

        Returns:
            Any: The initialized search tool
        """
        try:
            if self.config.provider == "tavily":
                return TavilySearchResults(
                    api_key=os.getenv("TAVILY_API_KEY"),
                    max_results=self.config.max_results
                )
            elif self.config.provider == "serper":
                search = GoogleSerperAPIWrapper(
                    serper_api_key=os.getenv("SERPER_API_KEY"),
                    k=self.config.max_results
                )
                return Tool(
                    name="Serper Search",
                    description="Search Google using Serper API for recent results.",
                    func=search.run
                )
            elif self.config.provider == "google":
                search = GoogleSearchAPIWrapper()
                return Tool(
                    name="Google Search",
                    description="Search Google for recent results.",
                    func=search.run
                )
            elif self.config.provider == "duckduckgo":
                return DuckDuckGoSearchRun()
            else:
                raise ValueError(f"Unsupported search provider: {self.config.provider}")
        except Exception as e:
            print(f"Warning: Could not initialize search tool: {str(e)}")
            # Use a mock implementation
            class MockSearchTool:
                def invoke(self, query):
                    return [f"Mock search result for: {query}"]
            return MockSearchTool()

    def _format_search_results(self, results: List[Any]) -> List[SearchResult]:
        """
        Format search results into a standardized format.

        Args:
            results: Raw search results from the provider

        Returns:
            List[SearchResult]: Formatted search results
        """
        formatted_results = []

        for result in results:
            # Handle string results (common with some providers)
            if isinstance(result, str):
                formatted_results.append(SearchResult(
                    title="Search Result",
                    url="No direct URL available",
                    snippet=result
                ))
                continue

            # Handle dictionary results
            if isinstance(result, dict):
                if self.config.provider == "tavily":
                    formatted_results.append(SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("url", "No URL"),
                        snippet=result.get("content", "No content")
                    ))
                elif self.config.provider == "serper":
                    formatted_results.append(SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("link", "No URL"),
                        snippet=result.get("snippet", "No snippet")
                    ))
                elif self.config.provider == "google":
                    formatted_results.append(SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("link", "No URL"),
                        snippet=result.get("snippet", "No snippet")
                    ))
                else:
                    # Generic dictionary handling
                    formatted_results.append(SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("url", result.get("link", "No URL")),
                        snippet=result.get("snippet", result.get("content", str(result)))
                    ))
            else:
                # Handle other types
                formatted_results.append(SearchResult(
                    title="Search Result",
                    url="No direct URL available",
                    snippet=str(result)
                ))

        return formatted_results

    def search(self, query: str) -> List[SearchResult]:
        """
        Perform a search using the configured provider.

        Args:
            query: Search query

        Returns:
            List[SearchResult]: Search results
        """
        raw_results = self.search_tool.invoke(query)

        # Handle different return types from different providers
        if not isinstance(raw_results, list):
            if self.config.provider == "serper":
                # Serper returns a dictionary with organic results
                if isinstance(raw_results, dict) and "organic" in raw_results:
                    raw_results = raw_results["organic"]
                else:
                    raw_results = [raw_results]
            else:
                raw_results = [raw_results]

        return self._format_search_results(raw_results)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state update in the multi-agent system.

        Args:
            state: Current state of the system

        Returns:
            Dict[str, Any]: Updated state
        """
        try:
            # Extract the query from the last message
            query = state["messages"][-1]["content"]

            # Perform search
            search_results = self.search(query)

            # Format results for display
            formatted_results = "\n\n".join([
                f"Title: {result.title}\nURL: {result.url}\nSnippet: {result.snippet}"
                for result in search_results
            ])

            # Generate response using LLM
            response = self.llm.invoke(
                self.prompt.format(
                    messages=state["messages"],
                    search_results=formatted_results
                )
            )

            # Extract content from response
            if isinstance(response, dict):
                content = response.get("content", "No search results available")
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Update state
            state["agent_outputs"]["search"] = {
                "results": [
                    result.model_dump() if hasattr(result, "model_dump")
                    else vars(result) if hasattr(result, "__dict__")
                    else {"title": str(result), "url": "", "snippet": str(result)}
                    for result in search_results
                ]
            }
            state["messages"].append({"role": "assistant", "content": content})
        except Exception as e:
            # Handle errors gracefully
            error_message = f"Search agent encountered an error: {str(e)}"
            print(error_message)

            # Update state with error information
            state["agent_outputs"]["search"] = {
                "error": str(e),
                "results": []
            }

            # Add error response to messages
            state["messages"].append({
                "role": "assistant",
                "content": f"I apologize, but I encountered an error while searching: {str(e)}"
            })

        return state


# Example usage
if __name__ == "__main__":
    # Set up environment variables
    if not os.getenv("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

    if not os.getenv("SERPER_API_KEY"):
        os.environ["SERPER_API_KEY"] = "your-serper-api-key"

    # Create search agent
    search_agent = SearchAgent(
        config=SearchAgentConfig(
            provider="serper",  # Use Serper as the default provider
            max_results=5
        )
    )

    # Test with a query
    state = {
        "messages": [{"role": "user", "content": "What is the latest news about AI?"}],
        "agent_outputs": {}
    }

    updated_state = search_agent(state)
    print(updated_state["messages"][-1]["content"])
