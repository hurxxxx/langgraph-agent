"""
ReAct Search Agent using LangGraph's create_react_agent

This module implements a search agent using LangGraph's create_react_agent function
and various search tools like Tavily, Serper, etc.
"""

import os
from typing import Dict, List, Any, Optional, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Search providers
from langchain_tavily import TavilySearch
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import Tool

# LangGraph components
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph


class ReactSearchAgent:
    """
    Search agent using LangGraph's create_react_agent function.
    Supports multiple search providers and streaming.
    """

    def __init__(self,
                 llm_model: str = "gpt-4o",
                 temperature: float = 0,
                 streaming: bool = True,
                 max_results: int = 5,
                 time_period: Optional[str] = None,
                 news_only: bool = False,
                 region: Optional[str] = None):
        """
        Initialize the ReAct search agent.

        Args:
            llm_model: LLM model to use
            temperature: Temperature for the LLM
            streaming: Whether to enable streaming
            max_results: Maximum number of search results
            time_period: Time period for search (e.g., "1d", "1w", "1m")
            news_only: Whether to search only for news
            region: Region for search results (e.g., "kr" for Korea)
        """
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            streaming=streaming
        )

        # Store configuration
        self.max_results = max_results
        self.time_period = time_period
        self.news_only = news_only
        self.region = region

        # Initialize search tools
        self.search_tools = self._initialize_search_tools()

        # Create ReAct agent with system message
        system_message = self._get_system_message()
        self.agent = create_react_agent(
            self.llm,
            self.search_tools,
            prompt=SystemMessage(content=system_message)
        )

        # Create agent graph
        AgentState = self._get_graph_state_schema()
        self.graph = StateGraph(AgentState)
        self.graph.add_node("agent", self.agent)
        self.graph.set_entry_point("agent")
        self.graph.add_edge("agent", END)
        self.compiled_graph = self.graph.compile()

    def _initialize_search_tools(self) -> List[Any]:
        """
        Initialize search tools based on available API keys.

        Returns:
            List[Any]: List of initialized search tools
        """
        tools = []

        # Initialize Tavily Search if API key is available
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            tavily_params = {
                "max_results": self.max_results,
                "api_key": tavily_api_key
            }

            # Set search type based on configuration
            if self.news_only:
                tavily_params["search_depth"] = "basic"
                tavily_params["topic"] = "news"
            else:
                tavily_params["topic"] = "general"

            # Add time constraints if specified
            if self.time_period:
                # Tavily doesn't directly support time filtering via search_depth
                # We'll use the most appropriate search_depth and add time info to the query
                tavily_params["search_depth"] = "advanced"

            tavily_tool = TavilySearch(**tavily_params)
            tools.append(tavily_tool)

        # Initialize Serper Search if API key is available
        serper_api_key = os.getenv("SERPER_API_KEY")
        if serper_api_key:
            serper_params = {
                "serper_api_key": serper_api_key,
                "k": self.max_results
            }

            # Add time period if specified
            if self.time_period:
                if self.time_period == "1d":
                    serper_params["tbs"] = "qdr:d"  # Past 24 hours
                elif self.time_period == "1w":
                    serper_params["tbs"] = "qdr:w"  # Past week
                elif self.time_period == "1m":
                    serper_params["tbs"] = "qdr:m"  # Past month

            # Set to news only if specified
            if self.news_only:
                serper_params["gl"] = self.region or "us"
                serper_params["search_type"] = "news"
            elif self.region:
                serper_params["gl"] = self.region

            serper_wrapper = GoogleSerperAPIWrapper(**serper_params)
            serper_tool = Tool(
                name="serper_search",
                description="Search Google using Serper API for recent results.",
                func=serper_wrapper.run
            )
            tools.append(serper_tool)

        return tools

    def _get_system_message(self) -> str:
        """
        Get the system message for the ReAct agent.

        Returns:
            str: System message
        """
        system_message = """
        You are a search agent that retrieves information from the web.

        Your job is to:
        1. Analyze the user's query carefully
        2. Use the search tools to find relevant information
        3. Synthesize the information into a comprehensive, accurate response
        4. ALWAYS cite your sources with URLs from the search results
        5. For news-related queries, organize information by topic or category
        6. Include publication dates for news articles when available
        7. Provide detailed information with specific facts, figures, and quotes from the sources
        8. For Korean news, maintain the same level of detail as you would for English news

        If the search results don't contain enough information to fully answer the question, explicitly state what specific information is missing.
        """

        # Add time-specific instructions if time_period is set
        if self.time_period:
            if self.time_period == "1d":
                system_message += "\nFocus on the most recent information from the past 24 hours."
            elif self.time_period == "1w":
                system_message += "\nFocus on recent information from the past week."
            elif self.time_period == "1m":
                system_message += "\nFocus on information from the past month."

        # Add news-specific instructions if news_only is set
        if self.news_only:
            system_message += "\nFocus on news articles and current events. Organize information by topic and include publication dates."

        return system_message

    def _get_graph_state_schema(self):
        """
        Get the state schema for the agent graph.

        Returns:
            TypedDict: State schema class
        """
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph.message import add_messages

        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            agent_outcome: Optional[Dict[str, Any]]

        return AgentState

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
            if "messages" in state and isinstance(state["messages"], list) and state["messages"]:
                if isinstance(state["messages"][-1], dict) and "content" in state["messages"][-1]:
                    query = state["messages"][-1]["content"]
                else:
                    query = str(state["messages"][-1])
            else:
                query = state.get("query", "")

            # Check if we have a subtask in the state (used by MCP)
            if "current_subtask" in state and state["current_subtask"]:
                subtask = state["current_subtask"]
                if "description" in subtask:
                    # Use the subtask description as the query
                    query = subtask["description"]

            # Create input for the agent
            agent_input = {"messages": [{"role": "user", "content": query}]}

            # Run the agent
            result = self.compiled_graph.invoke(agent_input)

            # Update the state with the agent's response
            if "messages" in result and result["messages"]:
                state["messages"] = state.get("messages", [])[:-1] + result["messages"]

            # Store agent outcome in the state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["search_agent"] = {
                "result": result,
                "query": query
            }

            return state

        except Exception as e:
            # Handle errors gracefully
            error_message = f"Search agent encountered an error: {str(e)}"

            # Update state with error information
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["search_agent"] = {
                "error": str(e),
                "has_error": True
            }

            # If this is a subtask in MCP, mark it for potential fallback
            if "current_subtask" in state:
                state["agent_outputs"]["search_agent"]["needs_fallback"] = True

            # Add error response to messages
            if "messages" in state:
                state["messages"].append({
                    "role": "assistant",
                    "content": f"I apologize, but I encountered an error while searching: {str(e)}. Please try again or consider using a different search provider."
                })

            return state

    def stream(self, state: Dict[str, Any], stream_mode: str = "values"):
        """
        Stream the agent's response.

        Args:
            state: Current state of the system
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
                yield chunk

        except Exception as e:
            # Handle errors gracefully
            yield {
                "messages": [
                    {"role": "user", "content": state.get("query", "")},
                    {"role": "assistant", "content": f"I apologize, but I encountered an error while searching: {str(e)}. Please try again or consider using a different search provider."}
                ]
            }
