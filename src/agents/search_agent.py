"""
Search Agent for Multi-Agent System

This module implements a search agent that can retrieve information from various search providers:
- Tavily
- Serper (Google Search API)
- Google Search
- DuckDuckGo
- Other search APIs

The agent can be configured to use multiple search providers simultaneously, supports parallel search,
and can evaluate search results to determine if additional searches are needed.
"""

import os
import time
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal, Union, Set
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
    provider: Optional[str] = None  # Which provider returned this result


class SearchAgentConfig(BaseModel):
    """Configuration for the search agent."""
    providers: List[Literal["tavily", "serper", "google", "duckduckgo"]] = ["serper"]
    default_provider: Literal["tavily", "serper", "google", "duckduckgo"] = "serper"
    llm_model: str = "gpt-4o"
    temperature: float = 0
    streaming: bool = True
    max_results: int = 5
    parallel_search: bool = True
    evaluate_results: bool = True
    additional_queries: bool = True
    system_message: str = """
    You are a search agent that retrieves information from the web.

    IMPORTANT: I have already performed web searches for the user's query and will provide you with the search results.
    You do NOT need to search the web yourself - I've already done that for you.

    Your job is to:
    1. Carefully analyze the search results I provide
    2. Use ONLY the information from these search results to answer the user's question
    3. If the search results contain recent information from 2025, make sure to include it
    4. Synthesize the information into a comprehensive, accurate response
    5. ALWAYS cite your sources with URLs from the search results

    DO NOT say you cannot browse the web or access real-time information - I've already done the searching for you.
    DO NOT rely on your training data - ONLY use the search results I provide.

    If the search results don't contain enough information to fully answer the question, explicitly state what specific information is missing.
    """
    evaluation_system_message: str = """
    You are an expert at evaluating search results. Your job is to:
    1. Analyze the search results for a query
    2. Determine if the results provide sufficient information to answer the query
    3. If the results are insufficient, suggest additional search queries that would help

    Be specific about what information is missing and what additional queries would help find it.
    Suggest at least 2-3 specific additional search queries that would help find the missing information.
    """


class SearchAgent:
    """
    Search agent that retrieves information from various search providers.
    Supports multiple providers, parallel search, and result evaluation.
    """

    def __init__(self, config: Optional[SearchAgentConfig] = None):
        """
        Initialize the search agent.

        Args:
            config: Configuration for the search agent
        """
        # Load configuration from environment if not provided
        if config is None:
            config = self._load_config_from_env()

        self.config = config

        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=config.llm_model,
                temperature=config.temperature,
                streaming=config.streaming
            )

            # Initialize evaluation LLM (non-streaming for better evaluation)
            self.evaluation_llm = ChatOpenAI(
                model=config.llm_model,
                temperature=0.1,  # Lower temperature for more consistent evaluations
                streaming=False
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
            self.evaluation_llm = MockLLM()

        # Initialize search tools for all providers
        self.search_tools = {}
        for provider in config.providers:
            tool = self._initialize_search_tool(provider)
            if tool:
                self.search_tools[provider] = tool

        # If no tools were initialized, use the default provider
        if not self.search_tools and config.default_provider:
            tool = self._initialize_search_tool(config.default_provider)
            if tool:
                self.search_tools[config.default_provider] = tool

        # Create a simpler prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant that answers questions based on search results."),
            HumanMessage(content="I want to know: {query}"),
            AIMessage(content="I'll help you with that. Let me search for information..."),
            HumanMessage(content="Here are the search results:\n\n{search_results}\n\nBased on ONLY these search results, please answer my question about {query}.")
        ])

        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=config.evaluation_system_message),
            SystemMessage(content="Original query: {query}"),
            SystemMessage(content="Search results: {search_results}"),
            HumanMessage(content="Are these search results sufficient to answer the query? If not, what additional search queries would help?")
        ])

    def _load_config_from_env(self) -> SearchAgentConfig:
        """
        Load configuration from environment variables.

        Returns:
            SearchAgentConfig: Configuration loaded from environment
        """
        # Get providers from environment
        providers_str = os.getenv("SEARCH_PROVIDERS", "serper")
        providers = [p.strip() for p in providers_str.split(",") if p.strip()]

        # Validate providers
        valid_providers = ["tavily", "serper", "google", "duckduckgo"]
        providers = [p for p in providers if p in valid_providers]

        # If no valid providers, use default
        if not providers:
            providers = ["serper"]

        # Get default provider
        default_provider = os.getenv("DEFAULT_SEARCH_PROVIDER", providers[0])
        if default_provider not in valid_providers:
            default_provider = providers[0]

        # Get other settings
        max_results = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
        parallel_search = os.getenv("SEARCH_PARALLEL", "true").lower() == "true"
        additional_queries = os.getenv("SEARCH_ADDITIONAL_QUERIES", "true").lower() == "true"

        return SearchAgentConfig(
            providers=providers,
            default_provider=default_provider,
            max_results=max_results,
            parallel_search=parallel_search,
            additional_queries=additional_queries
        )

    def _initialize_search_tool(self, provider: str) -> Any:
        """
        Initialize a search tool for the specified provider.

        Args:
            provider: The search provider to initialize

        Returns:
            Any: The initialized search tool, or None if initialization failed
        """
        try:
            if provider == "tavily":
                api_key = os.getenv("TAVILY_API_KEY")
                if not api_key:
                    print("Warning: TAVILY_API_KEY not found in environment")
                    return None

                # Create Tavily search tool
                tavily_tool = TavilySearchResults(
                    api_key=api_key,
                    max_results=self.config.max_results
                )
                print(f"Successfully initialized Tavily search tool with API key: {api_key[:5]}...")
                return tavily_tool
            elif provider == "serper":
                api_key = os.getenv("SERPER_API_KEY")
                if not api_key:
                    print("Warning: SERPER_API_KEY not found in environment")
                    return None

                # Create Serper search wrapper
                search = GoogleSerperAPIWrapper(
                    serper_api_key=api_key,
                    k=self.config.max_results
                )

                # Create tool from wrapper
                serper_tool = Tool(
                    name="Serper Search",
                    description="Search Google using Serper API for recent results.",
                    func=search.run
                )

                print(f"Successfully initialized Serper search tool with API key: {api_key[:5]}...")
                return serper_tool
            elif provider == "google":
                api_key = os.getenv("GOOGLE_API_KEY")
                cse_id = os.getenv("GOOGLE_CSE_ID")
                if not api_key or not cse_id:
                    print("Warning: GOOGLE_API_KEY or GOOGLE_CSE_ID not found in environment")
                    return None

                search = GoogleSearchAPIWrapper(
                    google_api_key=api_key,
                    google_cse_id=cse_id
                )
                return Tool(
                    name="Google Search",
                    description="Search Google for recent results.",
                    func=search.run
                )
            elif provider == "duckduckgo":
                return DuckDuckGoSearchRun()
            else:
                print(f"Warning: Unsupported search provider: {provider}")
                return None
        except Exception as e:
            print(f"Warning: Could not initialize search tool for {provider}: {str(e)}")
            return None

    def _format_search_results(self, results: List[Any], provider: str) -> List[SearchResult]:
        """
        Format search results into a standardized format.

        Args:
            results: Raw search results from the provider
            provider: The provider that returned these results

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
                    snippet=result,
                    provider=provider
                ))
                continue

            # Handle dictionary results
            if isinstance(result, dict):
                if provider == "tavily":
                    formatted_results.append(SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("url", "No URL"),
                        snippet=result.get("content", "No content"),
                        provider=provider
                    ))
                elif provider == "serper":
                    formatted_results.append(SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("link", "No URL"),
                        snippet=result.get("snippet", "No snippet"),
                        provider=provider
                    ))
                elif provider == "google":
                    formatted_results.append(SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("link", "No URL"),
                        snippet=result.get("snippet", "No snippet"),
                        provider=provider
                    ))
                else:
                    # Generic dictionary handling
                    formatted_results.append(SearchResult(
                        title=result.get("title", "No title"),
                        url=result.get("url", result.get("link", "No URL")),
                        snippet=result.get("snippet", result.get("content", str(result))),
                        provider=provider
                    ))
            else:
                # Handle other types
                formatted_results.append(SearchResult(
                    title="Search Result",
                    url="No direct URL available",
                    snippet=str(result),
                    provider=provider
                ))

        return formatted_results

    def _search_with_provider(self, query: str, provider: str, max_retries: int = 3) -> List[SearchResult]:
        """
        Perform a search using a specific provider.

        Args:
            query: Search query
            provider: Provider to use for search
            max_retries: Maximum number of retries for transient errors

        Returns:
            List[SearchResult]: Search results
        """
        print(f"Searching with provider: {provider}, query: {query}")

        if provider not in self.search_tools:
            print(f"Provider {provider} is not available in search_tools")
            return [SearchResult(
                title="Provider Error",
                url="",
                snippet=f"Provider {provider} is not available.",
                provider=provider
            )]

        search_tool = self.search_tools[provider]
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                print(f"Invoking search tool for {provider}...")
                raw_results = search_tool.invoke(query)
                print(f"Got raw results from {provider}: {type(raw_results)}")

                if raw_results is None:
                    print(f"Warning: {provider} returned None")
                    raw_results = []

                # Handle different return types from different providers
                if not isinstance(raw_results, list):
                    if provider == "serper":
                        # Serper returns a dictionary with organic results
                        if isinstance(raw_results, dict) and "organic" in raw_results:
                            print(f"Extracting organic results from Serper response")
                            raw_results = raw_results["organic"]
                        elif isinstance(raw_results, dict) and "error" in raw_results:
                            # Handle Serper API error
                            error_msg = raw_results.get("error", "Unknown Serper API error")
                            print(f"Serper API error: {error_msg}")

                            # Check if it's a rate limit error (common with API services)
                            if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                                # This is a transient error, retry after a delay
                                retry_count += 1
                                if retry_count < max_retries:
                                    print(f"Rate limit hit, retrying in {retry_count} seconds...")
                                    time.sleep(retry_count)  # Exponential backoff
                                    continue

                            # Return a helpful error message as a search result
                            return [SearchResult(
                                title="Search Error",
                                url="",
                                snippet=f"Error searching with {provider}: {error_msg}. Please try again later.",
                                provider=provider
                            )]
                        else:
                            print(f"Converting single result to list for {provider}")
                            raw_results = [raw_results]
                    else:
                        print(f"Converting single result to list for {provider}")
                        raw_results = [raw_results]

                formatted_results = self._format_search_results(raw_results, provider)
                print(f"Formatted {len(formatted_results)} results from {provider}")
                return formatted_results

            except Exception as e:
                last_error = e
                print(f"Search error with {provider} (attempt {retry_count + 1}/{max_retries}): {str(e)}")

                # Print detailed traceback for debugging
                import traceback
                print(f"Detailed error traceback for {provider}:")
                traceback.print_exc()

                # Check if it's a connection error or timeout (common transient errors)
                error_str = str(e).lower()
                is_transient = any(term in error_str for term in [
                    "timeout", "connection", "network", "temporary",
                    "rate limit", "too many requests", "503", "502"
                ])

                if is_transient and retry_count < max_retries - 1:
                    # This is a transient error, retry after a delay
                    retry_count += 1
                    print(f"Transient error, retrying in {retry_count} seconds...")
                    time.sleep(retry_count)  # Exponential backoff
                else:
                    # Non-transient error or max retries reached
                    break

        # If we get here, all retries failed
        error_message = str(last_error) if last_error else "Unknown search error"
        print(f"Search with {provider} failed after {max_retries} attempts: {error_message}")

        # Return a helpful error message as a search result
        error_result = SearchResult(
            title="Search Error",
            url="",
            snippet=f"Error performing search with {provider}: {error_message}. Please try again later.",
            provider=provider
        )
        print(f"Returning error result for {provider}: {error_result}")
        return [error_result]

    async def _search_with_provider_async(self, query: str, provider: str, max_retries: int = 3) -> List[SearchResult]:
        """
        Perform a search using a specific provider asynchronously.

        Args:
            query: Search query
            provider: Provider to use for search
            max_retries: Maximum number of retries for transient errors

        Returns:
            List[SearchResult]: Search results
        """
        # Use ThreadPoolExecutor to run the synchronous search in a separate thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                executor, self._search_with_provider, query, provider, max_retries
            )

    async def _search_parallel(self, query: str, max_retries: int = 3) -> List[SearchResult]:
        """
        Perform searches with all available providers in parallel.

        Args:
            query: Search query
            max_retries: Maximum number of retries for transient errors

        Returns:
            List[SearchResult]: Combined search results from all providers
        """
        # Create tasks for each provider
        tasks = []
        for provider in self.search_tools.keys():
            task = self._search_with_provider_async(query, provider, max_retries)
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Flatten the results
        all_results = []
        for provider_results in results:
            all_results.extend(provider_results)

        return all_results

    def search(self, query: str, max_retries: int = 3) -> List[SearchResult]:
        """
        Perform a search using the configured providers.

        Args:
            query: Search query
            max_retries: Maximum number of retries for transient errors

        Returns:
            List[SearchResult]: Search results
        """
        # If no search tools are available, return an error
        if not self.search_tools:
            return [SearchResult(
                title="Search Error",
                url="",
                snippet="No search providers are available. Please check your API keys.",
                provider="none"
            )]

        # If parallel search is enabled and we have multiple providers, use parallel search
        if self.config.parallel_search and len(self.search_tools) > 1:
            try:
                # Create an event loop if one doesn't exist
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Run the parallel search
                results = loop.run_until_complete(self._search_parallel(query, max_retries))
                return results
            except Exception as e:
                print(f"Error in parallel search: {str(e)}")
                print("Falling back to sequential search")
                # Fall back to sequential search

        # Sequential search with all providers
        all_results = []
        for provider in self.search_tools.keys():
            results = self._search_with_provider(query, provider, max_retries)
            all_results.extend(results)

        return all_results

    def evaluate_search_results(self, query: str, results: List[SearchResult]) -> Dict[str, Any]:
        """
        Evaluate search results to determine if they are sufficient or if additional searches are needed.

        Args:
            query: Original search query
            results: Search results to evaluate

        Returns:
            Dict[str, Any]: Evaluation results including sufficiency and additional queries
        """
        print(f"\nEvaluating search results for query: {query}")
        print(f"Number of results to evaluate: {len(results)}")

        if not self.config.evaluate_results:
            print("Evaluation disabled in config, skipping")
            return {
                "sufficient": True,
                "additional_queries": []
            }

        if not results:
            print("No results found, marking as insufficient")
            # No results, definitely need more
            return {
                "sufficient": False,
                "additional_queries": [f"more information about {query}"]
            }

        # Format results for evaluation
        formatted_results = "=== SEARCH RESULTS FOR EVALUATION ===\n\n"

        # Group results by provider for better organization
        results_by_provider = {}
        for result in results:
            provider = result.provider or "unknown"
            if provider not in results_by_provider:
                results_by_provider[provider] = []
            results_by_provider[provider].append(result)

        # Add provider sections
        for provider, provider_results in results_by_provider.items():
            formatted_results += f"--- Results from {provider.upper()} ---\n\n"

            for i, result in enumerate(provider_results, 1):
                formatted_results += f"RESULT {i}:\n"
                formatted_results += f"Title: {result.title}\n"
                formatted_results += f"URL: {result.url}\n"
                formatted_results += f"Content: {result.snippet}\n\n"

        formatted_results += "=== END OF SEARCH RESULTS ==="

        print(f"Formatted {len(results)} results for evaluation")

        try:
            # Generate evaluation using LLM
            print("Invoking evaluation LLM...")
            response = self.evaluation_llm.invoke(
                self.evaluation_prompt.format(
                    query=query,
                    search_results=formatted_results
                )
            )

            # Extract content from response
            if isinstance(response, dict):
                evaluation = response.get("content", "")
            elif hasattr(response, "content"):
                evaluation = response.content
            else:
                evaluation = str(response)

            print(f"Evaluation response received, length: {len(evaluation)}")
            print(f"Evaluation summary: {evaluation[:100]}...")

            # Parse the evaluation to determine if results are sufficient
            evaluation_lower = evaluation.lower()
            sufficient = (
                "sufficient" in evaluation_lower or
                "adequate" in evaluation_lower or
                "enough information" in evaluation_lower
            ) and not (
                "not sufficient" in evaluation_lower or
                "insufficient" in evaluation_lower or
                "not enough" in evaluation_lower or
                "need more" in evaluation_lower
            )

            print(f"Results sufficient: {sufficient}")

            # Extract additional queries if results are insufficient
            additional_queries = []
            if not sufficient and self.config.additional_queries:
                print("Results insufficient, extracting additional queries")
                # Look for suggested queries in the evaluation
                lines = evaluation.split('\n')
                for line in lines:
                    line = line.strip()
                    if (
                        "search for" in line.lower() or
                        "query about" in line.lower() or
                        "search query" in line.lower() or
                        "additional query" in line.lower() or
                        line.startswith('"') and line.endswith('"') or
                        line.startswith("- ")
                    ):
                        # Clean up the line to extract just the query
                        query_text = line
                        for prefix in ["search for", "query about", "search query", "additional query", "- "]:
                            if prefix in query_text.lower():
                                query_text = query_text.lower().split(prefix, 1)[1].strip()

                        # Remove quotes if present
                        if query_text.startswith('"') and query_text.endswith('"'):
                            query_text = query_text[1:-1]

                        if query_text and len(query_text) > 3:  # Minimum query length
                            additional_queries.append(query_text)
                            print(f"Added additional query: {query_text}")

            # If no additional queries were found but results are insufficient, add a default one
            if not additional_queries and not sufficient:
                default_query = f"latest information about {query} in 2025"
                additional_queries.append(default_query)
                print(f"No specific additional queries found, adding default: {default_query}")

            final_queries = additional_queries[:3]  # Limit to 3 additional queries
            print(f"Final additional queries: {final_queries}")

            return {
                "sufficient": sufficient,
                "evaluation": evaluation,
                "additional_queries": final_queries
            }

        except Exception as e:
            print(f"Error evaluating search results: {str(e)}")
            import traceback
            traceback.print_exc()

            # Default to insufficient with a generic additional query if evaluation fails
            return {
                "sufficient": False,
                "evaluation": f"Error evaluating results: {str(e)}",
                "additional_queries": [f"more detailed information about {query}"]
            }

    def perform_search_with_evaluation(self, query: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Perform a search with evaluation and additional searches if needed.

        Args:
            query: Search query
            max_retries: Maximum number of retries for transient errors

        Returns:
            Dict[str, Any]: Search results and evaluation
        """
        # Perform initial search
        initial_results = self.search(query, max_retries)

        # Skip evaluation if disabled or if we got errors
        if not self.config.evaluate_results or any("Error" in result.title for result in initial_results):
            return {
                "results": initial_results,
                "evaluation": {
                    "sufficient": True,
                    "additional_queries": []
                },
                "all_queries": [query]
            }

        # Evaluate the results
        evaluation = self.evaluate_search_results(query, initial_results)

        # If results are sufficient or additional queries are disabled, return
        if evaluation["sufficient"] or not self.config.additional_queries or not evaluation["additional_queries"]:
            return {
                "results": initial_results,
                "evaluation": evaluation,
                "all_queries": [query]
            }

        # Perform additional searches
        all_results = list(initial_results)  # Copy initial results
        all_queries = [query]  # Track all queries used

        for additional_query in evaluation["additional_queries"]:
            print(f"Performing additional search: {additional_query}")
            additional_results = self.search(additional_query, max_retries)
            all_results.extend(additional_results)
            all_queries.append(additional_query)

        # Remove duplicates (based on URL)
        unique_results = []
        seen_urls = set()

        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return {
            "results": unique_results,
            "evaluation": evaluation,
            "all_queries": all_queries
        }

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

            # Check if we have a subtask in the state (used by MCP)
            if "current_subtask" in state and state["current_subtask"]:
                subtask = state["current_subtask"]
                if "description" in subtask:
                    # Use the subtask description as the query
                    query = subtask["description"]
                    print(f"Using subtask description as query: {query}")

            # Perform search with evaluation and additional searches if needed
            search_data = self.perform_search_with_evaluation(query)
            search_results = search_data["results"]
            evaluation = search_data["evaluation"]
            all_queries = search_data["all_queries"]

            # Check if we got an error result
            has_error = any("Error" in result.title for result in search_results)

            # Format results for display
            formatted_results = "=== SEARCH RESULTS ===\n\n"

            # Group results by provider for better organization
            results_by_provider = {}
            for result in search_results:
                provider = result.provider or "unknown"
                if provider not in results_by_provider:
                    results_by_provider[provider] = []
                results_by_provider[provider].append(result)

            # Add provider sections
            for provider, provider_results in results_by_provider.items():
                formatted_results += f"--- Results from {provider.upper()} ---\n\n"

                for i, result in enumerate(provider_results, 1):
                    formatted_results += f"RESULT {i}:\n"
                    formatted_results += f"Title: {result.title}\n"
                    formatted_results += f"URL: {result.url}\n"
                    formatted_results += f"Content: {result.snippet}\n\n"

            formatted_results += "=== END OF SEARCH RESULTS ===\n\n"
            formatted_results += "IMPORTANT: Use ONLY the information from these search results to answer the user's question."

            # Print formatted results for debugging
            print("\nFormatted search results for LLM:")
            print(formatted_results[:500] + "..." if len(formatted_results) > 500 else formatted_results)

            # Generate response using LLM with simplified prompt
            prompt_with_results = self.prompt.format(
                query=query,
                search_results=formatted_results
            )

            # Print prompt for debugging
            print("\nPrompt for LLM (first 500 chars):")

            # Convert to messages for better debugging
            messages = prompt_with_results.to_messages()
            prompt_str = "\n".join([f"{msg.type}: {msg.content[:100]}..." if len(msg.content) > 100 else f"{msg.type}: {msg.content}" for msg in messages])
            print(prompt_str[:500] + "..." if len(prompt_str) > 500 else prompt_str)

            # Invoke LLM
            response = self.llm.invoke(messages)

            # Extract content from response
            if isinstance(response, dict):
                content = response.get("content", "No search results available")
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Update state
            state["agent_outputs"]["search_agent"] = {
                "results": [
                    result.model_dump() if hasattr(result, "model_dump")
                    else vars(result) if hasattr(result, "__dict__")
                    else {"title": str(result), "url": "", "snippet": str(result), "provider": getattr(result, "provider", "unknown")}
                    for result in search_results
                ],
                "has_error": has_error,
                "evaluation": evaluation,
                "all_queries": all_queries
            }

            # Add error information if there was an error
            if has_error:
                error_results = [r for r in search_results if "Error" in r.title]
                if error_results:
                    state["agent_outputs"]["search_agent"]["error"] = error_results[0].snippet

                    # If this is a subtask in MCP, mark it for potential fallback
                    if "current_subtask" in state:
                        state["agent_outputs"]["search_agent"]["needs_fallback"] = True

            state["messages"].append({"role": "assistant", "content": content})

        except Exception as e:
            # Handle errors gracefully
            error_message = f"Search agent encountered an error: {str(e)}"
            print(error_message)

            # Get detailed error information
            import traceback
            trace = traceback.format_exc()
            print(f"Detailed error: {trace}")

            # Update state with error information
            state["agent_outputs"]["search_agent"] = {
                "error": str(e),
                "traceback": trace,
                "results": [],
                "has_error": True
            }

            # If this is a subtask in MCP, mark it for potential fallback
            if "current_subtask" in state:
                state["agent_outputs"]["search_agent"]["needs_fallback"] = True

            # Add error response to messages
            state["messages"].append({
                "role": "assistant",
                "content": f"I apologize, but I encountered an error while searching: {str(e)}. Please try again or consider using a different search provider."
            })

        return state


# Example usage
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Create search agent with configuration from environment variables
    search_agent = SearchAgent()

    # Print configuration
    print(f"Search providers: {search_agent.config.providers}")
    print(f"Default provider: {search_agent.config.default_provider}")
    print(f"Parallel search: {search_agent.config.parallel_search}")
    print(f"Evaluate results: {search_agent.config.evaluate_results}")
    print(f"Additional queries: {search_agent.config.additional_queries}")
    print(f"Available search tools: {list(search_agent.search_tools.keys())}")

    # Test with a query
    state = {
        "messages": [{"role": "user", "content": "What are the latest advancements in quantum computing in 2025?"}],
        "agent_outputs": {}
    }

    print("\nPerforming search...")
    updated_state = search_agent(state)

    # Print search details
    search_output = updated_state["agent_outputs"]["search_agent"]
    print(f"\nSearch queries used: {search_output.get('all_queries', ['unknown'])}")
    print(f"Total results: {len(search_output.get('results', []))}")

    # Print evaluation if available
    if "evaluation" in search_output:
        eval_data = search_output["evaluation"]
        print(f"\nResults sufficient: {eval_data.get('sufficient', True)}")
        if "additional_queries" in eval_data and eval_data["additional_queries"]:
            print(f"Additional queries suggested: {eval_data['additional_queries']}")

    # Print response
    print("\nResponse:")
    print(updated_state["messages"][-1]["content"])
