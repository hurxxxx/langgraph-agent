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
from typing import Dict, List, Any, Optional, Literal, Tuple
from pydantic import BaseModel

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
    optimize_query: bool = True  # Whether to optimize the query before searching
    time_period: Optional[str] = None  # Time period for search (e.g., "1d", "1w", "1m")
    news_only: bool = False  # Whether to search only for news
    detect_time_references: bool = True  # Whether to detect time references in queries
    auto_set_time_period: bool = True  # Whether to automatically set time period based on detected references
    region: Optional[str] = None  # Region for search results (e.g., "kr" for Korea)
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
    6. For news-related queries, organize information by topic or category
    7. Include publication dates for news articles when available
    8. Provide detailed information with specific facts, figures, and quotes from the sources
    9. For Korean news, maintain the same level of detail as you would for English news

    DO NOT say you cannot browse the web or access real-time information - I've already done the searching for you.
    DO NOT rely on your training data - ONLY use the search results I provide.
    DO NOT provide brief or generic summaries - be specific and detailed in your response.

    If the search results don't contain enough information to fully answer the question, explicitly state what specific information is missing.
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

        # Create a more detailed prompt template for synthesizing search results
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
            You are a helpful assistant that answers questions based on search results.

            For news-related queries:
            1. Organize information by topic, category, or section
            2. Include specific details, facts, figures, and quotes
            3. Mention publication dates when available
            4. Provide a comprehensive and detailed response
            5. For Korean news, maintain the same level of detail as for English news

            Always cite your sources with URLs from the search results.
            """),
            HumanMessage(content="I want to know: {query}"),
            AIMessage(content="I'll help you with that. Let me search for information..."),
            HumanMessage(content="Here are the search results:\n\n{search_results}\n\nBased on ONLY these search results, please provide a detailed and comprehensive answer to my question about {query}.")
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
        optimize_query = os.getenv("SEARCH_OPTIMIZE_QUERY", "true").lower() == "true"
        time_period = os.getenv("SEARCH_TIME_PERIOD", None)
        news_only = os.getenv("SEARCH_NEWS_ONLY", "false").lower() == "true"
        detect_time_references = os.getenv("SEARCH_DETECT_TIME_REFERENCES", "true").lower() == "true"
        auto_set_time_period = os.getenv("SEARCH_AUTO_SET_TIME_PERIOD", "true").lower() == "true"
        region = os.getenv("SEARCH_REGION", None)

        return SearchAgentConfig(
            providers=providers,
            default_provider=default_provider,
            max_results=max_results,
            parallel_search=parallel_search,
            additional_queries=additional_queries,
            optimize_query=optimize_query,
            time_period=time_period,
            news_only=news_only,
            detect_time_references=detect_time_references,
            auto_set_time_period=auto_set_time_period,
            region=region
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

                # Create Tavily search tool with additional parameters
                params = {
                    "api_key": api_key,
                    "max_results": self.config.max_results
                }

                # Add time period if specified
                if self.config.time_period:
                    if self.config.time_period == "1d":
                        params["search_depth"] = "recent"
                    elif self.config.time_period == "1w":
                        params["search_depth"] = "moderate"
                    else:
                        params["search_depth"] = "deep"

                # Set to news only if specified
                if self.config.news_only:
                    # Default news domains
                    news_domains = ["news.google.com", "bbc.com", "cnn.com", "reuters.com",
                                  "nytimes.com", "washingtonpost.com", "theguardian.com",
                                  "apnews.com", "bloomberg.com", "ft.com", "wsj.com"]

                    # Add Korean news domains if region is Korea
                    if self.config.region == "kr":
                        korean_news_domains = ["news.naver.com", "news.daum.net", "yna.co.kr",
                                             "chosun.com", "joongang.co.kr", "donga.com",
                                             "hani.co.kr", "mk.co.kr", "hankyung.com",
                                             "mt.co.kr", "sedaily.com", "edaily.co.kr"]
                        news_domains.extend(korean_news_domains)

                    params["include_domains"] = news_domains

                tavily_tool = TavilySearchResults(**params)
                print(f"Successfully initialized Tavily search tool with API key: {api_key[:5]}...")
                return tavily_tool
            elif provider == "serper":
                api_key = os.getenv("SERPER_API_KEY")
                if not api_key:
                    print("Warning: SERPER_API_KEY not found in environment")
                    return None

                # Create Serper search wrapper with additional parameters
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
                    # Set region based on config or default to US
                    params["gl"] = self.config.region or "us"
                    params["search_type"] = "news"
                elif self.config.region:
                    # Set region if specified in config
                    params["gl"] = self.config.region

                search = GoogleSerperAPIWrapper(**params)

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

                # Create Google search wrapper with additional parameters
                params = {
                    "google_api_key": api_key,
                    "google_cse_id": cse_id
                }

                # Add time period if specified
                if self.config.time_period:
                    if self.config.time_period == "1d":
                        params["dateRestrict"] = "d1"  # Past 24 hours
                    elif self.config.time_period == "1w":
                        params["dateRestrict"] = "w1"  # Past week
                    elif self.config.time_period == "1m":
                        params["dateRestrict"] = "m1"  # Past month

                # Set to news only if specified
                if self.config.news_only:
                    params["cr"] = "countryUS"  # Set region to US for more news results

                search = GoogleSearchAPIWrapper(**params)

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
        if provider not in self.search_tools:
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
                raw_results = search_tool.invoke(query)

                if raw_results is None:
                    raw_results = []

                # Handle different return types from different providers
                if not isinstance(raw_results, list):
                    if provider == "serper":
                        # Serper returns a dictionary with organic results
                        if isinstance(raw_results, dict) and "organic" in raw_results:
                            raw_results = raw_results["organic"]
                        elif isinstance(raw_results, dict) and "error" in raw_results:
                            # Handle Serper API error
                            error_msg = raw_results.get("error", "Unknown Serper API error")

                            # Check if it's a rate limit error (common with API services)
                            if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
                                # This is a transient error, retry after a delay
                                retry_count += 1
                                if retry_count < max_retries:
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
                            raw_results = [raw_results]
                    else:
                        raw_results = [raw_results]

                formatted_results = self._format_search_results(raw_results, provider)
                return formatted_results

            except Exception as e:
                last_error = e

                # Check if it's a connection error or timeout (common transient errors)
                error_str = str(e).lower()
                is_transient = any(term in error_str for term in [
                    "timeout", "connection", "network", "temporary",
                    "rate limit", "too many requests", "503", "502"
                ])

                if is_transient and retry_count < max_retries - 1:
                    # This is a transient error, retry after a delay
                    retry_count += 1
                    time.sleep(retry_count)  # Exponential backoff
                else:
                    # Non-transient error or max retries reached
                    break

        # If we get here, all retries failed
        error_message = str(last_error) if last_error else "Unknown search error"

        # Return a helpful error message as a search result
        return [SearchResult(
            title="Search Error",
            url="",
            snippet=f"Error performing search with {provider}: {error_message}. Please try again later.",
            provider=provider
        )]

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

    def detect_time_references(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect time references in a query and determine the appropriate time period.
        Uses LLM to analyze the query instead of hardcoded patterns.

        Args:
            query: Search query

        Returns:
            Tuple[bool, Optional[str]]:
                - Whether a time reference was detected
                - The detected time period (e.g., "1d", "1w", "1m")
        """
        if not self.config.detect_time_references:
            return False, None

        # Use LLM to analyze the query for time references
        system_content = """
        You are an expert at analyzing search queries for time references.
        Analyze the query and determine if it contains time references like:
        - Today, current, latest, recent, now
        - Yesterday, last day
        - This week, current week, last few days, past week
        - This month, current month, last month, last 30 days
        - News-related terms (news, report, headlines, articles)
        - Korean time references (오늘, 어제, 이번 주, 이번 달, 뉴스, etc.)

        Return a JSON with these fields:
        - has_time_reference: true/false
        - time_period: "1d" for today/yesterday, "1w" for this week, "1m" for this month, or null
        - reason: brief explanation of why you determined this
        """

        try:
            # Use a lower temperature for more consistent results
            response = self.evaluation_llm.invoke([
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Query: {query}"}
            ])

            # Extract content from response
            if isinstance(response, dict):
                content = response.get("content", "")
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Try to parse JSON from the response
            import json

            # Try to parse the entire content as JSON first
            try:
                result = json.loads(content)
                has_time_ref = result.get("has_time_reference", False)
                time_period = result.get("time_period")

                if has_time_ref and time_period:
                    return True, time_period
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                start_idx = content.find('{')
                end_idx = content.rfind('}')

                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    try:
                        json_str = content[start_idx:end_idx+1]
                        result = json.loads(json_str)
                        has_time_ref = result.get("has_time_reference", False)
                        time_period = result.get("time_period")

                        if has_time_ref and time_period:
                            return True, time_period
                    except json.JSONDecodeError:
                        pass

            # Fallback for news queries
            if "news" in query.lower() or "뉴스" in query:
                return True, "1d"

            return False, None

        except Exception:
            # Fallback to a simple check for news-related terms
            if "news" in query.lower() or "뉴스" in query:
                return True, "1d"
            return False, None

    def optimize_query(self, query: str) -> List[str]:
        """
        Optimize a search query to improve search results using LLM.
        Uses time_period option instead of adding dates to queries.

        Args:
            query: Original search query

        Returns:
            List[str]: List of optimized search queries
        """
        if not self.config.optimize_query:
            return [query]

        try:
            # Detect time references
            has_time_ref, time_period = self.detect_time_references(query)

            # If time reference detected and auto_set_time_period is enabled, update config
            if has_time_ref and self.config.auto_set_time_period:
                self.config.time_period = time_period

                # Check if this is a news query using LLM
                news_check_prompt = f"Is this query about news or current events? Query: '{query}'. Answer only 'yes' or 'no'."
                news_response = self.evaluation_llm.invoke([{"role": "user", "content": news_check_prompt}])

                # Extract content from response
                if isinstance(news_response, dict):
                    news_content = news_response.get("content", "").lower()
                elif hasattr(news_response, "content"):
                    news_content = news_response.content.lower()
                else:
                    news_content = str(news_response).lower()

                is_news_query = "yes" in news_content

                if is_news_query:
                    self.config.news_only = True

                    # Increase max_results for news queries to get more comprehensive results
                    original_max_results = self.config.max_results
                    self.config.max_results = max(10, original_max_results)

            # Use LangChain's query optimization capabilities
            system_content = """
            You are an expert at optimizing search queries. Your job is to:
            1. Analyze the user's search intent
            2. Extract key concepts and terms
            3. Generate 3-5 optimized search queries that will yield better results

            For each query type:
            - News queries: Focus on specific topics, sources, and relevant keywords
            - Technical queries: Include specific technologies, frameworks, and problem descriptions
            - General knowledge: Focus on precise terminology and alternative phrasings

            IMPORTANT: DO NOT include dates or years in the queries. The search system will handle time filtering automatically.

            Return ONLY the optimized queries, one per line, without any explanation or numbering.
            """

            # Use the LLM to generate optimized queries
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Original query: {query}"}
            ]

            response = self.evaluation_llm.invoke(messages)

            # Extract content from response
            if isinstance(response, dict):
                content = response.get("content", "")
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Split into individual queries
            optimized_queries = [q.strip() for q in content.strip().split('\n') if q.strip()]

            # Always include the original query
            if query not in optimized_queries:
                optimized_queries.append(query)

            # If no optimized queries were generated, just use the original
            if not optimized_queries:
                optimized_queries = [query]

            return optimized_queries

        except Exception:
            return [query]  # Return the original query if optimization fails

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

        # Optimize the query if enabled
        if self.config.optimize_query:
            optimized_queries = self.optimize_query(query)
        else:
            optimized_queries = [query]

        all_results = []

        # Search with each optimized query
        for optimized_query in optimized_queries:
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
                    results = loop.run_until_complete(self._search_parallel(optimized_query, max_retries))
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error in parallel search: {str(e)}")
                    print("Falling back to sequential search")
                    # Fall back to sequential search
                    for provider in self.search_tools.keys():
                        results = self._search_with_provider(optimized_query, provider, max_retries)
                        all_results.extend(results)
            else:
                # Sequential search with all providers
                for provider in self.search_tools.keys():
                    results = self._search_with_provider(optimized_query, provider, max_retries)
                    all_results.extend(results)

        # Remove duplicates based on URL
        unique_results = []
        seen_urls = set()

        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return unique_results

    def evaluate_search_results(self, query: str, results: List[SearchResult]) -> Dict[str, Any]:
        """
        Evaluate search results to determine if they are sufficient or if additional searches are needed.
        Uses LLM to analyze the results and suggest additional queries if needed.

        Args:
            query: Original search query
            results: Search results to evaluate

        Returns:
            Dict[str, Any]: Evaluation results including sufficiency and additional queries
        """
        if not self.config.evaluate_results:
            return {
                "sufficient": True,
                "additional_queries": []
            }

        if not results:
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
            # Use a structured prompt for the LLM to evaluate results
            system_content = """
            You are an expert at evaluating search results. Your task is to:
            1. Analyze the search results for the given query
            2. Determine if the results provide sufficient information to answer the query
            3. If the results are insufficient, suggest additional search queries

            Return your evaluation as a JSON object with the following structure:
            {
                "sufficient": true/false,
                "reason": "Brief explanation of your assessment",
                "additional_queries": ["query1", "query2", "query3"]
            }

            If the results are sufficient, the "additional_queries" array can be empty.
            If the results are insufficient, suggest 2-3 specific additional queries that would help find the missing information.

            For news-related queries, check if the results are recent and from reliable sources.
            For technical queries, check if the results provide detailed technical information.
            For general knowledge queries, check if the results provide comprehensive information.
            """

            # Generate evaluation using LLM
            print("Invoking evaluation LLM...")
            response = self.evaluation_llm.invoke([
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Query: {query}\n\nSearch Results:\n{formatted_results}"}
            ])

            # Extract content from response
            if isinstance(response, dict):
                evaluation = response.get("content", "")
            elif hasattr(response, "content"):
                evaluation = response.content
            else:
                evaluation = str(response)



            # Try to parse JSON from the response
            import json

            # Try to parse the entire content as JSON first
            try:
                result = json.loads(evaluation)
                sufficient = result.get("sufficient", False)
                reason = result.get("reason", "No reason provided")
                additional_queries = result.get("additional_queries", [])

                # Limit to 3 additional queries
                final_queries = additional_queries[:3]

                return {
                    "sufficient": sufficient,
                    "evaluation": reason,
                    "additional_queries": final_queries
                }
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                start_idx = evaluation.find('{')
                end_idx = evaluation.rfind('}')

                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    try:
                        json_str = evaluation[start_idx:end_idx+1]
                        result = json.loads(json_str)
                        sufficient = result.get("sufficient", False)
                        reason = result.get("reason", "No reason provided")
                        additional_queries = result.get("additional_queries", [])

                        # Limit to 3 additional queries
                        final_queries = additional_queries[:3]

                        return {
                            "sufficient": sufficient,
                            "evaluation": reason,
                            "additional_queries": final_queries
                        }
                    except json.JSONDecodeError:
                        # Fall back to text analysis
                        pass

            # Fallback: Simple text analysis if JSON parsing fails
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

            # Extract additional queries using simple pattern matching
            additional_queries = []
            if not sufficient:
                # Look for lines that might contain suggested queries
                lines = evaluation.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 10 and not line.startswith('{') and not line.startswith('}'):
                        # Skip lines that are likely part of the analysis
                        if any(term in line.lower() for term in ["sufficient", "evaluation", "analysis", "result"]):
                            continue
                        # Add as a potential query if it looks like one
                        if "?" in line or any(term in line.lower() for term in ["search", "query", "find", "look for"]):
                            # Clean up the line
                            query_text = line
                            # Remove common prefixes
                            for prefix in ["search for", "query about", "search query", "additional query", "- "]:
                                if prefix in query_text.lower():
                                    query_text = query_text.lower().split(prefix, 1)[1].strip()
                            # Remove quotes if present
                            if query_text.startswith('"') and query_text.endswith('"'):
                                query_text = query_text[1:-1]
                            # Add if it's a reasonable length
                            if query_text and len(query_text) > 3:
                                additional_queries.append(query_text)

            # If no additional queries were found but results are insufficient, add a default one
            if not additional_queries and not sufficient:
                default_query = f"latest information about {query}"
                additional_queries.append(default_query)

            # Limit to 3 additional queries
            final_queries = additional_queries[:3]

            return {
                "sufficient": sufficient,
                "evaluation": evaluation,
                "additional_queries": final_queries
            }

        except Exception as e:
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

            # Generate response using LLM with simplified prompt
            try:
                # Determine if this is a news-related query
                is_news_query = self.config.news_only or any(term in query.lower() for term in [
                    'news', 'report', 'headlines', 'articles', 'latest', 'today', 'current',
                    '뉴스', '기사', '소식', '보도', '헤드라인', '속보', '오늘', '최신'
                ])

                # Create system message with appropriate instructions
                system_content = """
                You are a helpful assistant that answers questions based on search results.
                """

                if is_news_query:
                    system_content += """
                    This is a news-related query. Your response should:
                    1. Organize information by topic, category, or section
                    2. Include specific details, facts, figures, and quotes
                    3. Mention publication dates when available
                    4. Provide a comprehensive and detailed response (at least 300-500 words)
                    5. For Korean news, maintain the same level of detail as for English news

                    Format your response with clear sections and subsections.
                    """

                # Create messages directly
                messages = [
                    SystemMessage(content=system_content),
                    HumanMessage(content=f"I want to know: {query}"),
                    AIMessage(content="I'll help you with that. Let me search for information..."),
                    HumanMessage(content=f"Here are the search results:\n\n{formatted_results}\n\nBased on ONLY these search results, please provide a detailed and comprehensive answer to my question about {query}.")
                ]

                # Invoke LLM
                response = self.llm.invoke(messages)
            except Exception as e:
                # Fallback to a simpler prompt
                response = self.llm.invoke([
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessage(content=f"Based on these search results: {formatted_results[:1000]}..., what are the latest advancements in quantum computing in 2025?")
                ])

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
            # Get detailed error information
            import traceback
            trace = traceback.format_exc()

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



