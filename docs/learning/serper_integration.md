# Serper API Integration with LangGraph (as of May 2025)

This document outlines how to integrate Serper API with LangGraph for web search capabilities in a multi-agent system.

## Overview

Serper is a low-cost Google Search API that provides access to Google search results at a fraction of the cost of the official Google Custom Search API. It's a popular choice for AI applications that need web search capabilities.

Key features of Serper API:
- Fast response times (1-2 seconds)
- Cost-effective (starting at $0.30 per 1000 queries)
- Access to Google search results, including answer boxes, knowledge graphs, and organic results
- Simple API with JSON responses

## Integration with LangChain

LangChain provides a wrapper for Serper API through the `GoogleSerperAPIWrapper` class in the `langchain_community.utilities` module.

### Installation

```bash
pip install langchain-community
```

### Basic Usage

```python
import os
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import Tool

# Set up the API key
os.environ["SERPER_API_KEY"] = "your-serper-api-key"

# Create the wrapper
search = GoogleSerperAPIWrapper(
    serper_api_key=os.environ["SERPER_API_KEY"],
    k=5  # Number of results to return
)

# Create a tool
serper_tool = Tool(
    name="Serper Search",
    description="Search Google using Serper API for recent results.",
    func=search.run
)

# Use the tool
results = serper_tool.invoke("What is LangGraph?")
print(results)
```

### Response Format

Serper API returns results in JSON format. The response typically includes:

```json
{
  "searchParameters": {
    "q": "your query",
    "gl": "us",
    "hl": "en",
    "num": 10,
    "type": "search"
  },
  "organic": [
    {
      "title": "Result title",
      "link": "https://example.com",
      "snippet": "Result snippet...",
      "position": 1
    },
    // More results...
  ],
  "answerBox": {
    "title": "Answer box title",
    "answer": "Answer text",
    "snippet": "Answer snippet"
  },
  "knowledgeGraph": {
    "title": "Knowledge graph title",
    "type": "Type",
    "description": "Description",
    "attributes": {
      "attribute1": "value1",
      "attribute2": "value2"
    }
  }
}
```

## Integration with LangGraph Multi-Agent System

In a LangGraph multi-agent system, Serper can be used as a search provider for a search agent. Here's how to integrate it:

### 1. Create a Search Agent with Serper

```python
from typing import Dict, List, Any
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.tools import Tool

class SearchResult(BaseModel):
    """Model for a search result."""
    title: str
    url: str
    snippet: str

class SearchAgent:
    """Search agent that uses Serper API."""
    
    def __init__(self, api_key, max_results=5):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Initialize Serper search
        search = GoogleSerperAPIWrapper(
            serper_api_key=api_key,
            k=max_results
        )
        
        self.search_tool = Tool(
            name="Serper Search",
            description="Search Google using Serper API for recent results.",
            func=search.run
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a search agent that retrieves information from the web."),
            MessagesPlaceholder(variable_name="messages"),
            SystemMessage(content="Search results: {search_results}")
        ])
    
    def __call__(self, state):
        # Extract query from state
        query = state["messages"][-1]["content"]
        
        # Perform search
        raw_results = self.search_tool.invoke(query)
        
        # Process results
        if isinstance(raw_results, dict) and "organic" in raw_results:
            results = raw_results["organic"]
        else:
            results = [raw_results]
        
        # Format results
        formatted_results = "\n\n".join([
            f"Title: {result.get('title', 'No title')}\n"
            f"URL: {result.get('link', 'No URL')}\n"
            f"Snippet: {result.get('snippet', 'No snippet')}"
            for result in results
        ])
        
        # Generate response
        response = self.llm.invoke(
            self.prompt.format(
                messages=state["messages"],
                search_results=formatted_results
            )
        )
        
        # Update state
        state["messages"].append({"role": "assistant", "content": response.content})
        return state
```

### 2. Add the Search Agent to the Supervisor

```python
from langgraph.graph import StateGraph
from supervisor import Supervisor, SupervisorConfig

# Initialize search agent
search_agent = SearchAgent(api_key=os.getenv("SERPER_API_KEY"))

# Initialize supervisor
supervisor = Supervisor(
    config=SupervisorConfig(
        llm_provider="openai",
        openai_model="gpt-4o",
        system_message="""
        You are a supervisor agent that coordinates multiple specialized agents.
        You have access to the following agents:
        - search_agent: For web searches using Serper API
        """
    ),
    agents={"search_agent": search_agent}
)

# Create graph
graph = StateGraph()
# ... (add nodes and edges)
```

## API Reference

### GoogleSerperAPIWrapper

```python
class GoogleSerperAPIWrapper:
    """Wrapper for Serper.dev Google Search API."""
    
    def __init__(
        self,
        serper_api_key: str = None,
        k: int = 10,
        gl: str = "us",
        hl: str = "en",
        search_type: str = "search",
        tbs: str = None,
    ):
        """
        Initialize the wrapper.
        
        Args:
            serper_api_key: Serper API key
            k: Number of search results to return
            gl: Google location (country code)
            hl: Language for search results
            search_type: Type of search ("search", "images", "news", "places")
            tbs: Time-based search parameter
        """
        pass
    
    def run(self, query: str) -> str:
        """Run query through Serper and parse result."""
        pass
    
    def results(self, query: str, **kwargs) -> dict:
        """Run query through Serper and return the raw result."""
        pass
```

## Best Practices

1. **API Key Management**: Store your Serper API key in environment variables or a secure configuration system.

2. **Rate Limiting**: Implement rate limiting to avoid exceeding your Serper API quota.

3. **Error Handling**: Handle API errors gracefully, especially network errors and rate limit errors.

4. **Result Processing**: Process the results to extract the most relevant information for your use case.

5. **Caching**: Consider implementing a caching layer to avoid redundant API calls for the same queries.

## Resources

- [Serper API Documentation](https://serper.dev/docs)
- [LangChain GoogleSerperAPIWrapper Documentation](https://python.langchain.com/docs/integrations/tools/google_serper/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
