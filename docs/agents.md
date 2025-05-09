# LangGraph Agents

This document describes the agents available in the LangGraph Agent system. Agents are LangGraph constructs that use tools to solve domain-specific problems.

## Agent Architecture

All agents in the system follow the same basic architecture:

1. They are implemented using LangGraph's `create_react_agent` function
2. They use a set of tools to accomplish their tasks
3. They follow the ReAct pattern (Reason, Act, Observe)
4. They can be invoked directly or through the supervisor

Each agent has:
- A configuration class that defines its behavior
- A set of tools that it can use
- A system message that guides its behavior
- A LangGraph that defines its execution flow

## Search Agent

The Search Agent searches the web for information using search tools like Serper and Tavily.

**Configuration**:
- `providers`: List of search providers to use (default: ["serper"])
- `llm_model`: LLM model to use (default: "gpt-4o")
- `temperature`: Temperature for the LLM (default: 0)
- `streaming`: Whether to stream the response (default: true)
- `max_results`: Maximum number of results to return (default: 5)
- `time_period`: Time period for search (default: None)
- `news_only`: Whether to search only for news (default: false)
- `region`: Region for search results (default: None)

**Usage**:
```python
from src.agents.search_agent import SearchAgent, SearchAgentConfig

# Create the agent
agent = SearchAgent(
    config=SearchAgentConfig(
        providers=["serper", "tavily"],
        max_results=5,
        time_period="1d",
        news_only=True
    )
)

# Use the agent
state = {
    "messages": [{"role": "user", "content": "What are the latest developments in AI?"}]
}
result = agent(state)
```

## Image Generation Agent

The Image Generation Agent generates images from text descriptions using image generation tools like DALL-E and GPT-Image.

**Configuration**:
- `llm_provider`: LLM provider to use (default: "openai")
- `openai_model`: OpenAI model to use (default: "gpt-4o")
- `anthropic_model`: Anthropic model to use (default: "claude-3-7-sonnet-20250219")
- `temperature`: Temperature for the LLM (default: 0)
- `streaming`: Whether to stream the response (default: true)
- `provider`: Image generation provider to use (default: "gpt-image")
- `dalle_model`: DALL-E model to use (default: "dall-e-3")
- `gpt_image_model`: GPT-Image model to use (default: "gpt-image-1")
- `image_size`: Image size (default: "1024x1024")
- `image_quality`: Image quality (default: "standard")
- `image_style`: Image style (default: None)
- `save_images`: Whether to save generated images (default: true)
- `images_dir`: Directory to save images (default: "./generated_images")

**Usage**:
```python
from src.agents.image_agent import ImageGenerationAgent, ImageGenerationAgentConfig

# Create the agent
agent = ImageGenerationAgent(
    config=ImageGenerationAgentConfig(
        provider="gpt-image",
        gpt_image_model="gpt-image-1",
        image_size="1024x1024",
        image_quality="high"
    )
)

# Use the agent
state = {
    "messages": [{"role": "user", "content": "Generate a Ghibli-style image of a peaceful forest"}]
}
result = agent(state)
```

## Report Generation Agent

The Report Generation Agent creates formal reports using document generation tools.

**Configuration**:
- `document_type`: Type of document to generate (default: "report")
- `default_sections`: Default sections for the report (default: ["Executive Summary", "Introduction", "Findings", "Analysis", "Recommendations", "Conclusion"])
- `formality_level`: Formality level for the report (default: "high")
- `include_executive_summary`: Whether to include an executive summary (default: true)
- `include_recommendations`: Whether to include recommendations (default: true)
- `include_charts`: Whether to include charts (default: false)
- `documents_dir`: Directory to save reports (default: "./generated_documents/reports")

**Usage**:
```python
from src.agents.report_agent import ReportGenerationAgent, ReportGenerationAgentConfig

# Create the agent
agent = ReportGenerationAgent(
    config=ReportGenerationAgentConfig(
        formality_level="high",
        include_executive_summary=True,
        include_recommendations=True
    )
)

# Use the agent
state = {
    "messages": [{"role": "user", "content": "Generate a report about renewable energy trends"}]
}
result = agent(state)
```

## SQL Query Agent

The SQL Query Agent executes SQL queries and retrieves data from databases.

**Configuration**:
- `connection_string`: Database connection string
- `schema`: Database schema to use
- `llm_model`: LLM model to use (default: "gpt-4o")
- `temperature`: Temperature for the LLM (default: 0)
- `streaming`: Whether to stream the response (default: true)

**Usage**:
```python
from src.agents.sql_agent import SQLQueryAgent, SQLQueryAgentConfig

# Create the agent
agent = SQLQueryAgent(
    config=SQLQueryAgentConfig(
        connection_string="postgresql://user:password@localhost:5432/db",
        schema="public"
    )
)

# Use the agent
state = {
    "messages": [{"role": "user", "content": "How many users registered in the last month?"}]
}
result = agent(state)
```

## Vector Storage Agent

The Vector Storage Agent stores and retrieves vector embeddings.

**Configuration**:
- `collection_name`: Name of the collection (default: "default")
- `embedding_model`: Embedding model to use (default: "text-embedding-3-small")
- `llm_model`: LLM model to use (default: "gpt-4o")
- `temperature`: Temperature for the LLM (default: 0)
- `streaming`: Whether to stream the response (default: true)

**Usage**:
```python
from src.agents.vector_agent import VectorStorageAgent, VectorStorageAgentConfig

# Create the agent
agent = VectorStorageAgent(
    config=VectorStorageAgentConfig(
        collection_name="my_collection",
        embedding_model="text-embedding-3-small"
    )
)

# Use the agent
state = {
    "messages": [{"role": "user", "content": "Store this information: The capital of France is Paris."}]
}
result = agent(state)
```
