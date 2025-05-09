# LangGraph Agent Tools

This document describes the tools available in the LangGraph Agent system. Tools are specific capabilities that perform discrete tasks and are used by agents to accomplish their goals.

## Search Tools

Search tools provide the ability to search the web for information.

### Serper Search Tool

The Serper Search Tool uses the Serper API to search Google for information.

**Configuration**:
- `serper_api_key`: Serper API key (from .env file)
- `max_results`: Maximum number of results to return (default: 5)
- `time_period`: Time period for search (e.g., "1d", "1w", "1m")
- `news_only`: Whether to search only for news (default: false)
- `region`: Region for search results (e.g., "kr" for Korea)

**Usage**:
```python
from src.tools.search import SerperSearchTool

# Create the tool
search_tool = SerperSearchTool(
    serper_api_key="your-api-key",
    max_results=5,
    time_period="1d",
    news_only=True,
    region="us"
)

# Use the tool
results = search_tool.run("Latest news about AI")
```

### Tavily Search Tool

The Tavily Search Tool uses the Tavily API to search the web for information.

**Configuration**:
- `tavily_api_key`: Tavily API key (from .env file)
- `max_results`: Maximum number of results to return (default: 5)
- `search_depth`: Search depth (default: "basic")
- `topic`: Search topic (default: "general")

**Usage**:
```python
from src.tools.search import TavilySearchTool

# Create the tool
search_tool = TavilySearchTool(
    tavily_api_key="your-api-key",
    max_results=5,
    search_depth="advanced",
    topic="news"
)

# Use the tool
results = search_tool.run("Latest news about AI")
```

## Image Generation Tools

Image generation tools provide the ability to generate images from text descriptions.

### DALL-E Image Generation Tool

The DALL-E Image Generation Tool uses OpenAI's DALL-E API to generate images from text descriptions.

**Configuration**:
- `model`: DALL-E model to use (default: "dall-e-3")
- `size`: Image size (default: "1024x1024")
- `quality`: Image quality (default: "standard")
- `style`: Image style (default: None)

**Usage**:
```python
from src.tools.image_gen import DALLEImageGenerationTool

# Create the tool
image_tool = DALLEImageGenerationTool(
    model="dall-e-3",
    size="1024x1024",
    quality="standard",
    style="vivid"
)

# Use the tool
image_url = image_tool.run("A beautiful sunset over the ocean")
```

### GPT-Image Generation Tool

The GPT-Image Generation Tool uses OpenAI's GPT-Image API to generate images from text descriptions.

**Configuration**:
- `model`: GPT-Image model to use (default: "gpt-image-1")
- `size`: Image size (default: "1024x1024")
- `quality`: Image quality (default: "standard")
- `style`: Image style (default: None)

**Usage**:
```python
from src.tools.image_gen import GPTImageGenerationTool

# Create the tool
image_tool = GPTImageGenerationTool(
    model="gpt-image-1",
    size="1024x1024",
    quality="high"
)

# Use the tool
image_url = image_tool.run("A Ghibli-style image of a peaceful forest")
```

## Vector Storage Tools

Vector storage tools provide the ability to store and retrieve vector embeddings.

### Chroma Vector Storage Tool

The Chroma Vector Storage Tool uses Chroma to store and retrieve vector embeddings.

**Configuration**:
- `collection_name`: Name of the collection (default: "default")
- `embedding_function`: Embedding function to use (default: OpenAIEmbeddings)

**Usage**:
```python
from src.tools.vector_store import ChromaVectorStoreTool
from langchain_openai import OpenAIEmbeddings

# Create the tool
vector_tool = ChromaVectorStoreTool(
    collection_name="my_collection",
    embedding_function=OpenAIEmbeddings()
)

# Use the tool
vector_tool.add_texts(["Text 1", "Text 2", "Text 3"])
results = vector_tool.search("Query text", k=3)
```

## SQL Database Tools

SQL database tools provide the ability to interact with SQL databases.

### PostgreSQL Tool

The PostgreSQL Tool provides the ability to interact with PostgreSQL databases.

**Configuration**:
- `connection_string`: PostgreSQL connection string
- `schema`: Database schema to use

**Usage**:
```python
from src.tools.sql_db import PostgreSQLTool

# Create the tool
sql_tool = PostgreSQLTool(
    connection_string="postgresql://user:password@localhost:5432/db",
    schema="public"
)

# Use the tool
results = sql_tool.run("SELECT * FROM users LIMIT 10")
```

## Document Generation Tools

Document generation tools provide the ability to generate documents such as reports, blog posts, and proposals.

### Report Generation Tool

The Report Generation Tool generates formal reports.

**Usage**:
```python
from src.tools.document_gen import ReportGenerationTool

# Create the tool
report_tool = ReportGenerationTool()

# Use the tool
report = report_tool.run("Generate a report about renewable energy trends")
```

### Blog Post Generation Tool

The Blog Post Generation Tool generates blog posts.

**Usage**:
```python
from src.tools.document_gen import BlogPostGenerationTool

# Create the tool
blog_tool = BlogPostGenerationTool()

# Use the tool
blog_post = blog_tool.run("Generate a blog post about AI trends")
```
