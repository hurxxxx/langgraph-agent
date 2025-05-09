# Specialized Agents Documentation (as of May 2025)

This document provides information about the specialized agents that will be implemented in our multi-agent supervisor system, including the latest APIs and libraries for each agent type.

## Search Agents

### Tavily
- **Latest API Version**: v1 (as of May 2025)
- **Key Features**: 
  - Advanced search capabilities
  - Context-aware results
  - Structured data extraction
- **Implementation**:
```python
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(
    api_key="your-tavily-api-key",
    max_results=5,
    include_raw_content=True,
    include_domains=["domain1.com", "domain2.com"],  # Optional
    exclude_domains=["exclude.com"],  # Optional
)
```

### Google Search
- **Latest API Version**: Custom Search JSON API v1 (as of May 2025)
- **Implementation**:
```python
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool

search = GoogleSearchAPIWrapper()
google_search_tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run
)
```

### DuckDuckGo
- **Latest API Version**: No official API, using community wrapper
- **Implementation**:
```python
from langchain_community.tools import DuckDuckGoSearchRun

ddg_search = DuckDuckGoSearchRun()
```

## SQL RAG Agents

### PostgreSQL
- **Latest Integration**: LangChain SQL Database Chain (as of May 2025)
- **Implementation**:
```python
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri("postgresql://user:pass@localhost:5432/db")
sql_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
```

### SQLite
- **Implementation**:
```python
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///path/to/db.sqlite")
sqlite_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
```

## Vector Storage Agents

### Chroma
- **Latest Version**: 0.5.x (as of May 2025)
- **Implementation**:
```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
chroma_db = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

### Qdrant
- **Latest Version**: 1.8.x (as of May 2025)
- **Implementation**:
```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
qdrant = Qdrant(
    collection_name="my_collection",
    embedding_function=embeddings,
    url="http://localhost:6333",
    api_key="qdrant-api-key"  # Optional
)
```

### Milvus
- **Latest Version**: 2.3.x (as of May 2025)
- **Implementation**:
```python
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
milvus_db = Milvus(
    collection_name="my_collection",
    embedding_function=embeddings,
    connection_args={"host": "localhost", "port": "19530"}
)
```

### pgvector
- **Latest Version**: 0.6.x (as of May 2025)
- **Implementation**:
```python
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
connection_string = "postgresql://user:pass@localhost:5432/db"
pg_vector = PGVector(
    collection_name="my_collection",
    embedding_function=embeddings,
    connection_string=connection_string
)
```

### Meilisearch
- **Latest Version**: 1.5.x (as of May 2025)
- **Implementation**:
```python
from langchain_community.vectorstores import Meilisearch
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
meili = Meilisearch(
    embeddings=embeddings,
    url="http://localhost:7700",
    api_key="meili-api-key"
)
```

## Vector Retrieval Agent
- **Implementation**:
```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

retrieval_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=retriever
)
```

## Reranking Agent

### Cohere Rerank
- **Latest API Version**: v1 (as of May 2025)
- **Implementation**:
```python
from langchain_community.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

base_retriever = vector_db.as_retriever()
reranker = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)
```

### Pinecone Hybrid Search
- **Latest Version**: 2.x (as of May 2025)
- **Implementation**:
```python
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
import pinecone

pinecone.init(api_key="your-api-key", environment="your-environment")
index = pinecone.Index("your-index-name")

embeddings = OpenAIEmbeddings()
pinecone_store = Pinecone(
    index=index,
    embedding=embeddings,
    text_key="text"
)

hybrid_retriever = pinecone_store.as_retriever(
    search_type="hybrid",
    search_kwargs={"alpha": 0.5}  # Balance between vector and keyword search
)
```

## Image Generation Agent

### DALL-E
- **Latest API Version**: DALL-E 3 (as of May 2025)
- **Implementation**:
```python
from langchain_openai import OpenAIImages

image_generator = OpenAIImages(
    api_key="your-openai-api-key",
    model="dall-e-3"
)

image_url = image_generator.generate(
    prompt="A futuristic city with flying cars and tall skyscrapers",
    size="1024x1024",
    quality="standard",
    n=1
)[0]
```

### GPT-4o Image Generation
- **Latest API Version**: GPT-4o (as of May 2025)
- **Implementation**:
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import requests

def generate_image_with_gpt4o(prompt):
    chat = ChatOpenAI(model="gpt-4o", temperature=0)
    
    response = chat.invoke([
        SystemMessage(content="You are an image generation assistant. Generate images based on user prompts."),
        HumanMessage(content=f"Generate an image of: {prompt}")
    ])
    
    # Extract image URL from response
    # Note: This is a simplified implementation and may need to be updated
    # based on the actual API response structure
    image_url = response.content
    
    return image_url
```

## Writer Agent
- **Implementation**:
```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

template = """
You are a professional writer. Write {content_type} about {topic}.
Style: {style}
Tone: {tone}
Length: {length}

Your response:
"""

prompt = PromptTemplate(
    input_variables=["content_type", "topic", "style", "tone", "length"],
    template=template
)

writer_chain = LLMChain(
    llm=ChatOpenAI(model="gpt-4o", temperature=0.7),
    prompt=prompt
)
```

## MCP (Master Control Program) Agent
- **Implementation**:
```python
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.prompts import PromptTemplate

system_message = """
You are the Master Control Program (MCP), responsible for high-level planning and orchestration.
Your job is to break down complex tasks into subtasks and delegate them to specialized agents.
"""

prompt = PromptTemplate.from_template(system_message)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
mcp_agent = create_openai_functions_agent(llm, tools, prompt)
mcp_executor = AgentExecutor(agent=mcp_agent, tools=tools)
```

## Quality Measurement Agent
- **Implementation**:
```python
from langchain.evaluation import StringEvaluator
from langchain_openai import ChatOpenAI

evaluator = StringEvaluator.from_llm(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    criteria={
        "accuracy": "Is the information accurate and factual?",
        "relevance": "Is the response relevant to the query?",
        "completeness": "Does the response fully address all aspects of the query?",
        "clarity": "Is the response clear and easy to understand?",
        "helpfulness": "Is the response helpful for the user's needs?"
    }
)
```
