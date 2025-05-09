# Specialized Agents Documentation (as of May 2025)

This document provides information about the specialized agents implemented in our multi-agent supervisor system.

## Search Agents

### Tavily
```python
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(
    api_key="your-tavily-api-key",
    max_results=5,
    include_raw_content=True
)
```

### Google Search
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
```python
from langchain_community.tools import DuckDuckGoSearchRun

ddg_search = DuckDuckGoSearchRun()
```

## SQL RAG Agents

### PostgreSQL
```python
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri("postgresql://user:pass@localhost:5432/db")
sql_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
```

### SQLite
```python
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///path/to/db.sqlite")
sqlite_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
```

## Vector Storage Agents

### Chroma
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
```python
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
qdrant = Qdrant(
    collection_name="my_collection",
    embedding_function=embeddings,
    url="http://localhost:6333"
)
```

### pgvector
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

## Vector Retrieval Agent
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

## Image Generation Agent

### DALL-E
```python
from langchain_openai import OpenAIImages

image_generator = OpenAIImages(
    model="dall-e-3"
)

image_url = image_generator.generate(
    prompt="A futuristic city with flying cars",
    size="1024x1024",
    quality="standard",
    n=1
)[0]
```

### GPT-4o Image Generation
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def generate_image_with_gpt4o(prompt):
    chat = ChatOpenAI(model="gpt-4o", temperature=0)

    response = chat.invoke([
        SystemMessage(content="You are an image generation assistant."),
        HumanMessage(content=f"Generate an image of: {prompt}")
    ])

    image_url = response.content
    return image_url
```

## Writer Agent
```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

template = """
You are a professional writer. Write {content_type} about {topic}.
Style: {style}
Tone: {tone}
Length: {length}
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

## Quality Measurement Agent
```python
from langchain.evaluation import StringEvaluator
from langchain_openai import ChatOpenAI

evaluator = StringEvaluator.from_llm(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    criteria={
        "accuracy": "Is the information accurate and factual?",
        "relevance": "Is the response relevant to the query?",
        "completeness": "Does the response fully address all aspects of the query?",
        "clarity": "Is the response clear and easy to understand?"
    }
)
```
