"""
Vector Retrieval Agent for Multi-Agent System

This module implements a vector retrieval agent that can:
- Generate embeddings for queries
- Retrieve documents from vector stores
- Provide relevant information based on semantic similarity
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
import psycopg2

# Import utility functions
# Import utility functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.caching import cache_result, MemoryCache

# Import LangChain components
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_community.vectorstores import Chroma, PGVector
except ImportError:
    print("Warning: LangChain components not available. Using mock implementations.")
    # Mock implementations for testing
    class ChatOpenAI:
        def __init__(self, model=None, temperature=0, streaming=False):
            self.model = model
            self.temperature = temperature
            self.streaming = streaming

        def invoke(self, messages):
            return {"content": f"Response from {self.model} about {messages[-1]['content']}"}

    class OpenAIEmbeddings:
        def __init__(self, model=None):
            self.model = model

        def embed_query(self, text):
            # Return a mock embedding (128-dimensional vector of random values)
            return [0.1] * 128

        def embed_documents(self, documents):
            # Return mock embeddings for each document
            return [[0.1] * 128 for _ in documents]

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    class SystemMessage:
        def __init__(self, content):
            self.content = content

    class MessagesPlaceholder:
        def __init__(self, variable_name):
            self.variable_name = variable_name

    class ChatPromptTemplate:
        @classmethod
        def from_messages(cls, messages):
            return cls()

        def format(self, **kwargs):
            return kwargs

    class Chroma:
        def __init__(self, collection_name=None, embedding_function=None, persist_directory=None):
            self.collection_name = collection_name
            self.embedding_function = embedding_function
            self.persist_directory = persist_directory

        def similarity_search_with_score(self, query, k=4):
            # Return mock results
            return [
                ({"page_content": f"Mock document about {query}", "metadata": {"source": "mock"}}, 0.9),
                ({"page_content": f"Another mock document about {query}", "metadata": {"source": "mock"}}, 0.8)
            ]

    class PGVector:
        def __init__(self, collection_name=None, embedding_function=None, connection_string=None):
            self.collection_name = collection_name
            self.embedding_function = embedding_function
            self.connection_string = connection_string

        def similarity_search_with_score(self, query, k=4):
            # Return mock results
            return [
                ({"page_content": f"Mock document about {query}", "metadata": {"source": "mock"}}, 0.9),
                ({"page_content": f"Another mock document about {query}", "metadata": {"source": "mock"}}, 0.8)
            ]


class RetrievalResult(BaseModel):
    """Model for a retrieval result."""
    query: str
    documents: List[Dict[str, Any]]
    scores: List[float]
    execution_time: float
    error: Optional[str] = None


class VectorRetrievalAgentConfig(BaseModel):
    """Configuration for the vector retrieval agent."""
    store_type: Literal["chroma", "pgvector"] = "chroma"
    collection_name: str = "default_collection"
    persist_directory: Optional[str] = "./vector_db"
    connection_string: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-opus-20240229"
    temperature: float = 0
    streaming: bool = True
    max_tokens: int = 4000
    top_k: int = 4
    # Caching configuration
    use_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    # System messages
    system_message: str = """
    You are a vector retrieval agent that can retrieve documents from a vector store
    based on semantic similarity to a query.

    Your job is to:
    1. Understand the user's question
    2. Generate embeddings for the query
    3. Retrieve relevant documents from the vector store
    4. Provide a helpful response based on the retrieved documents

    Always cite your sources and explain why the retrieved documents are relevant to the query.
    """


class VectorRetrievalAgent:
    """
    Vector retrieval agent that can retrieve documents from a vector store.
    """

    def __init__(self, config: VectorRetrievalAgentConfig = VectorRetrievalAgentConfig()):
        """
        Initialize the vector retrieval agent.

        Args:
            config: Configuration for the vector retrieval agent
        """
        self.config = config

        # Initialize cache if enabled
        if self.config.use_cache:
            self.cache = MemoryCache(ttl=self.config.cache_ttl)
        else:
            self.cache = None

        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=config.openai_model,
                temperature=config.temperature,
                streaming=config.streaming
            )
        except Exception as e:
            print(f"Warning: Could not initialize ChatOpenAI: {str(e)}")
            # Use a mock implementation
            class MockLLM:
                def invoke(self, messages):
                    return {"content": f"Mock response about vector retrieval"}
            self.llm = MockLLM()

        # Initialize embeddings
        try:
            self.embeddings = OpenAIEmbeddings(
                model=config.embedding_model
            )
        except Exception as e:
            print(f"Warning: Could not initialize OpenAIEmbeddings: {str(e)}")
            self.embeddings = OpenAIEmbeddings()

        # Initialize vector store
        self.vector_store = self._initialize_vector_store()

        # Create prompt template
        self.response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=config.system_message),
            MessagesPlaceholder(variable_name="messages"),
            SystemMessage(content="Retrieved documents: {documents}")
        ])

    def _initialize_vector_store(self):
        """
        Initialize the vector store.

        Returns:
            Vector store
        """
        try:
            if self.config.store_type == "chroma":
                return Chroma(
                    collection_name=self.config.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.config.persist_directory
                )
            elif self.config.store_type == "pgvector":
                if not self.config.connection_string:
                    # Use default connection string
                    self.config.connection_string = "postgresql://postgres:102938@localhost:5432/langgraph_agent_db"

                # Check if pgvector extension is available
                try:
                    conn = psycopg2.connect(self.config.connection_string)
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                    pgvector_available = cursor.fetchone() is not None
                    conn.close()

                    if not pgvector_available:
                        print("Warning: pgvector extension not available. Using Chroma instead.")
                        return Chroma(
                            collection_name=self.config.collection_name,
                            embedding_function=self.embeddings,
                            persist_directory=self.config.persist_directory
                        )
                except Exception as e:
                    print(f"Warning: Could not check pgvector availability: {str(e)}")
                    print("Using Chroma instead.")
                    return Chroma(
                        collection_name=self.config.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=self.config.persist_directory
                    )

                return PGVector(
                    collection_name=self.config.collection_name,
                    embedding_function=self.embeddings,
                    connection_string=self.config.connection_string
                )
            else:
                raise ValueError(f"Unsupported vector store type: {self.config.store_type}")
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            # Return a mock vector store
            return Chroma(
                collection_name="mock_collection",
                embedding_function=self.embeddings,
                persist_directory="./mock_vector_db"
            )

    def _retrieve_documents(self, query: str) -> RetrievalResult:
        """
        Retrieve documents from the vector store.

        Args:
            query: Query to retrieve documents for

        Returns:
            RetrievalResult: Retrieval result
        """
        # Use cache if enabled
        if self.cache:
            cache_key = f"vector_retrieval:{query}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result

        try:
            # Retrieve documents
            start_time = time.time()
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=self.config.top_k
            )
            execution_time = time.time() - start_time

            # Parse results
            documents = []
            scores = []
            for doc, score in results:
                documents.append({
                    "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {}
                })
                scores.append(float(score))

            result = RetrievalResult(
                query=query,
                documents=documents,
                scores=scores,
                execution_time=execution_time
            )

            # Cache the result if caching is enabled
            if self.cache:
                self.cache.set(cache_key, result)

            return result

        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return RetrievalResult(
                query=query,
                documents=[],
                scores=[],
                execution_time=0,
                error=str(e)
            )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a state update in the multi-agent system.

        Args:
            state: Current state of the system

        Returns:
            Dict[str, Any]: Updated state
        """
        # Extract the message from the last message
        message = state["messages"][-1]["content"]

        # Retrieve documents
        retrieval_result = self._retrieve_documents(message)

        # Format documents for the LLM
        if retrieval_result.error:
            formatted_documents = f"Error: {retrieval_result.error}"
        else:
            formatted_documents = json.dumps(
                [
                    {
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": score
                    }
                    for doc, score in zip(retrieval_result.documents, retrieval_result.scores)
                ],
                indent=2
            )

        # Generate response using LLM
        response = self.llm.invoke(
            self.response_prompt.format(
                messages=state["messages"],
                documents=formatted_documents
            )
        )

        # Update state
        state["agent_outputs"]["vector_retrieval"] = {
            "query": retrieval_result.query,
            "documents": retrieval_result.documents,
            "scores": retrieval_result.scores,
            "execution_time": retrieval_result.execution_time,
            "error": retrieval_result.error
        }

        state["messages"].append({"role": "assistant", "content": response.content})

        return state


# Example usage
if __name__ == "__main__":
    # Create vector retrieval agent
    vector_retrieval_agent = VectorRetrievalAgent(
        config=VectorRetrievalAgentConfig(
            store_type="chroma",
            collection_name="test_collection",
            persist_directory="./test_vector_db"
        )
    )

    # Test with a query
    state = {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "agent_outputs": {}
    }

    updated_state = vector_retrieval_agent(state)
    print(updated_state["messages"][-1]["content"])
